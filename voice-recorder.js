/**
 * VoiceRecorder: combines the browser's built-in Web Speech API
 * (SpeechRecognition) for transcription with a simple Web Audio
 * volume-based Voice Activity Detector (VAD) for timing.
 *
 * Why both: SpeechRecognition gives words but no reliable amplitude/pause
 * timing; a pure volume threshold alone is a poor speech detector (spec
 * explicitly warns against relying only on microphone volume). Combining
 * them gives real words (from STT) and real pause/speech-duration timing
 * (from VAD) without over-claiming precision neither technique actually
 * has on its own.
 *
 * No API key or account is needed -- SpeechRecognition is built into
 * Chrome/Edge and calls the browser vendor's own recognition service
 * transparently; this code only calls the browser object.
 */

class VoiceRecorder {
  constructor({ onResult, onError, silenceThresholdRms = 0.02, pauseGapMs = 350, lang = 'en-US' } = {}) {
    this.onResult = onResult || (() => {});
    this.onError = onError || (() => {});
    this.silenceThresholdRms = silenceThresholdRms;
    this.pauseGapMs = pauseGapMs;
    // BCP-47 locale for SpeechRecognition (e.g. 'en-US', 'ta-IN', 'kn-IN', 'hi-IN').
    this.lang = lang || 'en-US';

    this.recognition = null;
    this.audioContext = null;
    this.analyser = null;
    this.mediaStream = null;
    this.vadRafId = null;

    this.startedAt = null;
    this.finalTranscript = '';
    this._speaking = false;
    this._speechMsTotal = 0;
    this._lastVadTick = null;
    this._pauseCount = 0;
    this._pauseMsTotal = 0;
    this._pendingPauseStart = null;
    this._firstSpeechAt = null;
  }

  static isSupported() {
    return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  }

  async start() {
    if (!VoiceRecorder.isSupported()) {
      this.onError('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
      return false;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.onError('Microphone access is not supported in this browser.');
      return false;
    }

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (_error) {
      this.onError('Microphone access was denied.');
      return false;
    }

    this._setupVad();
    this._setupRecognition();

    this.startedAt = performance.now();
    this.finalTranscript = '';
    this._speaking = false;
    this._speechMsTotal = 0;
    this._pauseCount = 0;
    this._pauseMsTotal = 0;
    this._pendingPauseStart = null;
    this._lastVadTick = performance.now();
    this._firstSpeechAt = null;

    this.recognition.start();
    this._vadLoop();
    return true;
  }

  _setupRecognition() {
    const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.recognition = new SpeechRecognitionImpl();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = this.lang;

    this.recognition.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          this.finalTranscript += `${transcript} `;
        } else {
          interim += transcript;
        }
      }
      this.onResult({ finalText: this.finalTranscript.trim(), interimText: interim.trim() });
    };

    this.recognition.onerror = (event) => {
      if (event.error === 'no-speech') return; // not fatal, keep listening
      this.onError(`Speech recognition error: ${event.error}`);
    };

    // Auto-restart if the browser stops recognition early while we're still recording.
    this.recognition.onend = () => {
      if (this._shouldBeRunning) {
        try {
          this.recognition.start();
        } catch (_error) {
          /* ignore */
        }
      }
    };
    this._shouldBeRunning = true;
  }

  _setupVad() {
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 512;
    source.connect(this.analyser);
  }

  _vadLoop() {
    if (!this.analyser) return;
    const buffer = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(buffer);
    let sumSquares = 0;
    for (let i = 0; i < buffer.length; i += 1) sumSquares += buffer[i] * buffer[i];
    const rms = Math.sqrt(sumSquares / buffer.length);
    const now = performance.now();
    const deltaMs = Math.max(0, now - this._lastVadTick);
    this._lastVadTick = now;
    const isSpeaking = rms > this.silenceThresholdRms;
    if (isSpeaking && this._firstSpeechAt === null) this._firstSpeechAt = now;
    if (isSpeaking) {
      this._speechMsTotal += deltaMs;
      if (this._pendingPauseStart !== null) {
        const pauseDuration = now - this._pendingPauseStart;
        if (pauseDuration >= this.pauseGapMs) {
          this._pauseCount += 1;
          this._pauseMsTotal += pauseDuration;
        }
        this._pendingPauseStart = null;
      }
    } else if (this._speaking) {
      this._pendingPauseStart = now;
    } else if (this._pendingPauseStart === null && this._firstSpeechAt !== null) {
      this._pendingPauseStart = now;
    }
    this._speaking = isSpeaking;
    this.vadRafId = requestAnimationFrame(() => this._vadLoop());
  }

  stop() {
    this._shouldBeRunning = false;
    const finish = () => {
      if (this.vadRafId) cancelAnimationFrame(this.vadRafId);
      if (this.mediaStream) this.mediaStream.getTracks().forEach((track) => track.stop());
      if (this.audioContext) this.audioContext.close().catch(() => {});
      const durationMs = this.startedAt ? performance.now() - this.startedAt : 0;
      return {
        recognizedText: this.finalTranscript.trim(),
        responseTimeMs: this._firstSpeechAt !== null && this.startedAt !== null ? Math.max(0, Math.round(this._firstSpeechAt - this.startedAt)) : Math.round(durationMs),
        totalDurationMs: Math.round(durationMs),
        speechDurationMs: Math.round(this._speechMsTotal),
        pauseCount: this._pauseCount,
        pauseDurationMs: Math.round(this._pauseMsTotal),
      };
    };

    return new Promise((resolve) => {
      let settled = false;
      const resolveOnce = () => {
        if (settled) return;
        settled = true;
        // Give the browser a short grace period for the final SpeechRecognition
        // result event to arrive after stop(). This fixes lost/empty transcripts.
        setTimeout(() => resolve(finish()), 250);
      };
      if (this.recognition) {
        this.recognition.onend = resolveOnce;
        try { this.recognition.stop(); } catch (_error) { resolveOnce(); }
      } else {
        resolveOnce();
      }
      setTimeout(resolveOnce, 1200);
    });
  }
}

window.VoiceRecorder = VoiceRecorder;
