/**
 * CursorMonitor: reusable browser-event-based cursor/mouse tracker.
 *
 * Tracks raw interaction events (mousemove/mousedown/click/dblclick) inside
 * a given container and derives internal-only metrics: reaction time,
 * hesitation time, click accuracy, cursor distance/speed, missed clicks,
 * double-click errors, and target selection accuracy for target-based
 * activities. None of this is rendered to the user during gameplay --
 * callers read the derived metrics only when saving a result.
 *
 * Usage for a target-based activity:
 *   const cursor = new CursorMonitor(containerEl);
 *   cursor.start();
 *   cursor.markTargetShown(targetEl); // call when a new target appears
 *   // ... user clicks the target (or elsewhere) ...
 *   cursor.recordClick(event, isCorrectTarget);
 *   // at the end:
 *   const metrics = cursor.getMetrics();
 *   cursor.stop();
 */

class CursorMonitor {
  constructor(containerEl) {
    this.containerEl = containerEl || document;
    this._lastPos = null;
    this._totalDistance = 0;
    this._moveSamples = [];
    this._targetShownAt = null;
    this._firstMoveAfterTargetAt = null;

    this._reactionTimes = [];
    this._hesitationTimes = [];
    this._correctClicks = 0;
    this._incorrectClicks = 0;
    this._missedClicks = 0;
    this._doubleClickErrors = 0;
    this._targetsShown = 0;
    this._targetsHit = 0;
    this._lastClickAt = 0;
    this._lastClickTarget = null;
    this._allClicks = 0;
    this._clickTimestamps = [];

    this._onMove = this._onMove.bind(this);
    this._onDblClick = this._onDblClick.bind(this);
    this._onAnyClick = this._onAnyClick.bind(this);
  }

  start() {
    this.containerEl.addEventListener('mousemove', this._onMove);
    this.containerEl.addEventListener('dblclick', this._onDblClick);
    this.containerEl.addEventListener('click', this._onAnyClick, true);
  }

  stop() {
    this.containerEl.removeEventListener('mousemove', this._onMove);
    this.containerEl.removeEventListener('dblclick', this._onDblClick);
    this.containerEl.removeEventListener('click', this._onAnyClick, true);
  }

  _onMove(event) {
    const pos = { x: event.clientX, y: event.clientY, t: performance.now() };
    if (this._lastPos) {
      const dx = pos.x - this._lastPos.x;
      const dy = pos.y - this._lastPos.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const dt = (pos.t - this._lastPos.t) / 1000;
      this._totalDistance += dist;
      if (dt > 0) this._moveSamples.push(dist / dt);
    }
    if (this._targetShownAt !== null && this._firstMoveAfterTargetAt === null) {
      this._firstMoveAfterTargetAt = pos.t;
    }
    this._lastPos = pos;
  }


  _onAnyClick(event) {
    this._allClicks += 1;
    this._clickTimestamps.push(performance.now());
    if (this._clickTimestamps.length > 20) this._clickTimestamps.shift();
  }

  _onDblClick() {
    this._doubleClickErrors += 1;
  }

  /** Call when a new target appears on screen. */
  markTargetShown() {
    this._targetShownAt = performance.now();
    this._firstMoveAfterTargetAt = null;
    this._targetsShown += 1;
  }

  /** Call on click/tap; isCorrectTarget is null for a click that hit nothing. */
  recordClick(isCorrectTarget) {
    const now = performance.now();

    if (this._targetShownAt !== null) {
      const reactionMs = now - this._targetShownAt;
      this._reactionTimes.push(reactionMs);

      if (this._firstMoveAfterTargetAt !== null) {
        this._hesitationTimes.push(this._firstMoveAfterTargetAt - this._targetShownAt);
      }
    }

    if (isCorrectTarget === true) {
      this._correctClicks += 1;
      this._targetsHit += 1;
    } else if (isCorrectTarget === false) {
      this._incorrectClicks += 1;
    }

    this._lastClickAt = now;
    this._targetShownAt = null;
  }

  /** Call once per activity end for any target that was shown but never clicked. */
  recordMissedTarget() {
    this._missedClicks += 1;
    this._targetShownAt = null;
  }

  getMetrics() {
    const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
    const targetSelectionAccuracy = this._targetsShown
      ? (this._targetsHit / this._targetsShown) * 100
      : null;
    const totalClicks = this._correctClicks + this._incorrectClicks;
    const clickAccuracy = totalClicks ? (this._correctClicks / totalClicks) * 100 : null;

    return {
      reaction_time: avg(this._reactionTimes),
      click_accuracy: clickAccuracy,
      correct_clicks: this._correctClicks,
      incorrect_clicks: this._incorrectClicks,
      missed_clicks: this._missedClicks,
      double_click_errors: this._doubleClickErrors,
      cursor_distance: Math.round(this._totalDistance),
      cursor_speed: avg(this._moveSamples),
      hesitation_time: avg(this._hesitationTimes),
      target_selection_accuracy: targetSelectionAccuracy,
    };
  }

  async save(gameName, sessionId) {
    const metrics = this.getMetrics();
    try {
      await fetch('/cursor/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_name: gameName, session_id: sessionId, ...metrics }),
        keepalive: true,
      });
    } catch (_error) {
      // Non-fatal: cursor metrics are supplementary.
    }
    return metrics;
  }
}

window.CursorMonitor = CursorMonitor;
