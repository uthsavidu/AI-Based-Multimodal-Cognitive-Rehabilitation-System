class WebcamMonitor {
  constructor({containerEl,sessionId,userName='User',initialFlagCount=0,onWarning=()=>{},onLimitReached=()=>{},flagLimit=4,
               warningMessage='A monitoring incident was recorded. Please check the monitoring status.',
               focusWarningMessage='Please Focus',
               limitMessage='The activity has been stopped because the monitoring limit was reached.'}) {
    this.containerEl=containerEl; this.sessionId=sessionId; this.userName=String(userName||'User'); this.onWarning=onWarning;
    this.onLimitReached=onLimitReached; this.flagLimit=flagLimit;
    this.warningMessage=warningMessage; this.focusWarningMessage=focusWarningMessage; this.limitMessage=limitMessage;
    this.video=null; this.stream=null; this.landmarker=null; this.rafId=null; this.running=false;
    this.counts=this._freshCounts();
    // Flag only the requested conditions. A normal blink/head turn is not a flag by itself --
    // sustained left/right head turns instead accumulate into the warning counters below.
    this.noFaceFlagMs=5000;
    this.multipleFaceFlagMs=2000;
    this.eyesClosedFlagMs=10000; // eyes continuously closed for 10 seconds
    this._noFaceStreakMs=0; this._multiFaceStreakMs=0; this._eyesClosedStreakMs=0; this._lookingAwayStreakMs=0;
    this._noFaceFlaggedThisEpisode=false; this._multiFaceFlaggedThisEpisode=false; this._eyesClosedFlaggedThisEpisode=false;
    this._headTurnStreakMs=0; this._headTurnDirection=null;
    this._lastBlinkState=false; this._yawnCooldownUntil=0; this._lastFrameTime=null; this._startedAt=null;
    this.flagCount=Math.max(0,Number(initialFlagCount)||0); this._stopped=false; this._stopPromise=null; this._terminating=false;
    this._poseSamples=[]; this._gazeSamples=[];
    this._expressionCounts={neutral:0,happy:0,sad:0,angry:0,surprised:0,fearful:0,disgusted:0};
    // Left/right "Please Focus" warning system (max 5 per side; a flag is only
    // generated once BOTH sides have accumulated 5 warnings each -- 10 total).
    this.focusWarnThresholdMs=1200;
    this.focusWarnLimitPerSide=5;
    this.leftWarningCount=0; this.rightWarningCount=0;
    this._focusWarnFlaggedThisEpisode=false;
    this.faceStatus='DETECTING';
  }
  _freshCounts(){return {
    face_missing_events:0,multiple_face_events:0,blink_count:0,long_eye_closure_events:0,looking_away_events:0,
    head_turned_left_events:0,head_turned_right_events:0,
    yawn_events:0,distraction_events:0,attention_events:0,total_flags:0,monitoring_seconds:0,
    face_check_attempts:0,face_verified_checks:0,face_mismatch_events:0,
    eyes_open_ms:0,eyes_closed_ms:0,gaze_center_count:0,gaze_left_count:0,gaze_right_count:0,gaze_away_count:0,
    expression_neutral:0,expression_happy:0,expression_sad:0,expression_angry:0,expression_surprised:0,
    expression_fearful:0,expression_disgusted:0,pitch_sum:0,yaw_sum:0,roll_sum:0,pose_samples:0
  };}
  async _ensureLandmarker(){
    if(this.landmarker)return;
    const vision=await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
    const resolver=await vision.FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
    this.landmarker=await vision.FaceLandmarker.createFromOptions(resolver,{baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',delegate:'GPU'},outputFaceBlendshapes:true,outputFacialTransformationMatrixes:true,runningMode:'VIDEO',numFaces:2});
  }
  _buildPanel(){
    this.containerEl.innerHTML=''; this.containerEl.classList.add('webcam-side-panel');
    const label=document.createElement('div'); label.className='webcam-side-panel-label'; label.textContent='Live Webcam Monitoring';
    const owner=document.createElement('div'); owner.className='webcam-side-panel-owner'; owner.textContent=`Logged-in user: ${this.userName}`;
    this.videoWrap=document.createElement('div'); this.videoWrap.className='webcam-video-wrap';
    this.video=document.createElement('video'); this.video.autoplay=true; this.video.muted=true; this.video.playsInline=true; this.video.className='webcam-side-panel-video';
    this.boxCanvas=document.createElement('canvas'); this.boxCanvas.className='webcam-face-boxes'; this.boxCanvas.setAttribute('aria-label','Face bounding boxes');
    this.videoWrap.append(this.video,this.boxCanvas);
    this.infoEl=document.createElement('div'); this.infoEl.className='webcam-side-panel-info';
    this.detailEl=document.createElement('div'); this.detailEl.className='webcam-side-panel-detail';
    this.detailEl.textContent='Gaze, eye and head-turn status are informational. A flag is recorded only for configured monitoring incidents.';
    this.warningEl=document.createElement('div'); this.warningEl.className='webcam-side-panel-warning'; this.warningEl.hidden=true;
    this.containerEl.append(label,owner,this.videoWrap,this.infoEl,this.detailEl,this.warningEl);
    this._setFaceStatus('DETECTING');
  }
  async start(){
    if(this.running)return;
    this._buildPanel();
    if(!navigator.mediaDevices?.getUserMedia){this._showWarning('Camera is not available in this browser.');return;}
    try{this.stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});}catch(_e){this._showWarning('Camera access was denied.');return;}
    this.video.srcObject=this.stream; await this.video.play().catch(()=>{});
    try{await this._ensureLandmarker();}catch(e){this.landmarker=null; this._setFaceStatus('MONITORING UNAVAILABLE'); if(this.detailEl)this.detailEl.textContent=`Face landmark model unavailable: ${e?.message||'load failed'}`;}
    this.running=true; this._stopped=false; this._setFaceStatus('DETECTING'); this._startedAt=performance.now(); this._lastFrameTime=this._startedAt; this._loop();
  }
  _showWarning(message){if(!this.warningEl)return;this.warningEl.textContent=message;this.warningEl.hidden=false;clearTimeout(this._warningTimeout);this._warningTimeout=setTimeout(()=>{this.warningEl.hidden=true;},4500);}
  async _raiseFlag(eventType){
    if(this.flagCount>=this.flagLimit)return;
    this.flagCount++; this.counts.total_flags++;
    const limit=this.flagCount>=this.flagLimit; const msg=limit?this.limitMessage:this.warningMessage;
    this._showWarning(msg); this._updateInfo(this.faceStatus,this.faceStatus==='FACE DETECTED');
    this._logEvent(eventType,limit?'limit':'warning',this.flagCount,msg);
    if(limit){try{this.onLimitReached(msg);}catch(_e){};setTimeout(()=>this._terminateSessionAndReturnHome(),250);} else this.onWarning(msg);
  }
  // Looking left/right is never an immediate flag -- it only accumulates a
  // per-side warning counter (max 5 each). A flag is raised, once, only
  // after BOTH sides have reached 5 (10 warnings total), and the counters
  // then reset so the next 10-warning cycle can be tracked independently.
  _raiseFocusWarning(direction){
    if(direction==='left'){this.leftWarningCount=Math.min(this.focusWarnLimitPerSide,this.leftWarningCount+1);this.counts.head_turned_left_events++;}
    else{this.rightWarningCount=Math.min(this.focusWarnLimitPerSide,this.rightWarningCount+1);this.counts.head_turned_right_events++;}
    this._showWarning(this.focusWarningMessage);
    if(this.detailEl)this.detailEl.textContent=`Please Focus -- left warnings ${this.leftWarningCount}/${this.focusWarnLimitPerSide} · right warnings ${this.rightWarningCount}/${this.focusWarnLimitPerSide}`;
    if(this.leftWarningCount>=this.focusWarnLimitPerSide && this.rightWarningCount>=this.focusWarnLimitPerSide){
      this.leftWarningCount=0; this.rightWarningCount=0;
      this._raiseFlag('left_right_warning_threshold');
    }
  }
  async _terminateSessionAndReturnHome(){
    if(this._terminating)return; this._terminating=true;
    // Save the final monitoring summary while the session is still in_progress;
    // otherwise the summary endpoint correctly rejects the already-terminated session.
    try{await this.stop();}catch(e){console.error('Could not save final webcam summary before termination:',e);}
    try{if(this.sessionId)await fetch(`/session/${encodeURIComponent(this.sessionId)}/terminate`,{method:'POST'});}catch(e){console.error('Could not terminate monitoring session:',e);}
    window.location.href='/';
  }
  async _logEvent(eventType,severity,flagNumber,message){try{await fetch('/monitoring/event',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:this.sessionId,event_type:eventType,severity,flag_number:flagNumber,message})});}catch(_e){}}
  _setFaceStatus(status){this.faceStatus=status;this._updateInfo(status,status==='FACE DETECTED');}
  _updateInfo(faceText,ok){
    if(!this.infoEl)return;
    this.infoEl.replaceChildren();
    const status=document.createElement('span'); status.className=ok?'webcam-face-ok':'webcam-face-bad'; status.textContent=faceText;
    const flags=document.createElement('span'); flags.textContent=`Flags: ${this.flagCount}/${this.flagLimit}`;
    this.infoEl.append(status,flags);
  }
  _drawFaceBoxes(faces){
    if(!this.boxCanvas||!this.video)return;
    const w=this.video.videoWidth||this.video.clientWidth||1, h=this.video.videoHeight||this.video.clientHeight||1;
    this.boxCanvas.width=w; this.boxCanvas.height=h;
    const ctx=this.boxCanvas.getContext('2d'); if(!ctx)return; ctx.clearRect(0,0,w,h);
    (faces||[]).forEach(landmarks=>{
      if(!landmarks||!landmarks.length)return;
      const xs=landmarks.map(p=>p.x*w), ys=landmarks.map(p=>p.y*h);
      const pad=Math.max(4,Math.min(w,h)*0.012);
      const x=Math.max(0,Math.min(...xs)-pad), y=Math.max(0,Math.min(...ys)-pad);
      const right=Math.min(w,Math.max(...xs)+pad), bottom=Math.min(h,Math.max(...ys)+pad);
      ctx.strokeStyle='#0ea5e9'; ctx.lineWidth=Math.max(2,w/320);
      ctx.strokeRect(x,y,right-x,bottom-y);
    });
  }
  _blendshape(result,name){const cats=result?.faceBlendshapes?.[0]?.categories||[];const c=cats.find(x=>x.categoryName===name);return c?Number(c.score):0;}
  _eulerFromMatrix(matrix){
    const d=matrix?.data||matrix; if(!d||d.length<16)return null;
    // MediaPipe uses a column-major 4x4 facial transform. Convert to Euler angles.
    const r00=d[0],r10=d[1],r20=d[2],r21=d[6],r22=d[10],r01=d[4],r02=d[8];
    const pitch=Math.atan2(-r20,Math.sqrt(r21*r21+r22*r22))*180/Math.PI;
    const yaw=Math.atan2(r10,r00)*180/Math.PI;
    const roll=Math.atan2(r21,r22)*180/Math.PI;
    return {pitch,yaw,roll};
  }
  _processResult(result,deltaMs){
    const faces=result?.faceLandmarks||[]; this._lastFaces=faces;
    const dt=Math.max(0,deltaMs||0);
    if(!faces.length){
      this._noFaceStreakMs+=dt;
      this._multiFaceStreakMs=0; this._multiFaceFlaggedThisEpisode=false;
      this._eyesClosedStreakMs=0; this._eyesClosedFlaggedThisEpisode=false;
      this._lookingAwayStreakMs=0; this._headTurnStreakMs=0; this._headTurnDirection=null; this._focusWarnFlaggedThisEpisode=false;
      if(!this._noFaceFlaggedThisEpisode && this._noFaceStreakMs>=this.noFaceFlagMs){
        this._noFaceFlaggedThisEpisode=true; this.counts.face_missing_events++;
        this._raiseFlag('no_face_detected');
      }
      this._drawFaceBoxes([]); this._setFaceStatus('NO FACE DETECTED');
      if(this.detailEl)this.detailEl.textContent=`No face detected for ${Math.round(this._noFaceStreakMs/1000)}s (flag after ${this.noFaceFlagMs/1000}s)`;
      return;
    }
    this._noFaceStreakMs=0; this._noFaceFlaggedThisEpisode=false;
    if(faces.length>1){
      this._multiFaceStreakMs+=dt;
      if(!this._multiFaceFlaggedThisEpisode && this._multiFaceStreakMs>=this.multipleFaceFlagMs){
        this._multiFaceFlaggedThisEpisode=true; this.counts.multiple_face_events++;
        this._raiseFlag('multiple_faces');
      }
      this._eyesClosedStreakMs=0; this._eyesClosedFlaggedThisEpisode=false;
      this._headTurnStreakMs=0; this._headTurnDirection=null; this._focusWarnFlaggedThisEpisode=false;
      this._drawFaceBoxes(faces); this._setFaceStatus('MULTIPLE FACES DETECTED');
      if(this.detailEl)this.detailEl.textContent=`Multiple faces for ${Math.round(this._multiFaceStreakMs/1000)}s (flag after ${this.multipleFaceFlagMs/1000}s)`;
      return;
    }
    this._multiFaceStreakMs=0; this._multiFaceFlaggedThisEpisode=false;
    this._setFaceStatus('FACE DETECTED');
    const blinkL=this._blendshape(result,'eyeBlinkLeft'),blinkR=this._blendshape(result,'eyeBlinkRight');
    const blink=((blinkL+blinkR)/2)>0.55;
    if(blink&&!this._lastBlinkState)this.counts.blink_count++;
    this._lastBlinkState=blink;
    const eyeClosed=blinkL>0.55&&blinkR>0.55;
    if(eyeClosed){
      this.counts.eyes_closed_ms+=dt; this._eyesClosedStreakMs+=dt;
      if(this._eyesClosedStreakMs>=1200&&this._eyesClosedStreakMs-dt<1200)this.counts.long_eye_closure_events++;
      if(!this._eyesClosedFlaggedThisEpisode && this._eyesClosedStreakMs>=this.eyesClosedFlagMs){
        this._eyesClosedFlaggedThisEpisode=true; this._raiseFlag('eyes_closed_10_seconds');
      }
    }else{
      this.counts.eyes_open_ms+=dt; this._eyesClosedStreakMs=0; this._eyesClosedFlaggedThisEpisode=false;
    }
    const jaw=this._blendshape(result,'jawOpen');
    if(jaw>0.55&&performance.now()>this._yawnCooldownUntil){this.counts.yawn_events++;this._yawnCooldownUntil=performance.now()+5000;}
    const pose=this._eulerFromMatrix(result?.facialTransformationMatrixes?.[0]);
    if(pose){this.counts.pitch_sum+=pose.pitch;this.counts.yaw_sum+=pose.yaw;this.counts.roll_sum+=pose.roll;this.counts.pose_samples++;}
    const left=(this._blendshape(result,'eyeLookInLeft')+this._blendshape(result,'eyeLookOutRight'))/2;
    const right=(this._blendshape(result,'eyeLookOutLeft')+this._blendshape(result,'eyeLookInRight'))/2;
    const yawSigned=pose?.yaw||0;
    const yaw=Math.abs(yawSigned);
    let gaze='center';
    if(yaw>22||Math.max(left,right)>0.58)gaze='away';
    else if(left>right+0.12)gaze='left'; else if(right>left+0.12)gaze='right';
    if(gaze==='center')this.counts.gaze_center_count++; if(gaze==='left')this.counts.gaze_left_count++; if(gaze==='right')this.counts.gaze_right_count++; if(gaze==='away')this.counts.gaze_away_count++;
    const lookingAway=gaze==='away';
    if(lookingAway)this._lookingAwayStreakMs+=dt; else this._lookingAwayStreakMs=0;
    if(this._lookingAwayStreakMs>=1800&&this._lookingAwayStreakMs-dt<1800)this.counts.looking_away_events++;
    // Sustained left/right head turn -> "Please Focus" warning (never an
    // immediate flag). A new episode starts whenever the turn direction
    // changes or the head returns toward center.
    const direction=yawSigned>0?'right':'left';
    if(yaw>25){
      if(this._headTurnDirection!==direction){this._headTurnDirection=direction;this._headTurnStreakMs=0;this._focusWarnFlaggedThisEpisode=false;}
      this._headTurnStreakMs+=dt;
      if(!this._focusWarnFlaggedThisEpisode && this._headTurnStreakMs>=this.focusWarnThresholdMs){
        this._focusWarnFlaggedThisEpisode=true;
        this._raiseFocusWarning(direction);
      }
    }else{
      this._headTurnStreakMs=0; this._headTurnDirection=null; this._focusWarnFlaggedThisEpisode=false;
    }
    const expr=this._classifyExpression(result);this.counts['expression_'+expr]++;
    // Deterministic event counts; never use random increments for reported metrics.
    if(lookingAway && !this._wasLookingAway)this.counts.distraction_events++;
    if(!lookingAway && this._wasLookingAway)this.counts.attention_events++;
    this._wasLookingAway=lookingAway;
    this._drawFaceBoxes(faces);
    if(this.detailEl && !(yaw>25)){
      const holdInfo=eyeClosed?` · Eyes closed ${Math.round(this._eyesClosedStreakMs/1000)}s/${this.eyesClosedFlagMs/1000}s`:'';
      this.detailEl.textContent=`Gaze: ${gaze} · Eyes: ${eyeClosed?'closed':'open'} · Head pose: ${pose?`${pose.pitch.toFixed(0)}° / ${pose.yaw.toFixed(0)}° / ${pose.roll.toFixed(0)}°`:'checking'}${holdInfo} · Focus warnings L ${this.leftWarningCount}/${this.focusWarnLimitPerSide} · R ${this.rightWarningCount}/${this.focusWarnLimitPerSide}`;
    }
  }
  _classifyExpression(result){
    const names=['mouthSmileLeft','mouthSmileRight','browInnerUp','browDownLeft','browDownRight','mouthFrownLeft','mouthFrownRight','eyeWideLeft','eyeWideRight','eyeSquintLeft','eyeSquintRight','mouthPucker'];
    const v=n=>this._blendshape(result,n);
    if(v('mouthSmileLeft')+v('mouthSmileRight')>0.9)return'happy';
    if(v('mouthFrownLeft')+v('mouthFrownRight')>0.8)return'sad';
    if(v('browDownLeft')+v('browDownRight')>0.9&&v('eyeSquintLeft')+v('eyeSquintRight')>0.5)return'angry';
    if(v('eyeWideLeft')+v('eyeWideRight')>1.0&&v('browInnerUp')>0.45)return'surprised';
    if(v('mouthPucker')>0.6)return'disgusted';
    return'neutral';
  }
  _loop(){
    if(!this.running)return;const now=performance.now();const dt=now-(this._lastFrameTime||now);this._lastFrameTime=now;
    if(this.video?.readyState>=2){try{if(this.landmarker){const result=this.landmarker.detectForVideo(this.video,now);this._processResult(result,dt);}}catch(e){this._setFaceStatus('MONITORING UNAVAILABLE');if(this.detailEl)this.detailEl.textContent=`Webcam monitoring error: ${e?.message||'unknown error'}`;}}
    this.rafId=requestAnimationFrame(()=>this._loop());
  }
  async _saveSummary(){
    if(!this.sessionId)return;
    const elapsed=this._startedAt?Math.round((performance.now()-this._startedAt)/1000):0;this.counts.monitoring_seconds=elapsed;
    const c=this.counts; c.pitch_avg=c.pose_samples?c.pitch_sum/c.pose_samples:null;c.yaw_avg=c.pose_samples?c.yaw_sum/c.pose_samples:null;c.roll_avg=c.pose_samples?c.roll_sum/c.pose_samples:null;
    try{const response=await fetch('/monitoring/webcam-summary',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:this.sessionId,...c})});if(!response.ok){const body=await response.text().catch(()=> '');console.error(`Webcam summary save failed (${response.status}):`,body);}}catch(error){console.error('Webcam summary request failed:',error);}
  }
  getSummary(){const c={...this.counts};c.monitoring_seconds=this._startedAt?Math.round((performance.now()-this._startedAt)/1000):0;c.pitch_avg=c.pose_samples?c.pitch_sum/c.pose_samples:null;c.yaw_avg=c.pose_samples?c.yaw_sum/c.pose_samples:null;c.roll_avg=c.pose_samples?c.roll_sum/c.pose_samples:null;return c;}
  async stop(){if(this._stopPromise)return this._stopPromise;this._stopped=true;this.running=false;if(this.rafId)cancelAnimationFrame(this.rafId);this._stopPromise=(async()=>{await this._saveSummary();if(this.stream)this.stream.getTracks().forEach(t=>t.stop());if(this.video)this.video.srcObject=null;})();return this._stopPromise;}
}
window.WebcamMonitor=WebcamMonitor;
