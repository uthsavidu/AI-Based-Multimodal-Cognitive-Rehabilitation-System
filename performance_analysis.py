"""Automatic session analysis: domain scores, weak/strong areas, recommendations.

All formulas here are explicit, transparent, and project-defined -- not
medical/clinical scoring. Every score is derived only from values actually
measured or estimated elsewhere in the app (see DIRECTLY MEASURED vs
MODEL-ESTIMATED vs PROJECT-DEFINED distinction below, matching what the
dashboard/report communicates to the user).

  DIRECTLY MEASURED   : reaction time, click accuracy, word counts, session duration
  MODEL-ESTIMATED     : blink/yawn/gaze/head-pose/attention (from MediaPipe landmarks, client-side)
  PROJECT-DEFINED      : every 0-100 "score" and "trend" below is a weighted index
                         this project defines for longitudinal comparison; it is not
                         a clinical or diagnostic measure.
"""

from __future__ import annotations

DOMAINS = ["Memory", "Reaction", "Voice", "Cursor", "Attention"]

# Project-defined stability band for session-to-session trend comparisons.
TREND_STABILITY_PERCENT = 5.0


def _avg(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def compute_domain_scores(db, user_id: int, session_id: str) -> dict[str, float | None]:
    """Compute the 5 domain scores (0-100) for one completed session.

    Any domain the user did not actually attempt in this session returns
    None (and is excluded from overall score / weak-strong analysis)
    rather than being silently scored as 0, which would misrepresent an
    un-attempted activity as a poor result.
    """
    scores: dict[str, float | None] = {}

    # --- Memory: average of Memory Test + Memory Match Game scores this session ---
    memory_rows = db.execute(
        """
        SELECT score FROM game_results
        WHERE user_id = ? AND session_id = ? AND game_name IN ('Memory Test', 'Memory Match Game')
        """,
        (user_id, session_id),
    ).fetchall()
    scores["Memory"] = _avg([float(r["score"]) for r in memory_rows]) if memory_rows else None

    # --- Reaction: Reaction Test score this session ---
    reaction_rows = db.execute(
        "SELECT score FROM game_results WHERE user_id = ? AND session_id = ? AND game_name = 'Reaction Test'",
        (user_id, session_id),
    ).fetchall()
    scores["Reaction"] = _avg([float(r["score"]) for r in reaction_rows]) if reaction_rows else None

    # --- Voice: average recall_accuracy / word-generation performance this session ---
    voice_rows = db.execute(
        "SELECT level, correct, recall_accuracy, valid_word_count, unique_words FROM voice_results WHERE user_id = ? AND session_id = ? ORDER BY level, round_number",
        (user_id, session_id),
    ).fetchall()
    # Voice is a composite of all three tasks, so one difficult category round
    # cannot hide successful word/sentence recall. Level 3 uses 10 unique valid
    # words as the project-defined 100-point ceiling.
    level_scores: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for row in voice_rows:
        level = int(row["level"] or 0)
        if level == 1 and row["correct"] is not None:
            level_scores[1].append(100.0 if int(row["correct"]) else 0.0)
        elif level == 2 and row["recall_accuracy"] is not None:
            level_scores[2].append(max(0.0, min(100.0, float(row["recall_accuracy"]))))
        elif level == 3:
            unique_count = float(row["unique_words"] if row["unique_words"] is not None else (row["valid_word_count"] or 0))
            level_scores[3].append(max(0.0, min(100.0, unique_count * 10.0)))
    weighted_parts = []
    weights = {1: 0.25, 2: 0.35, 3: 0.40}
    for level, weight in weights.items():
        if level_scores[level]:
            weighted_parts.append((_avg(level_scores[level]) or 0.0, weight))
    if weighted_parts:
        total_weight = sum(weight for _, weight in weighted_parts)
        scores["Voice"] = sum(score * weight for score, weight in weighted_parts) / total_weight
    else:
        scores["Voice"] = None

    # --- Cursor: average of per-game cursor interaction scores this session ---
    cursor_rows = db.execute(
        "SELECT click_accuracy, target_selection_accuracy FROM cursor_results WHERE user_id = ? AND session_id = ?",
        (user_id, session_id),
    ).fetchall()
    cursor_scores: list[float] = []
    for row in cursor_rows:
        parts = [v for v in (row["click_accuracy"], row["target_selection_accuracy"]) if v is not None]
        if parts:
            cursor_scores.append(sum(parts) / len(parts))
    scores["Cursor"] = _avg(cursor_scores) if cursor_scores else None

    # --- Attention: derived from webcam-based behavioral signal counts this session ---
    webcam_row = db.execute(
        "SELECT * FROM webcam_results WHERE user_id = ? AND session_id = ?",
        (user_id, session_id),
    ).fetchone()
    if webcam_row is not None:
        distraction_events = int(webcam_row["distraction_events"] or 0)
        looking_away_events = int(webcam_row["looking_away_events"] or 0)
        long_eye_closure_events = int(webcam_row["long_eye_closure_events"] or 0)
        face_missing_events = int(webcam_row["face_missing_events"] or 0)
        head_turned_events = int((webcam_row["head_turned_left_events"] if "head_turned_left_events" in webcam_row.keys() else 0) or 0) + int((webcam_row["head_turned_right_events"] if "head_turned_right_events" in webcam_row.keys() else 0) or 0)
        # Project-defined penalty formula: start at 100, subtract weighted
        # penalties per behavioral event, floor at 0. Not a clinical measure.
        # head_turned_* events are the 10-second-hold flags (a full, sustained
        # head turn); they get the same weight as a prolonged eye closure.
        penalty = (
            distraction_events * 6
            + looking_away_events * 4
            + long_eye_closure_events * 5
            + face_missing_events * 8
            + head_turned_events * 5
        )
        scores["Attention"] = max(0.0, min(100.0, 100.0 - penalty))
    else:
        scores["Attention"] = None

    return scores


def compute_overall_score(domain_scores: dict[str, float | None]) -> float | None:
    return _avg(list(domain_scores.values()))


def identify_strong_weak(domain_scores: dict[str, float | None]) -> tuple[str | None, str | None]:
    attempted = {k: v for k, v in domain_scores.items() if v is not None}
    if not attempted:
        return None, None
    strongest = max(attempted, key=attempted.get)
    weakest = min(attempted, key=attempted.get)
    return strongest, weakest


_RECOMMENDATION_FOCUS = {
    "Memory": "Practice the Memory Test and Memory Match Game, especially recalling longer sequences.",
    "Reaction": "Practice the Reaction Test to build faster, more consistent response times.",
    "Voice": "Practice the Voice Cognitive Game, especially rapid category word generation.",
    "Cursor": "Practice the target-selection activities to improve click accuracy and timing.",
    "Attention": "Practice attention-focused cognitive activities in a quiet, distraction-free setting.",
}


def generate_recommendation(strongest: str | None, weakest: str | None) -> str:
    if not strongest or not weakest:
        return "Complete at least one activity in a session to receive a personalized recommendation."
    if strongest == weakest:
        return f"Your performance was consistent across activities, with {strongest} as your most reliable area."

    focus = _RECOMMENDATION_FOCUS.get(weakest, "Practice a mix of activities in your next session.")
    return (
        f"Your strongest performance was in {strongest}. "
        f"Your area needing the most improvement was {weakest}. "
        f"Suggested focus for your next session: {focus}"
    )


def analyze_session(db, user_id: int, session_id: str) -> dict:
    domain_scores = compute_domain_scores(db, user_id, session_id)
    overall = compute_overall_score(domain_scores)
    strongest, weakest = identify_strong_weak(domain_scores)
    recommendation = generate_recommendation(strongest, weakest)
    return {
        "domain_scores": domain_scores,
        "overall_score": round(overall, 2) if overall is not None else None,
        "strongest_area": strongest,
        "weakest_area": weakest,
        "recommendation": recommendation,
    }



def compute_user_domain_snapshot(db, user_id: int) -> dict[str, float | None]:
    """Available 0-100 project-defined scores used to choose therapy focus."""
    scores = {}
    r=db.execute("SELECT AVG(COALESCE(memory_performance_score,score)) AS v FROM game_results WHERE user_id=? AND game_name='Memory Test'",(user_id,)).fetchone()
    mt=float(r['v']) if r and r['v'] is not None else None
    r=db.execute("SELECT AVG(score) AS v FROM game_results WHERE user_id=? AND game_name='Memory Match Game'",(user_id,)).fetchone()
    mg=float(r['v']) if r and r['v'] is not None else None
    scores['Memory']=_avg([x for x in (mt,mg) if x is not None])
    r=db.execute("SELECT AVG(score) AS v FROM game_results WHERE user_id=? AND game_name='Reaction Test'",(user_id,)).fetchone()
    scores['Reaction']=float(r['v']) if r and r['v'] is not None else None
    rows=db.execute("SELECT level,correct,recall_accuracy,valid_word_count,unique_words FROM voice_results WHERE user_id=?",(user_id,)).fetchall()
    lv={1:[],2:[],3:[]}
    for r in rows:
        l=int(r['level'] or 0)
        if l==1 and r['correct'] is not None: lv[1].append(100.0 if int(r['correct']) else 0.0)
        elif l==2 and r['recall_accuracy'] is not None: lv[2].append(float(r['recall_accuracy']))
        elif l==3: lv[3].append(min(100.0,float(r['unique_words'] if r['unique_words'] is not None else (r['valid_word_count'] or 0))*10.0))
    parts=[(_avg(lv[l]) or 0.0,w) for l,w in ((1,.25),(2,.35),(3,.40)) if lv[l]]
    scores['Voice']=sum(v*w for v,w in parts)/sum(w for _,w in parts) if parts else None
    rows=db.execute("SELECT click_accuracy,target_selection_accuracy FROM cursor_results WHERE user_id=?",(user_id,)).fetchall()
    vals=[]
    for r in rows:
        x=[v for v in (r['click_accuracy'],r['target_selection_accuracy']) if v is not None]
        if x: vals.append(sum(x)/len(x))
    scores['Cursor']=_avg(vals)
    r=db.execute("SELECT distraction_events,looking_away_events,long_eye_closure_events,face_missing_events,head_turned_left_events,head_turned_right_events FROM webcam_results WHERE user_id=? ORDER BY updated_at DESC LIMIT 1",(user_id,)).fetchone()
    if r:
        head_turned=int((r['head_turned_left_events'] or 0))+int((r['head_turned_right_events'] or 0))
        penalty=int(r['distraction_events'] or 0)*6+int(r['looking_away_events'] or 0)*4+int(r['long_eye_closure_events'] or 0)*5+int(r['face_missing_events'] or 0)*8+head_turned*5
        scores['Attention']=max(0.0,min(100.0,100.0-penalty))
    else: scores['Attention']=None
    return scores

def generate_therapy_plan(domain_scores: dict[str,float|None]) -> dict:
    attempted={k:v for k,v in domain_scores.items() if v is not None}
    weakest=min(attempted,key=attempted.get) if attempted else 'Memory'
    score=attempted.get(weakest)
    plan=THERAPY_PLANS = {
        'Memory': {'activity_key':'memory_test','activity_name':'Memory Test','focus':'recall accuracy and longer sequences','steps':['Practice short sequences first.','Repeat with longer sequences when accuracy is steady.','Focus on remembering the order before responding.']},
        'Reaction': {'activity_key':'reaction_test','activity_name':'Reaction Test','focus':'response speed while maintaining accuracy','steps':['Wait for the target instead of anticipating it.','Respond immediately when it appears.','Aim for consistent response times.']},
        'Voice': {'activity_key':'voice_game','activity_name':'Voice Cognitive Game','focus':'verbal recall, sentence recall and rapid word retrieval','steps':['Practice single-word recall clearly.','Repeat sentences with the important words in order.','Generate unique category words and gradually reduce pauses/repetitions.']},
        'Cursor': {'activity_key':'memory_game','activity_name':'Memory Match Game','focus':'accurate target selection and controlled movement','steps':['Prioritize accuracy.','Reduce unnecessary cursor movement.','Increase speed only after accuracy is stable.']},
        'Attention': {'activity_key':'memory_test','activity_name':'Memory Test','focus':'sustained focused performance','steps':['Use a quiet environment.','Keep your face visible to the camera.','Stay focused on the task.']},
    }[weakest]
    return {'weakest_area':weakest,'weakest_score':round(score,1) if score is not None else None,**plan}

def compute_session_history_trends(db, user_id: int) -> list[dict]:
    """Return every completed session with its domain scores, in order,
    plus a per-domain improving/declining/stable label versus the
    immediately preceding session (project-defined ±5% stability band).
    """
    session_rows = db.execute(
        """
        SELECT session_id, start_time, end_time, duration, status, overall_score, strongest_area, weakest_area, recommendation
        FROM sessions
        WHERE user_id = ? AND status = 'completed'
        ORDER BY start_time ASC
        """,
        (user_id,),
    ).fetchall()

    history: list[dict] = []
    previous_scores: dict[str, float] | None = None

    for row in session_rows:
        domain_scores = compute_domain_scores(db, user_id, row["session_id"])
        trend_labels: dict[str, str] = {}
        if previous_scores is not None:
            for domain in DOMAINS:
                current = domain_scores.get(domain)
                prior = previous_scores.get(domain)
                if current is None or prior is None or prior == 0:
                    trend_labels[domain] = "No prior data"
                    continue
                change_percent = ((current - prior) / prior) * 100.0
                if abs(change_percent) <= TREND_STABILITY_PERCENT:
                    trend_labels[domain] = "Stable"
                elif change_percent > 0:
                    trend_labels[domain] = "Improving"
                else:
                    trend_labels[domain] = "Declining"
        else:
            trend_labels = {domain: "Baseline session" for domain in DOMAINS}

        history.append(
            {
                "session_id": row["session_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "duration": row["duration"],
                "status": row["status"],
                "overall_score": row["overall_score"],
                "strongest_area": row["strongest_area"],
                "weakest_area": row["weakest_area"],
                "recommendation": row["recommendation"],
                "domain_scores": domain_scores,
                "trend_labels": trend_labels,
            }
        )
        previous_scores = {k: v for k, v in domain_scores.items() if v is not None} or previous_scores

    return history
