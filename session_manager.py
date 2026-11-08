"""Assessment-session lifecycle helpers.

One assessment_id/session_id is generated when the user starts the assessment and
is carried through all four games, webcam monitoring, cursor monitoring, and
the final results report.

Session duration defaults to 5 minutes but is configurable (see
SESSION_DURATION_SECONDS) so it can later be changed to 5-10 minutes
without touching the flow logic.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

# Configurable session length. Individual activity pacing is a UI-side
# concern (instruction screens, Level 3's fixed 5s/30s timing); this value
# is the overall assessment-session target used for display/instructions only.
# Raised from 5 to 20 minutes so a session has enough time to run all four
# activities (memory test, reaction test, memory match game, voice game)
# back to back, plus the 10-second monitoring hold times.
SESSION_DURATION_SECONDS = 20 * 60
SESSION_WARNING_SECONDS = 30

# Guided-session configuration. A session now runs ALL FOUR activities in
# sequence (previously this dict mapped every activity to itself, which
# meant a session only ever repeated the single "weakest area" game and
# never actually reached the other three -- that was the root cause of
# reports only ever showing one game's data).
#
# The user's weakest measured domain is still used -- it decides which
# activity goes FIRST, so the person practices their weakest area while
# freshest, but every session still completes all four activities before
# the report is generated.
ORDERED_ACTIVITIES = ["memory_test", "reaction_test", "memory_game", "voice_game"]

ACTIVITY_ENDPOINTS = {
    "memory_test": "memory_test",
    "reaction_test": "reaction_test",
    "memory_game": "memory_game",
    "voice_game": "voice_game",
}

THERAPY_PLANS = {
    "Memory": {"activity_key":"memory_test", "activity_name":"Memory Test", "focus":"Recall accuracy and longer sequences", "steps":["Practice short sequences first.", "Repeat with longer sequences when accuracy is steady.", "Try to recall the sequence in the same order."]},
    "Reaction": {"activity_key":"reaction_test", "activity_name":"Reaction Test", "focus":"Faster responses with accuracy", "steps":["Wait for the target to appear.", "Respond quickly without anticipating.", "Aim for consistent response times across trials."]},
    "Voice": {"activity_key":"voice_game", "activity_name":"Voice Cognitive Game", "focus":"Verbal recall, sentence recall and word retrieval", "steps":["Recall a single word clearly.", "Repeat a short sentence while keeping important words in order.", "Generate unique category words while reducing pauses and repetitions."]},
    "Cursor": {"activity_key":"memory_game", "activity_name":"Memory Match Game", "focus":"Accurate target selection", "steps":["Select targets accurately.", "Reduce unnecessary cursor movement.", "Increase speed after accuracy is consistent."]},
    "Attention": {"activity_key":"memory_test", "activity_name":"Memory Test", "focus":"Sustained focused performance", "steps":["Use a quiet environment.", "Keep your face clearly visible to the camera.", "Stay with the task without unnecessary switching."]},
}


ACTIVITY_INSTRUCTIONS = {
    "memory_test": "Remember the sequence of colors, then click the numbers in the same order.",
    "reaction_test": "Wait for the target to appear, then click it as quickly and accurately as possible.",
    "memory_game": "Remember the positions of the matching cards and select the matching pairs.",
    "voice_game": "Follow each instruction and speak clearly when recording begins.",
}


def new_session_id() -> str:
    return str(uuid4())


def create_session(db, user_id: int) -> str:
    session_id = new_session_id()
    now = datetime.now().isoformat(timespec="seconds")
    db.execute(
        """
        INSERT INTO sessions (session_id, user_id, start_time, status, created_at)
        VALUES (?, ?, ?, 'in_progress', ?)
        """,
        (session_id, user_id, now, now),
    )
    db.commit()
    return session_id


def get_session(db, session_id: str, user_id: int):
    return db.execute(
        "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
        (session_id, user_id),
    ).fetchone()


def get_therapy_plan(weakest_area: str | None) -> dict:
    return THERAPY_PLANS.get(weakest_area or "Memory", THERAPY_PLANS["Memory"])


def session_activity_order(weakest_area: str | None = None) -> list[str]:
    """Full 4-activity order for one assessment.

    The weakest-measured-domain activity is scheduled first (so the user
    trains their weakest area while freshest); the remaining three follow
    in their default order. This is computed from the session's stored
    weakest_area, so it stays fixed for the whole session.
    """
    first_key = get_therapy_plan(weakest_area)["activity_key"]
    if first_key not in ORDERED_ACTIVITIES:
        first_key = ORDERED_ACTIVITIES[0]
    return [first_key] + [key for key in ORDERED_ACTIVITIES if key != first_key]


def next_activity_endpoint(current_activity_key: str | None, weakest_area: str | None = None) -> str | None:
    """Return the next activity's Flask endpoint, or None once all 4 are done."""
    order = session_activity_order(weakest_area)
    if current_activity_key is None:
        return ACTIVITY_ENDPOINTS.get(order[0])
    if current_activity_key not in order:
        return ACTIVITY_ENDPOINTS.get(order[0])
    idx = order.index(current_activity_key)
    if idx + 1 < len(order):
        return ACTIVITY_ENDPOINTS.get(order[idx + 1])
    return None


def session_remaining_seconds(db, session_id: str, user_id: int) -> int:
    row = get_session(db, session_id, user_id)
    if not row or not row["start_time"] or row["status"] != "in_progress":
        return 0
    try:
        start = datetime.fromisoformat(row["start_time"])
        return max(0, SESSION_DURATION_SECONDS - int((datetime.now() - start).total_seconds()))
    except ValueError:
        return 0


def mark_session_complete(db, session_id: str, user_id: int, analysis: dict, status: str = "completed") -> None:
    now = datetime.now().isoformat(timespec="seconds")
    session_row = get_session(db, session_id, user_id)
    duration_seconds = None
    if session_row and session_row["start_time"]:
        try:
            start = datetime.fromisoformat(session_row["start_time"])
            duration_seconds = int((datetime.now() - start).total_seconds())
        except ValueError:
            duration_seconds = None

    db.execute(
        """
        UPDATE sessions
        SET end_time = ?, duration = ?, status = ?,
            overall_score = ?, strongest_area = ?, weakest_area = ?, recommendation = ?
        WHERE session_id = ? AND user_id = ?
        """,
        (
            now,
            duration_seconds,
            status,
            analysis.get("overall_score"),
            analysis.get("strongest_area"),
            analysis.get("weakest_area"),
            analysis.get("recommendation"),
            session_id,
            user_id,
        ),
    )
    db.commit()
