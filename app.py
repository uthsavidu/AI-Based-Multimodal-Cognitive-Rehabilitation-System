from __future__ import annotations

from datetime import datetime, timedelta
import os
import sqlite3
import json
import re
from pathlib import Path
import numpy as np
from uuid import uuid4
from random import randint, choice
from io import BytesIO

from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for, send_file
from werkzeug.security import check_password_hash, generate_password_hash

from face_verification import (
    MODEL_NAME,
    FaceEnrollmentError,
    FaceVerificationError,
    extract_registered_face,
    verify_login_face,
    verify_monitoring_faces,
)
import session_manager
import performance_analysis
import voice_analysis
import voice_content

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


def create_app() -> Flask:
    app = Flask(__name__)
    # For a real app, set this via environment variable.
    app.secret_key = "dev-secret-key-change-me"
    db_path = "rehab.db"

    # Keep public LFW data untouched. Webcam enrollment images go here.
    registered_faces_root = Path(app.root_path) / "dataset" / "registered_users"

    def save_registered_face_photo(
        user_id: int, full_name: str, image_bytes: bytes, sample_index: int
    ) -> str:
        """Save an original webcam image in a user-specific folder."""
        safe_name = re.sub(
            r"[^A-Za-z0-9_-]+", "_", (full_name or "user").strip()
        ).strip("_") or "user"
        person_folder = registered_faces_root / f"{safe_name}_{user_id}"
        person_folder.mkdir(parents=True, exist_ok=True)
        destination = person_folder / f"face_{sample_index:02d}.jpg"
        try:
            from PIL import Image
            with Image.open(BytesIO(image_bytes)) as image:
                image.convert("RGB").save(destination, format="JPEG", quality=95)
        except Exception as exc:
            raise ValueError("Could not save the captured face photo.") from exc
        return str(destination)

    @app.context_processor
    def inject_session_ui():
        sid = request.values.get("session_id") or request.view_args.get("session_id") if request.view_args else None
        remaining = 0
        ui_flag_count = 0
        if sid and session.get("user_id"):
            try:
                db = get_db()
                active = session_manager.get_session(db, sid, int(session["user_id"]))
                if active:
                    remaining = session_manager.session_remaining_seconds(db, sid, int(session["user_id"]))
                    ui_flag_count = db.execute(
                        "SELECT COUNT(*) AS n FROM monitoring_events WHERE user_id=? AND session_id=? AND severity IN ('warning','limit')",
                        (int(session["user_id"]), sid),
                    ).fetchone()["n"]
            except Exception:
                remaining = 0
                ui_flag_count = 0
        return {"ui_session_id": sid or "", "ui_remaining_seconds": remaining,
                "ui_flag_count": ui_flag_count}

    def ensure_db_schema() -> None:
        """
        Ensure the SQLite database uses the latest schema.

        IMPORTANT:
        - Older versions of this project used a users table with a required `username` column.
        - The new, real authentication system uses email + password_hash + profile fields.
        - SQLite can't drop columns, so we rebuild the users table when we detect the old schema.
        """
        if not os.path.exists(db_path):
            return

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            has_users = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='users'"
            ).fetchone()
            if not has_users:
                return

            cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]

            # Old schema detected: `username` column exists (often NOT NULL), and new columns may be missing.
            if "username" in cols:
                backup_name = f"rehab_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                # Best-effort backup of the existing DB file before modifying schema.
                try:
                    conn.close()
                    os.replace(db_path, backup_name)
                except OSError:
                    # If backup fails (file locked, permission, etc.), fall back to in-place migration.
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                else:
                    # Create a fresh DB with the new schema.
                    fresh = sqlite3.connect(db_path)
                    fresh.row_factory = sqlite3.Row
                    fresh.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS users (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          full_name TEXT NOT NULL,
                          email TEXT UNIQUE NOT NULL,
                          password_hash TEXT NOT NULL,
                          age INTEGER,
                          gender TEXT,
                          patient_type TEXT,
                          created_at TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS game_results (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id INTEGER NOT NULL,
                          game_name TEXT NOT NULL,
                          score INTEGER NOT NULL,
                          accuracy REAL NOT NULL,
                          time_taken_seconds INTEGER NOT NULL,
                          moves INTEGER NOT NULL,
                          correct_matches INTEGER NOT NULL,
                          wrong_attempts INTEGER NOT NULL,
                          avg_reaction_ms INTEGER,
                          best_reaction_ms INTEGER,
                          session_id TEXT,
                          difficulty INTEGER,
                          number_of_cards INTEGER,
                          number_of_pairs INTEGER,
                          completion_time_seconds INTEGER,
                          total_moves INTEGER,
                          successful_matches INTEGER,
                          total_pairs INTEGER,
                          move_efficiency REAL,
                          time_efficiency REAL,
                          time_per_pair REAL,
                          final_memory_match_score REAL,
                          baseline_memory_match_score REAL,
                          baseline_accuracy REAL,
                          baseline_move_efficiency REAL,
                          baseline_time_per_pair REAL,
                          session_number INTEGER,
                          change_from_baseline_percent REAL,
                          trend_status TEXT,
                          performance_level TEXT,
                          created_at TEXT NOT NULL,
                          FOREIGN KEY(user_id) REFERENCES users(id)
                        );
                        """
                    )
                    fresh.commit()
                    fresh.close()
                    return

                # In-place migration (no backup available): rebuild users table to remove username dependency.
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users_new (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      full_name TEXT NOT NULL,
                      email TEXT UNIQUE NOT NULL,
                      password_hash TEXT NOT NULL,
                      age INTEGER,
                      gender TEXT,
                      patient_type TEXT,
                      created_at TEXT NOT NULL
                    )
                    """
                )

                # Copy forward what we can. Old demo rows may not have these columns; we provide safe defaults.
                # Any users without an email are skipped (email is required in the new system).
                select_cols = set(cols)
                full_name_expr = "username" if "username" in select_cols else "email"
                created_expr = "created_at" if "created_at" in select_cols else "''"
                email_expr = "email" if "email" in select_cols else "NULL"
                password_expr = "password_hash" if "password_hash" in select_cols else "NULL"
                age_expr = "age" if "age" in select_cols else "NULL"
                gender_expr = "gender" if "gender" in select_cols else "NULL"
                patient_expr = "patient_type" if "patient_type" in select_cols else "NULL"

                # For rows that don't have a password_hash (old demo), set an unusable placeholder hash.
                placeholder_hash = generate_password_hash("change-me")

                conn.execute(
                    f"""
                    INSERT OR IGNORE INTO users_new
                      (id, full_name, email, password_hash, age, gender, patient_type, created_at)
                    SELECT
                      id,
                      COALESCE(NULLIF(TRIM({full_name_expr}), ''), {email_expr}),
                      {email_expr},
                      COALESCE(NULLIF({password_expr}, ''), ?),
                      {age_expr},
                      {gender_expr},
                      {patient_expr},
                      COALESCE(NULLIF({created_expr}, ''), ?)
                    FROM users
                    WHERE {email_expr} IS NOT NULL AND TRIM({email_expr}) <> ''
                    """,
                    (placeholder_hash, datetime.now().isoformat(timespec="seconds")),
                )

                conn.execute("DROP TABLE users")
                conn.execute("ALTER TABLE users_new RENAME TO users")
                conn.commit()
        finally:
            conn.close()

    def get_db() -> sqlite3.Connection:
        if "db" not in g:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            g.db = conn
        return g.db

    def init_db() -> None:
        db = get_db()
        ensure_db_schema()
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              full_name TEXT NOT NULL,
              email TEXT UNIQUE NOT NULL,
              password_hash TEXT NOT NULL,
              age INTEGER,
              gender TEXT,
              patient_type TEXT,
              created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS game_results (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              game_name TEXT NOT NULL,
              score REAL NOT NULL,
              accuracy REAL NOT NULL,
              time_taken_seconds INTEGER NOT NULL,
              moves INTEGER NOT NULL,
              correct_matches INTEGER NOT NULL,
              wrong_attempts INTEGER NOT NULL,
              avg_reaction_ms INTEGER,
              best_reaction_ms INTEGER,
              median_reaction_ms INTEGER,
              performance_level TEXT,
              reaction_times_json TEXT,
              reference_status TEXT,
              maximum_sequence_length INTEGER,
              maximum_level_completed INTEGER,
              correct_responses INTEGER,
              wrong_responses INTEGER,
              total_responses INTEGER,
              memory_performance_score REAL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS face_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                representation TEXT,
                embedding TEXT,
                embedding_model TEXT,
                face_image BLOB,
                face_method TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
        )
            """
            )

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                sample_index INTEGER NOT NULL,
                face_image BLOB NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, sample_index),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        # --- New tables for the cognitive rehabilitation upgrade ---
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              start_time TEXT NOT NULL,
              end_time TEXT,
              duration INTEGER,
              status TEXT NOT NULL DEFAULT 'in_progress',
              overall_score REAL,
              strongest_area TEXT,
              weakest_area TEXT,
              recommendation TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS session_activities (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              user_id INTEGER NOT NULL,
              activity_key TEXT NOT NULL,
              started_at TEXT NOT NULL,
              ended_at TEXT,
              duration_seconds INTEGER,
              status TEXT NOT NULL DEFAULT 'in_progress',
              UNIQUE(session_id, activity_key),
              FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_results (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              session_id TEXT,
              level INTEGER NOT NULL,
              round_number INTEGER NOT NULL DEFAULT 1,
              task_type TEXT,
              category TEXT,
              expected_text TEXT,
              recognized_text TEXT,
              correct INTEGER,
              correct_words INTEGER,
              incorrect_words INTEGER,
              missing_words INTEGER,
              extra_words INTEGER,
              unique_words INTEGER,
              repeated_words INTEGER,
              valid_word_count INTEGER,
              invalid_word_count INTEGER,
              response_time_ms INTEGER,
              speech_duration_ms INTEGER,
              speech_rate REAL,
              pause_count INTEGER,
              pause_duration_ms INTEGER,
              filler_count INTEGER,
              recall_accuracy REAL,
              trend TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        voice_results_cols = [r["name"] for r in db.execute("PRAGMA table_info(voice_results)").fetchall()]
        if "language" not in voice_results_cols:
            db.execute("ALTER TABLE voice_results ADD COLUMN language TEXT")

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS webcam_results (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              session_id TEXT NOT NULL UNIQUE,
              face_missing_events INTEGER NOT NULL DEFAULT 0,
              multiple_face_events INTEGER NOT NULL DEFAULT 0,
              blink_count INTEGER NOT NULL DEFAULT 0,
              long_eye_closure_events INTEGER NOT NULL DEFAULT 0,
              looking_away_events INTEGER NOT NULL DEFAULT 0,
              yawn_events INTEGER NOT NULL DEFAULT 0,
              distraction_events INTEGER NOT NULL DEFAULT 0,
              attention_events INTEGER NOT NULL DEFAULT 0,
              total_flags INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        webcam_cols = [r["name"] for r in db.execute("PRAGMA table_info(webcam_results)").fetchall()]
        for column_name, column_sql in {
            "monitoring_seconds": "ALTER TABLE webcam_results ADD COLUMN monitoring_seconds INTEGER NOT NULL DEFAULT 0",
            "face_check_attempts": "ALTER TABLE webcam_results ADD COLUMN face_check_attempts INTEGER NOT NULL DEFAULT 0",
            "face_verified_checks": "ALTER TABLE webcam_results ADD COLUMN face_verified_checks INTEGER NOT NULL DEFAULT 0",
            "face_mismatch_events": "ALTER TABLE webcam_results ADD COLUMN face_mismatch_events INTEGER NOT NULL DEFAULT 0",
        }.items():
            if column_name not in webcam_cols:
                db.execute(column_sql)
        webcam_signal_columns = {
            "eyes_open_ms": "ALTER TABLE webcam_results ADD COLUMN eyes_open_ms INTEGER NOT NULL DEFAULT 0",
            "eyes_closed_ms": "ALTER TABLE webcam_results ADD COLUMN eyes_closed_ms INTEGER NOT NULL DEFAULT 0",
            "gaze_center_count": "ALTER TABLE webcam_results ADD COLUMN gaze_center_count INTEGER NOT NULL DEFAULT 0",
            "gaze_left_count": "ALTER TABLE webcam_results ADD COLUMN gaze_left_count INTEGER NOT NULL DEFAULT 0",
            "gaze_right_count": "ALTER TABLE webcam_results ADD COLUMN gaze_right_count INTEGER NOT NULL DEFAULT 0",
            "gaze_away_count": "ALTER TABLE webcam_results ADD COLUMN gaze_away_count INTEGER NOT NULL DEFAULT 0",
            "pitch_sum": "ALTER TABLE webcam_results ADD COLUMN pitch_sum REAL NOT NULL DEFAULT 0",
            "yaw_sum": "ALTER TABLE webcam_results ADD COLUMN yaw_sum REAL NOT NULL DEFAULT 0",
            "roll_sum": "ALTER TABLE webcam_results ADD COLUMN roll_sum REAL NOT NULL DEFAULT 0",
            "pose_samples": "ALTER TABLE webcam_results ADD COLUMN pose_samples INTEGER NOT NULL DEFAULT 0",
            "expression_neutral": "ALTER TABLE webcam_results ADD COLUMN expression_neutral INTEGER NOT NULL DEFAULT 0",
            "expression_happy": "ALTER TABLE webcam_results ADD COLUMN expression_happy INTEGER NOT NULL DEFAULT 0",
            "expression_sad": "ALTER TABLE webcam_results ADD COLUMN expression_sad INTEGER NOT NULL DEFAULT 0",
            "expression_angry": "ALTER TABLE webcam_results ADD COLUMN expression_angry INTEGER NOT NULL DEFAULT 0",
            "expression_surprised": "ALTER TABLE webcam_results ADD COLUMN expression_surprised INTEGER NOT NULL DEFAULT 0",
            "expression_fearful": "ALTER TABLE webcam_results ADD COLUMN expression_fearful INTEGER NOT NULL DEFAULT 0",
            "expression_disgusted": "ALTER TABLE webcam_results ADD COLUMN expression_disgusted INTEGER NOT NULL DEFAULT 0",
            "head_turned_left_events": "ALTER TABLE webcam_results ADD COLUMN head_turned_left_events INTEGER NOT NULL DEFAULT 0",
            "head_turned_right_events": "ALTER TABLE webcam_results ADD COLUMN head_turned_right_events INTEGER NOT NULL DEFAULT 0",
        }
        for column_name, column_sql in webcam_signal_columns.items():
            if column_name not in webcam_cols:
                db.execute(column_sql)

        face_profile_cols = [r["name"] for r in db.execute("PRAGMA table_info(face_profiles)").fetchall()]
        for column_name, column_sql in {
            "embedding": "ALTER TABLE face_profiles ADD COLUMN embedding TEXT",
            "embedding_model": "ALTER TABLE face_profiles ADD COLUMN embedding_model TEXT",
            "face_image": "ALTER TABLE face_profiles ADD COLUMN face_image BLOB",
            "face_method": "ALTER TABLE face_profiles ADD COLUMN face_method TEXT",
        }.items():
            if column_name not in face_profile_cols:
                db.execute(column_sql)

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS cursor_results (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              session_id TEXT,
              game_name TEXT NOT NULL,
              reaction_time REAL,
              click_accuracy REAL,
              correct_clicks INTEGER,
              incorrect_clicks INTEGER,
              missed_clicks INTEGER,
              double_click_errors INTEGER,
              cursor_distance REAL,
              cursor_speed REAL,
              hesitation_time REAL,
              target_selection_accuracy REAL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS monitoring_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              session_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              severity TEXT,
              flag_number INTEGER,
              message TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        # Add new columns if running on an existing database.
        cols = [r["name"] for r in db.execute("PRAGMA table_info(game_results)").fetchall()]
        for column_name, column_sql in {
            "avg_reaction_ms": "ALTER TABLE game_results ADD COLUMN avg_reaction_ms INTEGER",
            "best_reaction_ms": "ALTER TABLE game_results ADD COLUMN best_reaction_ms INTEGER",
            "median_reaction_ms": "ALTER TABLE game_results ADD COLUMN median_reaction_ms INTEGER",
            "performance_level": "ALTER TABLE game_results ADD COLUMN performance_level TEXT",
            "reaction_times_json": "ALTER TABLE game_results ADD COLUMN reaction_times_json TEXT",
            "reference_status": "ALTER TABLE game_results ADD COLUMN reference_status TEXT",
            "baseline_median_reaction_ms": "ALTER TABLE game_results ADD COLUMN baseline_median_reaction_ms INTEGER",
            "baseline_status": "ALTER TABLE game_results ADD COLUMN baseline_status TEXT",
            "change_from_baseline_percent": "ALTER TABLE game_results ADD COLUMN change_from_baseline_percent REAL",
            "trend_status": "ALTER TABLE game_results ADD COLUMN trend_status TEXT",
            "maximum_sequence_length": "ALTER TABLE game_results ADD COLUMN maximum_sequence_length INTEGER",
            "maximum_level_completed": "ALTER TABLE game_results ADD COLUMN maximum_level_completed INTEGER",
            "correct_responses": "ALTER TABLE game_results ADD COLUMN correct_responses INTEGER",
            "wrong_responses": "ALTER TABLE game_results ADD COLUMN wrong_responses INTEGER",
            "total_responses": "ALTER TABLE game_results ADD COLUMN total_responses INTEGER",
            "memory_performance_score": "ALTER TABLE game_results ADD COLUMN memory_performance_score REAL",
            "session_id": "ALTER TABLE game_results ADD COLUMN session_id TEXT",
            "difficulty": "ALTER TABLE game_results ADD COLUMN difficulty INTEGER",
            "number_of_cards": "ALTER TABLE game_results ADD COLUMN number_of_cards INTEGER",
            "number_of_pairs": "ALTER TABLE game_results ADD COLUMN number_of_pairs INTEGER",
            "completion_time_seconds": "ALTER TABLE game_results ADD COLUMN completion_time_seconds INTEGER",
            "total_moves": "ALTER TABLE game_results ADD COLUMN total_moves INTEGER",
            "successful_matches": "ALTER TABLE game_results ADD COLUMN successful_matches INTEGER",
            "total_pairs": "ALTER TABLE game_results ADD COLUMN total_pairs INTEGER",
            "move_efficiency": "ALTER TABLE game_results ADD COLUMN move_efficiency REAL",
            "time_efficiency": "ALTER TABLE game_results ADD COLUMN time_efficiency REAL",
            "time_per_pair": "ALTER TABLE game_results ADD COLUMN time_per_pair REAL",
            "final_memory_match_score": "ALTER TABLE game_results ADD COLUMN final_memory_match_score REAL",
            "baseline_memory_match_score": "ALTER TABLE game_results ADD COLUMN baseline_memory_match_score REAL",
            "baseline_accuracy": "ALTER TABLE game_results ADD COLUMN baseline_accuracy REAL",
            "baseline_move_efficiency": "ALTER TABLE game_results ADD COLUMN baseline_move_efficiency REAL",
            "baseline_time_per_pair": "ALTER TABLE game_results ADD COLUMN baseline_time_per_pair REAL",
            "session_number": "ALTER TABLE game_results ADD COLUMN session_number INTEGER",
            "change_from_baseline_percent": "ALTER TABLE game_results ADD COLUMN change_from_baseline_percent REAL",
            "trend_status": "ALTER TABLE game_results ADD COLUMN trend_status TEXT",
        }.items():
            if column_name not in cols:
                db.execute(column_sql)

        # face_profiles: OpenCV Haar Cascade + LBPH stores the normalized
        # registered face crop directly in SQLite. Keep the old columns for
        # backward compatibility, but they are no longer used for login.
        face_cols = [r["name"] for r in db.execute("PRAGMA table_info(face_profiles)").fetchall()]
        for column_name, column_sql in {
            "embedding": "ALTER TABLE face_profiles ADD COLUMN embedding TEXT",
            "embedding_model": "ALTER TABLE face_profiles ADD COLUMN embedding_model TEXT",
            "face_image": "ALTER TABLE face_profiles ADD COLUMN face_image BLOB",
            "face_method": "ALTER TABLE face_profiles ADD COLUMN face_method TEXT",
        }.items():
            if column_name not in face_cols:
                db.execute(column_sql)

        db.commit()

    @app.teardown_appcontext
    def close_db(_exc: Exception | None) -> None:
        db = g.pop("db", None)
        if db is not None:
            db.close()

    with app.app_context():
        init_db()

    # ------------------------------------------------------------------
    # Session-flow helpers
    # ------------------------------------------------------------------

    def get_next_activity_url(current_activity_key: str, session_id: str | None) -> str:
        if not session_id:
            return url_for("dashboard")
        db=get_db(); user_id=int(session["user_id"])
        row=session_manager.get_session(db,session_id,user_id)
        if not row or row["status"]!="in_progress" or session_manager.session_remaining_seconds(db,session_id,user_id)<=0:
            return url_for("session_complete",session_id=session_id)
        endpoint=session_manager.next_activity_endpoint(current_activity_key,row["weakest_area"])
        return url_for(endpoint or "session_complete",session_id=session_id)

    def active_session_id() -> str | None:
        sid = request.values.get("session_id")
        if not sid or not session.get("user_id"):
            return None
        db = get_db()
        row = session_manager.get_session(db, sid, int(session["user_id"]))
        if row is None or row["status"] != "in_progress":
            return None
        return sid

    # ------------------------------------------------------------------
    # Public / auth
    # ------------------------------------------------------------------

    @app.get("/")
    def home():
        features = [
            {
                "title": "Memory games",
                "desc": "Short, friendly exercises to practice recall and attention.",
            },
            {
                "title": "Daily routine support",
                "desc": "Lightweight check-ins that help users stay consistent.",
            },
            {
                "title": "Progress tracking",
                "desc": "Simple summaries to see improvement over time.",
            },
        ]

        return render_template(
            "index.html",
            app_name="AI Cognitive Rehabilitation System",
            features=features,
            year=datetime.now().year,
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = (request.form.get("email") or "").strip().lower()
            password = request.form.get("password") or ""

            db = get_db()
            user = db.execute(
                "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()

            if user and user["password_hash"] and check_password_hash(user["password_hash"], password):
                # Password verified. Do NOT log the user in yet -- face
                # verification is a required second factor, not a bypassable
                # extra. Full session["user_id"] is only set after a face match.
                session.clear()
                session["pending_face_user_id"] = int(user["id"])
                session["pending_face_full_name"] = user["full_name"]
                session["pending_face_email"] = user["email"]
                return redirect(url_for("login_face_verify"))

            flash("Invalid email or password. Please try again.", "error")

        return render_template(
            "login.html",
            app_name="AI Cognitive Rehabilitation System",
            year=datetime.now().year,
        )

    @app.get("/login/face-verify")
    def login_face_verify():
        pending_user_id = session.get("pending_face_user_id")
        if not pending_user_id:
            flash("Please login with your email and password first.", "error")
            return redirect(url_for("login"))

        db = get_db()
        sample_count = db.execute(
            "SELECT COUNT(*) AS count FROM face_samples WHERE user_id = ?", (pending_user_id,)
        ).fetchone()["count"]
        profile = db.execute(
            "SELECT face_image FROM face_profiles WHERE user_id = ?", (pending_user_id,)
        ).fetchone()
        has_face_profile = bool(sample_count or (profile and profile["face_image"]))

        return render_template(
            "login_face_verify.html",
            app_name="AI Cognitive Rehabilitation System",
            year=datetime.now().year,
            full_name=session.get("pending_face_full_name", "User"),
            has_face_profile=has_face_profile,
        )

    @app.post("/login/face-verify/check")
    def login_face_verify_check():
        pending_user_id = session.get("pending_face_user_id")
        if not pending_user_id:
            return jsonify(message="Your login session expired. Please login again."), 401

        image_file = request.files.get("image")
        if image_file is None:
            return jsonify(message="Capture an image before verifying."), 400

        image_bytes = image_file.read()
        if len(image_bytes) > 5 * 1024 * 1024:
            return jsonify(message="The captured image is too large."), 400

        db = get_db()
        samples = db.execute(
            "SELECT face_image FROM face_samples WHERE user_id = ? ORDER BY sample_index",
            (pending_user_id,),
        ).fetchall()
        if not samples:
            profile = db.execute(
                "SELECT face_image FROM face_profiles WHERE user_id = ?", (pending_user_id,)
            ).fetchone()
            if profile and profile["face_image"]:
                samples = [{"face_image": profile["face_image"]}]

        if not samples:
            return jsonify(
                message="No registered face is on file for this account. Please complete face enrollment first.",
                missingProfile=True,
            ), 400

        try:
            is_match, confidence = verify_login_face(
                image_bytes,
                [row["face_image"] for row in samples],
            )
        except FaceVerificationError as error:
            return jsonify(message=str(error)), 400

        if not is_match:
            return jsonify(
                message="Face does not match the registered face. Access denied.",
                confidence=confidence,
            ), 401

        user = db.execute(
            "SELECT id, full_name, email FROM users WHERE id = ?", (pending_user_id,)
        ).fetchone()
        if not user:
            return jsonify(message="Account not found."), 400

        session.clear()
        session["user_id"] = int(user["id"])
        session["full_name"] = user["full_name"]
        session["email"] = user["email"]
        return jsonify(message="Face verified. Access granted.", redirect=url_for("dashboard"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            full_name = (request.form.get("full_name") or "").strip()
            email = (request.form.get("email") or "").strip().lower()
            password = request.form.get("password") or ""
            age_raw = (request.form.get("age") or "").strip()
            gender = (request.form.get("gender") or "").strip()
            patient_type = (request.form.get("patient_type") or "").strip()

            age = None
            if age_raw:
                try:
                    age = int(age_raw)
                except ValueError:
                    flash("Age must be a number.", "error")
                    return redirect(url_for("register"))

            if not full_name or not email or not password:
                flash("Please fill in all required fields.", "error")
            else:
                db = get_db()
                existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
                if existing:
                    flash("An account with this email already exists. Please login instead.", "error")
                else:
                    cursor = db.execute(
                        """
                        INSERT INTO users (full_name, email, password_hash, age, gender, patient_type, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            full_name,
                            email,
                            generate_password_hash(password),
                            age,
                            gender,
                            patient_type,
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                    )
                    db.commit()
                    # Registration flow continues straight into face registration
                    # (see master flow: REGISTER -> FACE REGISTRATION -> LOGIN).
                    # A lightweight pending-enrollment marker lets the enrollment
                    # page accept a capture without a full login session.
                    session.clear()
                    session["enroll_pending_user_id"] = int(cursor.lastrowid)
                    flash("Account created. Now let's register your face for secure login.", "success")
                    return redirect(url_for("face_enrollment"))

        return render_template(
            "register.html",
            app_name="AI Cognitive Rehabilitation System",
            year=datetime.now().year,
        )

    @app.get("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "success")
        return redirect(url_for("home"))

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    @app.get("/dashboard")
    def dashboard():
        if not session.get("user_id"):
            flash("Please login to access your dashboard.", "error")
            return redirect(url_for("login"))

        user_id = session.get("user_id")
        username = session.get("full_name", "User")
        db = get_db()

        # Get completed count and best score
        completed = db.execute(
            "SELECT COUNT(*) as count FROM game_results WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        completed_count = completed["count"] if completed else 0

        best = db.execute(
            "SELECT MAX(score) as best FROM game_results WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        best_score = best["best"] if best and best["best"] else 0

        # Get average score
        avg = db.execute(
            "SELECT AVG(score) as avg FROM game_results WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        avg_score = int(avg["avg"]) if avg and avg["avg"] else 0

        # Get recent progress (last 10 results)
        recent = db.execute(
            """
            SELECT game_name, score, created_at
            FROM game_results
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (user_id,)
        ).fetchall()

        progress = []
        for item in recent:
            created = datetime.fromisoformat(item["created_at"])
            when = created.strftime("%I:%M %p on %m/%d")
            progress.append({
                "test": item["game_name"],
                "score": item["score"],
                "when": when
            })

        # Get memory test scores for graph
        memory_results = db.execute(
            """
            SELECT
              COALESCE(memory_performance_score, score) AS score_value,
              created_at
            FROM game_results
            WHERE user_id = ? AND game_name = 'Memory Test'
            ORDER BY created_at ASC
            """,
            (user_id,)
        ).fetchall()

        memory_scores = [float(r["score_value"] or 0.0) for r in memory_results]
        memory_times = []
        for r in memory_results:
            dt = datetime.fromisoformat(r["created_at"])
            memory_times.append(dt.strftime("%m/%d %I:%M %p"))

        scores_json = json.dumps(memory_scores)
        times_json = json.dumps(memory_times)

        # Calculate trend
        trend = "Stable ➖"
        if len(memory_scores) >= 2:
            if memory_scores[-1] > memory_scores[-2]:
                trend = "Improving 📈"
            elif memory_scores[-1] < memory_scores[-2]:
                trend = "Declining 📉"

        # Calculate cognitive scores using the project-defined memory score out of 100.
        memory_result = db.execute(
            """
            SELECT AVG(COALESCE(memory_performance_score, score)) as avg_score
            FROM game_results
            WHERE user_id = ? AND game_name = 'Memory Test'
            """,
            (user_id,)
        ).fetchone()
        memory_score = float(memory_result["avg_score"] or 0.0) if memory_result else 0.0
        memory_percent = min(memory_score, 100)

        latest_memory_result = db.execute(
            """
            SELECT
              COALESCE(memory_performance_score, score) AS score_value,
              COALESCE(maximum_sequence_length, maximum_level_completed, 0) AS max_sequence_length,
              COALESCE(accuracy, 0) AS accuracy,
              COALESCE(correct_responses, 0) AS correct_responses,
              COALESCE(wrong_responses, 0) AS wrong_responses,
              created_at
            FROM game_results
            WHERE user_id = ? AND game_name = 'Memory Test'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,)
        ).fetchone()
        latest_memory_score = float(latest_memory_result["score_value"] or 0.0) if latest_memory_result else 0.0
        latest_max_sequence_length = int(latest_memory_result["max_sequence_length"] or 0) if latest_memory_result else 0
        latest_accuracy = float(latest_memory_result["accuracy"] or 0.0) if latest_memory_result else 0.0
        latest_correct_responses = int(latest_memory_result["correct_responses"] or 0) if latest_memory_result else 0
        latest_wrong_responses = int(latest_memory_result["wrong_responses"] or 0) if latest_memory_result else 0

        reaction_result = db.execute(
            """
            SELECT MIN(best_reaction_ms) as best_time, AVG(best_reaction_ms) as avg_time
            FROM game_results
            WHERE user_id = ? AND game_name = 'Reaction Test'
            """,
            (user_id,)
        ).fetchone()
        if reaction_result and reaction_result["best_time"] and reaction_result["avg_time"]:
            reaction_percent = min((reaction_result["best_time"] / reaction_result["avg_time"]) * 100, 100)
        else:
            reaction_percent = 0

        game_result = db.execute(
            "SELECT AVG(score) as avg_score FROM game_results WHERE user_id = ? AND game_name = 'Memory Match Game'",
            (user_id,)
        ).fetchone()
        game_score = float(game_result["avg_score"] or 0) if game_result else 0.0
        game_percent = min(game_score, 100.0)

        # Calculate overall score
        scores_available = [s for s in [memory_percent, reaction_percent, game_percent] if s > 0]
        overall_score = sum(scores_available) / len(scores_available) if scores_available else 0

        # Performance label
        if overall_score >= 75:
            performance_label = "Good Performance 🌟"
        elif overall_score >= 50:
            performance_label = "Average Performance 👍"
        else:
            performance_label = "Needs Improvement 💪"

        # Get test breakdown
        test_breakdown = {}
        test_icons = {
            "Memory Test": "🧠",
            "Reaction Test": "⚡",
            "Memory Match Game": "🎮",
            "Voice Cognitive Game": "🎙"
        }

        for test_name, icon in test_icons.items():
            results = db.execute(
                """
                SELECT COUNT(*) as count, MAX(score) as best, AVG(score) as avg
                FROM game_results
                WHERE user_id = ? AND game_name = ?
                """,
                (user_id, test_name)
            ).fetchone()

            if results and results["count"] > 0:
                test_breakdown[test_name] = {
                    "icon": icon,
                    "count": results["count"],
                    "best": int(results["best"]) if results["best"] else 0,
                    "avg": int(results["avg"]) if results["avg"] else 0
                }

        # Voice history is stored separately from game_results. Calculate a clear
        # 0-100 composite so the dashboard also shows the voice activity.
        voice_rows_all = db.execute(
            "SELECT level, correct, recall_accuracy, unique_words, valid_word_count FROM voice_results WHERE user_id = ? ORDER BY created_at",
            (int(user_id),),
        ).fetchall()
        voice_level_values = {1: [], 2: [], 3: []}
        for vr in voice_rows_all:
            level = int(vr["level"] or 0)
            if level == 1 and vr["correct"] is not None:
                voice_level_values[1].append(100.0 if int(vr["correct"]) else 0.0)
            elif level == 2 and vr["recall_accuracy"] is not None:
                voice_level_values[2].append(float(vr["recall_accuracy"]))
            elif level == 3:
                count = float(vr["unique_words"] if vr["unique_words"] is not None else (vr["valid_word_count"] or 0))
                voice_level_values[3].append(min(100.0, count * 10.0))
        voice_parts = []
        for level, weight in ((1, .25), (2, .35), (3, .40)):
            if voice_level_values[level]:
                voice_parts.append((sum(voice_level_values[level]) / len(voice_level_values[level]), weight))
        voice_percent = round(sum(v*w for v,w in voice_parts) / sum(w for _,w in voice_parts), 1) if voice_parts else 0
        voice_count = len(voice_rows_all)
        voice_history = db.execute(
            "SELECT level, round_number, task_type, category, recall_accuracy, valid_word_count, unique_words, repeated_words, created_at FROM voice_results WHERE user_id = ? ORDER BY created_at DESC LIMIT 12",
            (int(user_id),),
        ).fetchall()

        # Adaptive therapy plan: use all available saved domain results.
        domain_snapshot = performance_analysis.compute_user_domain_snapshot(db, int(user_id))
        therapy_plan = performance_analysis.generate_therapy_plan(domain_snapshot)
        voice_count_row = db.execute("SELECT COUNT(*) AS count FROM voice_results WHERE user_id = ?", (int(user_id),)).fetchone()
        voice_result_count = int(voice_count_row["count"] or 0)
        voice_history = db.execute("SELECT level,round_number,task_type,category,recall_accuracy,valid_word_count,unique_words,repeated_words,created_at FROM voice_results WHERE user_id = ? ORDER BY created_at DESC LIMIT 20", (int(user_id),)).fetchall()
        memory_percent = float(domain_snapshot.get("Memory") or 0.0)
        reaction_percent = float(domain_snapshot.get("Reaction") or 0.0)
        voice_percent = float(domain_snapshot.get("Voice") or 0.0)
        cursor_percent = float(domain_snapshot.get("Cursor") or 0.0)
        attention_percent = float(domain_snapshot.get("Attention") or 0.0)
        attempted_scores=[v for v in domain_snapshot.values() if v is not None]
        overall_score=(sum(attempted_scores)/len(attempted_scores)) if attempted_scores else 0.0
        performance_label="Good Performance" if overall_score>=75 else ("Average Performance" if overall_score>=50 else "Needs Improvement")

        # --- New: guided-session history with 5-domain scores, trends, recommendations ---
        session_history = performance_analysis.compute_session_history_trends(db, int(user_id))
        latest_session = session_history[-1] if session_history else None
        if voice_count:
            test_breakdown["Voice Cognitive Game"] = {
                "icon": "🎤", "count": voice_count, "best": int(round(voice_percent)), "avg": int(round(voice_percent))
            }

        session_domain_labels_json = json.dumps(performance_analysis.DOMAINS)
        session_history_chart_json = json.dumps(
            [
                {
                    "label": (
                        datetime.fromisoformat(s["start_time"]).strftime("%m/%d %I:%M %p")
                        if s["start_time"] else s["session_id"][:8]
                    ),
                    "overall": s["overall_score"],
                    "domains": s["domain_scores"],
                }
                for s in session_history
            ]
        )

        has_face_profile = bool(
            db.execute(
                "SELECT 1 FROM face_samples WHERE user_id = ? LIMIT 1",
                (user_id,),
            ).fetchone()
            or db.execute(
                "SELECT 1 FROM face_profiles WHERE user_id = ? AND face_image IS NOT NULL",
                (user_id,),
            ).fetchone()
        )

        return render_template(
            "dashboard.html",
            app_name="AI Cognitive Rehabilitation System",
            year=datetime.now().year,
            username=username,
            progress=progress,
            completed_count=completed_count,
            best_score=best_score,
            avg_score=avg_score,
            scores_json=scores_json,
            times_json=times_json,
            trend=trend,
            memory_percent=round(memory_percent, 1),
            reaction_percent=round(reaction_percent, 1),
            game_percent=round(game_percent, 1),
            voice_percent=round(voice_percent, 1),
            voice_result_count=voice_result_count,
            voice_history=voice_history,
            cursor_percent=round(cursor_percent, 1),
            attention_percent=round(attention_percent, 1),
            domain_snapshot=domain_snapshot,
            therapy_plan=therapy_plan,
            overall_score=round(overall_score, 1),
            performance_label=performance_label,
            test_breakdown=test_breakdown,
            latest_memory_score=round(latest_memory_score, 2),
            latest_max_sequence_length=latest_max_sequence_length,
            latest_accuracy=round(latest_accuracy, 2),
            latest_correct_responses=latest_correct_responses,
            latest_wrong_responses=latest_wrong_responses,
            has_face_profile=has_face_profile,
            session_history=session_history,
            latest_session=latest_session,
            session_domain_labels_json=session_domain_labels_json,
            session_history_chart_json=session_history_chart_json,
            session_duration_minutes=session_manager.SESSION_DURATION_SECONDS // 60,
        )

    # ------------------------------------------------------------------
    # Face enrollment (register-time and re-enrollment from dashboard)
    # ------------------------------------------------------------------

    @app.get("/face-enrollment")
    def face_enrollment():
        if not session.get("user_id") and not session.get("enroll_pending_user_id"):
            flash("Please login to enroll your face.", "error")
            return redirect(url_for("login"))

        return render_template(
            "face_enrollment.html",
            app_name="AI Cognitive Rehabilitation System",
            year=datetime.now().year,
            is_registration_flow=bool(session.get("enroll_pending_user_id")),
        )

    @app.post("/face-enrollment/capture")
    def capture_face_enrollment():
        target_user_id = session.get("user_id") or session.get("enroll_pending_user_id")
        if not target_user_id:
            return jsonify(message="Please login to enroll your face."), 401

        image_file = request.files.get("image")
        if image_file is None:
            return jsonify(message="Capture an image before enrolling."), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify(message="The captured image is empty."), 400
        if len(image_bytes) > 5 * 1024 * 1024:
            return jsonify(message="The captured image is too large."), 400

        try:
            sample_index = int(request.form.get("sample_index", "1"))
        except (TypeError, ValueError):
            return jsonify(message="Invalid face sample number."), 400
        if sample_index < 1 or sample_index > 10:
            return jsonify(message="Face sample number must be between 1 and 10."), 400

        db = get_db()
        user = db.execute(
            "SELECT id, full_name FROM users WHERE id = ?", (int(target_user_id),)
        ).fetchone()
        if user is None:
            return jsonify(message="User account not found."), 404

        try:
            # Pretrained FaceNet creates a normalized 512-value embedding.
            registered_face = extract_registered_face(image_bytes)
            saved_photo_path = save_registered_face_photo(
                int(target_user_id), user["full_name"], image_bytes, sample_index
            )
        except (FaceEnrollmentError, ValueError) as error:
            return jsonify(message=str(error)), 400

        timestamp = datetime.now().isoformat(timespec="seconds")

        # Starting sample 1 begins a fresh ten-image multi-pose enrollment.
        if sample_index == 1:
            db.execute(
                "DELETE FROM face_samples WHERE user_id = ?",
                (int(target_user_id),),
            )

        db.execute(
            """
            INSERT INTO face_samples (user_id, sample_index, face_image, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, sample_index) DO UPDATE SET
                face_image = excluded.face_image,
                created_at = excluded.created_at
            """,
            (int(target_user_id), sample_index, sqlite3.Binary(registered_face), timestamp),
        )

        embedding_values = np.frombuffer(registered_face, dtype=np.float32).tolist()
        db.execute(
            """
            INSERT INTO face_profiles
                (user_id, representation, embedding, embedding_model, face_image,
                 face_method, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                representation = excluded.representation,
                embedding = excluded.embedding,
                embedding_model = excluded.embedding_model,
                face_image = CASE WHEN ? = 1 THEN excluded.face_image
                                  ELSE face_profiles.face_image END,
                face_method = excluded.face_method,
                updated_at = excluded.updated_at
            """,
            (
                int(target_user_id),
                "FaceNet 512-dimensional embedding",
                json.dumps(embedding_values),
                MODEL_NAME,
                sqlite3.Binary(registered_face),
                "facenet-vggface2",
                timestamp,
                timestamp,
                sample_index,
            ),
        )
        db.commit()

        is_registration_flow = bool(session.get("enroll_pending_user_id"))
        if sample_index < 10:
            return jsonify(
                message=f"Face sample {sample_index} saved successfully.",
                sample_index=sample_index,
                total_samples=10,
                complete=False,
                photo_saved=True,
                photo_path=saved_photo_path,
                redirect=None,
            )

        if is_registration_flow:
            session.pop("enroll_pending_user_id", None)

        return jsonify(
            message="FaceNet enrollment completed successfully with 10 multi-pose samples.",
            sample_index=10,
            total_samples=10,
            complete=True,
            photo_saved=True,
            photo_path=saved_photo_path,
            redirect=url_for("login") if is_registration_flow else None,
        )

    def _start_activity(db, user_id: int, session_id: str | None, activity_key: str) -> bool:
        if not session_id:
            return False
        row = session_manager.get_session(db, session_id, user_id)
        if row is None or row["status"] != "in_progress":
            return False
        now = datetime.now().isoformat(timespec="seconds")
        db.execute(
            """
            INSERT INTO session_activities
              (session_id, user_id, activity_key, started_at, status)
            VALUES (?, ?, ?, ?, 'in_progress')
            ON CONFLICT(session_id, activity_key) DO UPDATE SET
              started_at = CASE WHEN session_activities.status = 'completed' THEN session_activities.started_at ELSE excluded.started_at END,
              status = CASE WHEN session_activities.status = 'completed' THEN session_activities.status ELSE 'in_progress' END
            """,
            (session_id, user_id, activity_key, now),
        )
        db.commit()
        return True


    def _finish_activity(db, user_id: int, session_id: str | None, activity_key: str) -> None:
        if not session_id:
            return
        row = db.execute(
            "SELECT started_at, status FROM session_activities WHERE session_id = ? AND user_id = ? AND activity_key = ?",
            (session_id, user_id, activity_key),
        ).fetchone()
        if not row or row["status"] == "completed":
            return
        now = datetime.now().isoformat(timespec="seconds")
        duration = None
        try:
            duration = max(0, int((datetime.fromisoformat(now) - datetime.fromisoformat(row["started_at"])).total_seconds()))
        except ValueError:
            pass
        db.execute(
            "UPDATE session_activities SET ended_at = ?, duration_seconds = ?, status = 'completed' WHERE session_id = ? AND user_id = ? AND activity_key = ?",
            (now, duration, session_id, user_id, activity_key),
        )
        db.commit()


    @app.post("/session/activity/start")
    def session_activity_start():
        if not session.get("user_id"):
            return jsonify(message="Not authenticated"), 401
        payload = request.get_json(silent=True) or {}
        session_id = (payload.get("session_id") or "").strip()
        activity_key = (payload.get("activity_key") or "").strip()
        if not session_id or not activity_key:
            return jsonify(message="session_id and activity_key are required"), 400
        db = get_db()
        if session_manager.session_remaining_seconds(db, session_id, int(session["user_id"])) <= 0:
            return jsonify(message="Session time has expired.", expired=True), 409
        _start_activity(db, int(session["user_id"]), session_id, activity_key)
        return jsonify(status="started", remaining_seconds=session_manager.session_remaining_seconds(db, session_id, int(session["user_id"])))


    @app.post("/session/<session_id>/terminate")
    def session_terminate(session_id: str):
        if not session.get("user_id"):
            return jsonify(message="Not authenticated"), 401
        db = get_db()
        user_id = int(session["user_id"])
        row = session_manager.get_session(db, session_id, user_id)
        if row is None:
            return jsonify(message="Session not found."), 404
        if row["status"] == "in_progress":
            for activity in db.execute("SELECT activity_key FROM session_activities WHERE session_id = ? AND user_id = ? AND status = 'in_progress'", (session_id, user_id)).fetchall():
                _finish_activity(db, user_id, session_id, activity["activity_key"])
            analysis = performance_analysis.analyze_session(db, user_id, session_id)
            session_manager.mark_session_complete(db, session_id, user_id, analysis, status="terminated")
        return jsonify(status="terminated", redirect=url_for("home"))


    @app.post("/monitoring/face-check")
    def monitoring_face_check():
        if not session.get("user_id"):
            return jsonify(message="Not authenticated"), 401
        session_id = (request.form.get("session_id") or "").strip()
        image_file = request.files.get("image")
        if not session_id or image_file is None:
            return jsonify(message="session_id and image are required", status="VERIFICATION UNAVAILABLE"), 400
        if image_file.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
            return jsonify(message="Unsupported webcam image type.", status="VERIFICATION UNAVAILABLE"), 415
        if request.content_length and request.content_length > 2 * 1024 * 1024 + 65536:
            return jsonify(message="Monitoring frame is too large.", status="VERIFICATION UNAVAILABLE"), 413
        image_bytes = image_file.stream.read(2 * 1024 * 1024 + 1)
        if not image_bytes or len(image_bytes) > 2 * 1024 * 1024:
            return jsonify(message="Monitoring frame is empty or too large.", status="VERIFICATION UNAVAILABLE"), 400
        user_id = int(session["user_id"])
        db = get_db()
        owner = session_manager.get_session(db, session_id, user_id)
        if owner is None or owner["status"] != "in_progress":
            return jsonify(message="This active session does not belong to the authenticated account.", status="VERIFICATION UNAVAILABLE"), 403
        samples = db.execute("SELECT face_image FROM face_samples WHERE user_id = ? ORDER BY sample_index", (user_id,)).fetchall()
        if not samples:
            profile = db.execute("SELECT face_image FROM face_profiles WHERE user_id = ?", (user_id,)).fetchone()
            if profile and profile["face_image"]:
                samples = [{"face_image": profile["face_image"]}]
        if not samples:
            return jsonify(message="No registered face samples found.", match=False, status="VERIFICATION UNAVAILABLE"), 409
        try:
            face_results, image_size = verify_monitoring_faces(
                image_bytes, [r["face_image"] for r in samples]
            )
        except FaceVerificationError as error:
            message = str(error)
            lowered = message.lower()
            if "no face" in lowered or "clear face" in lowered or "image is too small" in lowered or "could not be aligned" in lowered:
                # A frame without a usable face is a normal monitoring outcome,
                # not a malformed HTTP request. Return a structured 200 response
                # so the client can apply the no-face timer without noisy 422 logs.
                return jsonify(message=message, match=False, faces=[], status="NO FACE DETECTED")
            app.logger.warning("Face verification failed for user_id=%s session_id=%s: %s", user_id, session_id, message)
            return jsonify(message=message, match=False, faces=[], status="VERIFICATION UNAVAILABLE"), 503

        # Return the result for every detected face; a different registered
        # account is still unknown in this session because comparison is scoped
        # only to this authenticated account's enrolled embeddings.
        for face in face_results:
            x1, y1, x2, y2 = face["box"]
            face["box"] = [
                round(x1 / image_size[0], 6), round(y1 / image_size[1], 6),
                round(x2 / image_size[0], 6), round(y2 / image_size[1], 6)
            ]
        matches = [face["match"] for face in face_results]
        overall_match = len(face_results) == 1 and matches[0]
        status = ("MULTIPLE PEOPLE DETECTED" if len(face_results) > 1 else
                  ("VERIFIED USER" if overall_match else "UNKNOWN PERSON"))
        return jsonify(match=bool(overall_match), confidence=(face_results[0]["distance"] if len(face_results) == 1 else None),
                       faces=face_results, status=status,
                       user_name=session.get("full_name", "User"), image_size=image_size)


    @app.post("/session/start")
    def session_start():
        if not session.get("user_id"):
            flash("Please login to start a session.", "error")
            return redirect(url_for("login"))

        db=get_db(); user_id=int(session["user_id"])
        session_id=session_manager.create_session(db,user_id)
        # The assessment always runs all four games. Weakest/strongest areas
        # are calculated after the games and shown only in the final report.
        return redirect(url_for("memory_test", session_id=session_id))

    @app.get("/session/<session_id>/instructions")
    def session_instructions(session_id: str):
        if not session.get("user_id"):
            flash("Please login to continue your session.", "error")
            return redirect(url_for("login"))

        db = get_db()
        row = session_manager.get_session(db, session_id, int(session["user_id"]))
        if row is None:
            flash("That session could not be found.", "error")
            return redirect(url_for("dashboard"))

        # Legacy route kept for bookmarked links, but the assessment no longer
        # uses a separate therapy/instruction session. Start the first game.
        return redirect(url_for("memory_test", session_id=session_id))

    @app.get("/session/<session_id>/complete")
    def session_complete(session_id: str):
        if not session.get("user_id"):
            flash("Please login to view your session results.", "error")
            return redirect(url_for("login"))

        db = get_db()
        user_id = int(session["user_id"])
        row = session_manager.get_session(db, session_id, user_id)
        if row is None:
            flash("That session could not be found.", "error")
            return redirect(url_for("dashboard"))

        for activity in db.execute("SELECT activity_key FROM session_activities WHERE session_id = ? AND user_id = ? AND status = 'in_progress'", (session_id, user_id)).fetchall():
            _finish_activity(db, user_id, session_id, activity["activity_key"])

        analysis = performance_analysis.analyze_session(db, user_id, session_id)
        if row["status"] == "in_progress":
            session_manager.mark_session_complete(db, session_id, user_id, analysis)
            row = session_manager.get_session(db, session_id, user_id)

        activities = db.execute("SELECT * FROM session_activities WHERE session_id = ? AND user_id = ? ORDER BY started_at", (session_id, user_id)).fetchall()
        monitoring = db.execute("SELECT * FROM webcam_results WHERE session_id = ? AND user_id = ?", (session_id, user_id)).fetchone()
        cursor_rows = db.execute("SELECT * FROM cursor_results WHERE session_id = ? AND user_id = ? ORDER BY created_at", (session_id, user_id)).fetchall()
        voice_rows = db.execute("SELECT level,round_number,task_type,category,recognized_text,correct,recall_accuracy,valid_word_count,unique_words,repeated_words,speech_rate,pause_count,pause_duration_ms,response_time_ms,speech_duration_ms,filler_count,created_at FROM voice_results WHERE session_id = ? AND user_id = ? ORDER BY level,round_number", (session_id,user_id)).fetchall()
        game_rows = db.execute("SELECT game_name, score, accuracy, time_taken_seconds, created_at FROM game_results WHERE session_id = ? AND user_id = ? ORDER BY created_at", (session_id, user_id)).fetchall()
        return render_template(
            "session_summary.html",
            app_name="AI Cognitive Rehabilitation System",
            year=datetime.now().year,
            session_id=session_id,
            analysis=analysis,
            domains=performance_analysis.DOMAINS,
            session_row=row,
            activities=activities,
            monitoring=monitoring,
            voice_rows=voice_rows,
            cursor_rows=cursor_rows,
            game_rows=game_rows,
        )

    # ------------------------------------------------------------------
    # Webcam + cursor monitoring endpoints (shared across all activities)
    # ------------------------------------------------------------------

    @app.post("/monitoring/event")
    def monitoring_event():
        if not session.get("user_id"):
            return jsonify(message="Not authenticated"), 401

        payload = request.get_json(silent=True) or {}
        session_id = (payload.get("session_id") or "").strip()
        event_type = (payload.get("event_type") or "").strip()
        if not session_id or not event_type:
            return jsonify(message="session_id and event_type are required"), 400

        db = get_db()
        user_id = int(session["user_id"])
        owner = session_manager.get_session(db, session_id, user_id)
        if owner is None or owner["status"] != "in_progress":
            return jsonify(message="Active session not found for authenticated user."), 403
        allowed_events = {
            "no_face_detected", "multiple_faces", "eyes_closed_5_seconds", "eyes_closed_10_seconds",
            "face_identity_mismatch", "face_verification_unavailable", "left_right_warning_threshold",
        }
        if event_type not in allowed_events:
            return jsonify(message="This event type is informational and cannot create a flag."), 400
        now_dt = datetime.now()
        now = now_dt.isoformat(timespec="seconds")
        recent = db.execute(
            "SELECT id FROM monitoring_events WHERE user_id=? AND session_id=? AND event_type=? AND created_at>=? ORDER BY created_at DESC LIMIT 1",
            (user_id, session_id, event_type, (now_dt - timedelta(seconds=3)).isoformat(timespec="seconds")),
        ).fetchone()
        if recent:
            return jsonify(status="debounced", message="This incident was already recorded."), 200
        db.execute(
            """
            INSERT INTO monitoring_events (user_id, session_id, event_type, severity, flag_number, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, session_id, event_type, payload.get("severity"),
             payload.get("flag_number"), str(payload.get("message") or event_type)[:1000], now),
        )
        db.commit()
        return jsonify(status="logged")

    @app.post("/monitoring/webcam-summary")
    def monitoring_webcam_summary():
        """Accumulate aggregated (not raw) webcam-monitoring counters for a session.

        Called by the client at the end of each activity with the counts
        observed during that activity; counts accumulate across the whole
        session via ON CONFLICT ... DO UPDATE.
        """
        if not session.get("user_id"):
            return jsonify(message="Not authenticated"), 401

        payload = request.get_json(silent=True) or {}
        session_id = (payload.get("session_id") or "").strip()
        if not session_id:
            return jsonify(message="session_id is required"), 400

        fields = [
            "face_missing_events", "multiple_face_events", "blink_count",
            "long_eye_closure_events", "looking_away_events", "yawn_events",
            "distraction_events", "attention_events", "total_flags",
            "monitoring_seconds", "face_check_attempts", "face_verified_checks", "face_mismatch_events",
            "eyes_open_ms", "eyes_closed_ms", "gaze_center_count", "gaze_left_count", "gaze_right_count", "gaze_away_count",
            "pitch_sum", "yaw_sum", "roll_sum", "pose_samples",
            "expression_neutral", "expression_happy", "expression_sad", "expression_angry", "expression_surprised", "expression_fearful", "expression_disgusted",
            "head_turned_left_events", "head_turned_right_events",
        ]
        values = {f: int(payload.get(f) or 0) for f in fields}
        now = datetime.now().isoformat(timespec="seconds")

        db = get_db()
        user_id = int(session["user_id"])
        owner = session_manager.get_session(db, session_id, user_id)
        if owner is None:
            return jsonify(message="Session not found for authenticated user."), 403
        # A final aggregated summary may arrive just after termination due to
        # overlapping pagehide/limit callbacks. Ownership is still mandatory.
        if owner["status"] not in {"in_progress", "terminated", "completed"}:
            return jsonify(message="Session is not available for monitoring summary."), 409
        db.execute(
            f"""
            INSERT INTO webcam_results
              (user_id, session_id, {", ".join(fields)}, created_at, updated_at)
            VALUES (?, ?, {", ".join(["?"] * len(fields))}, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
              {", ".join(f"{f} = {f} + excluded.{f}" for f in fields)},
              updated_at = excluded.updated_at
            """,
            (
                int(session["user_id"]),
                session_id,
                *[values[f] for f in fields],
                now,
                now,
            ),
        )
        db.commit()
        return jsonify(status="saved")

    @app.post("/cursor/save")
    def cursor_save():
        if not session.get("user_id"):
            return jsonify(message="Not authenticated"), 401

        payload = request.get_json(silent=True) or {}
        game_name = (payload.get("game_name") or "").strip()
        if not game_name:
            return jsonify(message="game_name is required"), 400

        db = get_db()
        db.execute(
            """
            INSERT INTO cursor_results
              (user_id, session_id, game_name, reaction_time, click_accuracy, correct_clicks,
               incorrect_clicks, missed_clicks, double_click_errors, cursor_distance, cursor_speed,
               hesitation_time, target_selection_accuracy, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(session["user_id"]),
                payload.get("session_id"),
                game_name,
                payload.get("reaction_time"),
                payload.get("click_accuracy"),
                payload.get("correct_clicks"),
                payload.get("incorrect_clicks"),
                payload.get("missed_clicks"),
                payload.get("double_click_errors"),
                payload.get("cursor_distance"),
                payload.get("cursor_speed"),
                payload.get("hesitation_time"),
                payload.get("target_selection_accuracy"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        db.commit()
        return jsonify(status="saved")

    # ------------------------------------------------------------------
    # Memory Test (existing game, extended with session_id pass-through)
    # ------------------------------------------------------------------

    @app.route("/memory-test", methods=["GET", "POST"])
    def memory_test():
        if not session.get("user_id"):
            flash("Please login to start a test.", "error")
            return redirect(url_for("login"))

        session_id = request.args.get("session_id")
        if session_id:
            _start_activity(get_db(), int(session["user_id"]), session_id, "memory_test")
        return render_template(
            "memory_test.html",
            app_name="AI Cognitive Rehabilitation System",
            year=datetime.now().year,
            session_id=session_id,
            instruction=session_manager.ACTIVITY_INSTRUCTIONS["memory_test"],
            next_url=get_next_activity_url("memory_test", session_id),
        )

    @app.post("/memory-test/save")
    def save_memory_test():
        if not session.get("user_id"):
            flash("Please login to save your score.", "error")
            return redirect(url_for("login"))

        session_id = request.form.get("session_id") or None

        try:
            score = float(request.form.get("memory_performance_score") or request.form.get("score") or 0.0)
            level = int(request.form.get("level") or 1)
            maximum_level_completed = int(request.form.get("maximum_level_completed") or request.form.get("maximum_sequence_length") or 0)
            maximum_sequence_length = int(request.form.get("maximum_sequence_length") or maximum_level_completed or 0)
            correct_responses = int(request.form.get("correct_responses") or 0)
            wrong_responses = int(request.form.get("wrong_responses") or 0)
            total_responses = int(request.form.get("total_responses") or 0)
            accuracy = float(request.form.get("accuracy") or 0.0)
            time_taken = int(request.form.get("time_taken") or 0)
        except ValueError:
            flash("Could not save memory test (invalid data). Please try again.", "error")
            return redirect(url_for("memory_test", session_id=session_id) if session_id else url_for("memory_test"))

        if total_responses <= 0:
            total_responses = correct_responses + wrong_responses
        if total_responses > 0:
            accuracy = max(0.0, min(100.0, (correct_responses / total_responses) * 100.0))
        else:
            accuracy = 0.0

        maximum_level_completed = max(0, maximum_level_completed)
        maximum_sequence_length = max(0, maximum_sequence_length)

        # The task is a gamified visual sequential-memory task inspired by established
        # cognitive memory paradigms. The raw measures include sequence length and recall
        # accuracy. The derived 0–100 score is a project-defined performance index and is
        # not a clinically validated diagnostic score.
        memory_span_score = (maximum_level_completed / 10.0) * 70.0 if maximum_level_completed else 0.0
        accuracy_score = accuracy * 0.30
        score = round(max(0.0, min(100.0, memory_span_score + accuracy_score)), 2)

        db = get_db()
        db.execute(
            """
            INSERT INTO game_results
              (
                user_id, game_name, score, accuracy, time_taken_seconds, moves,
                correct_matches, wrong_attempts, maximum_sequence_length,
                maximum_level_completed, correct_responses, wrong_responses,
                total_responses, memory_performance_score, session_id, created_at
              )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(session.get("user_id")),
                "Memory Test",
                float(score),
                float(accuracy),
                time_taken,
                total_responses,
                correct_responses,
                wrong_responses,
                maximum_sequence_length,
                maximum_level_completed,
                correct_responses,
                wrong_responses,
                total_responses,
                float(score),
                session_id,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        db.commit()

        # Cursor/interaction metrics for this activity, derived from the same
        # response data already collected for the memory task.
        db.execute(
            """
            INSERT INTO cursor_results
              (user_id, session_id, game_name, reaction_time, click_accuracy, correct_clicks,
               incorrect_clicks, missed_clicks, double_click_errors, cursor_distance, cursor_speed,
               hesitation_time, target_selection_accuracy, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(session.get("user_id")),
                session_id,
                "Memory Test",
                None,
                accuracy,
                correct_responses,
                wrong_responses,
                None,
                None,
                None,
                None,
                None,
                accuracy,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        db.commit()

        _finish_activity(get_db(), int(session.get("user_id")), session_id, "memory_test")
        flash(
            f"Memory Test completed! Memory Performance Score: {score:.2f}/100, "
            f"Level: {max(level, maximum_level_completed)}/10, Accuracy: {accuracy:.2f}%",
            "success",
        )
        return redirect(get_next_activity_url("memory_test", session_id))

    # ------------------------------------------------------------------
    # Reaction Test (existing game, extended with session_id pass-through)
    # ------------------------------------------------------------------

    @app.get("/reaction-test")
    def reaction_test():
        if not session.get("user_id"):
            flash("Please login to start a test.", "error")
            return redirect(url_for("login"))

        session_id = request.args.get("session_id")
        if session_id:
            _start_activity(get_db(), int(session["user_id"]), session_id, "reaction_test")
        return render_template(
            "reaction_test.html",
            app_name="AI Cognitive Rehabilitation System",
            year=datetime.now().year,
            session_id=session_id,
            instruction=session_manager.ACTIVITY_INSTRUCTIONS["reaction_test"],
            next_url=get_next_activity_url("reaction_test", session_id),
        )

    @app.get("/reaction-test/baseline")
    def reaction_test_baseline():
        if not session.get("user_id"):
            return {
                "baselineMedianMs": None,
                "baselineStatus": "Baseline not established",
                "changeFromBaselinePercent": 0.0,
                "trendStatus": "Baseline not established",
            }

        user_id = int(session.get("user_id"))
        db = get_db()
        prior_sessions = db.execute(
            """
            SELECT median_reaction_ms
            FROM game_results
            WHERE user_id = ? AND game_name = 'Reaction Test' AND median_reaction_ms IS NOT NULL
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()

        session_medians = [int(row["median_reaction_ms"]) for row in prior_sessions if row["median_reaction_ms"] is not None]
        if not session_medians:
            return {
                "baselineMedianMs": None,
                "baselineStatus": "Baseline not established",
                "changeFromBaselinePercent": 0.0,
                "trendStatus": "Baseline not established",
            }

        ordered = sorted(session_medians)
        if len(ordered) % 2 == 1:
            baseline_median = ordered[len(ordered) // 2]
        else:
            baseline_median = round((ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2)

        return {
            "baselineMedianMs": baseline_median,
            "baselineStatus": "Baseline established",
            "changeFromBaselinePercent": 0.0,
            "trendStatus": "Stable compared with baseline",
        }

    @app.post("/reaction-test/save")
    def save_reaction_test():
        if not session.get("user_id"):
            flash("Please login to save your score.", "error")
            return redirect(url_for("login"))

        session_id = request.form.get("session_id") or None

        try:
            avg_ms = int(request.form.get("avg_reaction_ms") or 0)
            median_ms = int(request.form.get("median_reaction_ms") or avg_ms or 0)
            best_ms = int(request.form.get("best_reaction_ms") or 0)
            score = float(request.form.get("score") or 0.0)
            performance_level = (request.form.get("performance_level") or "").strip()
            baseline_status = (request.form.get("baseline_status") or "").strip()
            change_from_baseline_percent = float(request.form.get("change_from_baseline_percent") or 0.0)
            trend_status = (request.form.get("trend_status") or "").strip()
            rounds = int(request.form.get("rounds") or 3)
            successful = int(request.form.get("successful_taps") or 0)
            early = int(request.form.get("early_clicks") or 0)
            reaction_times_json = request.form.get("reaction_times_json") or "[]"
            baseline_median_reaction_ms = int(request.form.get("baseline_median_reaction_ms") or 0)
        except ValueError:
            flash("Could not save reaction test (invalid data). Please restart and try again.", "error")
            return redirect(url_for("reaction_test", session_id=session_id) if session_id else url_for("reaction_test"))

        db = get_db()
        user_id = int(session.get("user_id"))

        prior_sessions = db.execute(
            """
            SELECT median_reaction_ms, created_at
            FROM game_results
            WHERE user_id = ? AND game_name = 'Reaction Test' AND median_reaction_ms IS NOT NULL
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()

        # This system does not use a universal reaction-time cutoff. It measures each user's
        # performance and establishes an individual baseline for longitudinal comparison. The
        # performance score and trend categories are project-specific and are not clinical
        # diagnostic measures.
        BASELINE_STABILITY_PERCENT = 5
        baseline_median = None
        if len(prior_sessions) >= 3:
            values = [int(row["median_reaction_ms"]) for row in prior_sessions if row["median_reaction_ms"] is not None]
            if values:
                values = sorted(values)
                if len(values) % 2 == 1:
                    baseline_median = values[len(values) // 2]
                else:
                    baseline_median = round((values[len(values) // 2 - 1] + values[len(values) // 2]) / 2)

        if baseline_median and baseline_median > 0:
            change_percent = ((baseline_median - median_ms) / baseline_median) * 100.0 if baseline_median > 0 else 0.0
            if abs(change_percent) <= BASELINE_STABILITY_PERCENT:
                trend_status = "Stable compared with baseline"
            elif change_percent > 0:
                trend_status = "Improved compared with baseline"
            else:
                trend_status = "Declined compared with baseline"
            baseline_status = trend_status
            change_from_baseline_percent = round(change_percent, 2)
        else:
            baseline_median = None
            baseline_status = "Baseline not established"
            trend_status = "Baseline not established"
            change_from_baseline_percent = 0.0

        accuracy = 0.0
        if rounds > 0:
            accuracy = round((successful / rounds) * 100.0, 2)

        if not performance_level:
            performance_level = trend_status if trend_status else "Baseline not established"

        db.execute(
            """
            INSERT INTO game_results
              (
                user_id, game_name, score, accuracy, time_taken_seconds, moves,
                correct_matches, wrong_attempts, avg_reaction_ms, best_reaction_ms,
                median_reaction_ms, performance_level, reaction_times_json,
                reference_status, baseline_median_reaction_ms, baseline_status,
                change_from_baseline_percent, trend_status, session_id, created_at
              )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                "Reaction Test",
                float(score),
                float(accuracy),
                int(max(0, avg_ms // 1000)),
                int(rounds),
                int(successful),
                int(early),
                int(avg_ms),
                int(best_ms),
                int(median_ms),
                performance_level,
                reaction_times_json,
                performance_level,
                int(baseline_median) if baseline_median else None,
                baseline_status,
                float(change_from_baseline_percent),
                trend_status,
                session_id,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        db.commit()

        # Cursor/interaction metrics for this activity (derived from the same round data).
        try:
            round_results = json.loads(reaction_times_json)
        except (TypeError, ValueError):
            round_results = []
        if round_results:
            db.execute(
                """
                INSERT INTO cursor_results
                  (user_id, session_id, game_name, reaction_time, click_accuracy, correct_clicks,
                   incorrect_clicks, missed_clicks, double_click_errors, cursor_distance, cursor_speed,
                   hesitation_time, target_selection_accuracy, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    "Reaction Test",
                    float(avg_ms),
                    float(accuracy),
                    int(successful),
                    int(early),
                    max(0, rounds - successful - early),
                    0,
                    None,
                    None,
                    None,
                    float(accuracy),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            db.commit()

        _finish_activity(get_db(), int(session.get("user_id")), session_id, "reaction_test")
        flash("Reaction Test score saved successfully.", "success")
        return redirect(get_next_activity_url("reaction_test", session_id))

    # ------------------------------------------------------------------
    # Memory Match Game (existing game, extended with session_id pass-through)
    # ------------------------------------------------------------------

    @app.get("/memory-game")
    def memory_game():
        if not session.get("user_id"):
            flash("Please login to access the memory game.", "error")
            return redirect(url_for("login"))

        db = get_db()
        baseline_rows = db.execute(
            """
            SELECT difficulty, AVG(final_memory_match_score) AS score
            FROM game_results
            WHERE user_id = ? AND game_name = 'Memory Match Game' AND difficulty IS NOT NULL
            GROUP BY difficulty
            """,
            (int(session.get("user_id")),),
        ).fetchall()
        baselines = {str(row["difficulty"]): round(float(row["score"]), 2) for row in baseline_rows}

        session_id = request.args.get("session_id")
        if session_id:
            _start_activity(get_db(), int(session["user_id"]), session_id, "memory_game")
        return render_template(
            "memory_game.html",
            app_name="AI Cognitive Rehabilitation System",
            year=datetime.now().year,
            baselines=baselines,
            session_id=session_id,
            instruction=session_manager.ACTIVITY_INSTRUCTIONS["memory_game"],
            next_url=get_next_activity_url("memory_game", session_id),
        )

    @app.post("/memory-game/save")
    def save_memory_game():
        if not session.get("user_id"):
            flash("Please login to save your score.", "error")
            return redirect(url_for("login"))

        session_id = request.form.get("session_id") or None

        try:
            game_session_id = (request.form.get("game_session_id") or str(uuid4())).strip()
            difficulty = int(request.form.get("difficulty") or 0)
            number_of_cards = int(request.form.get("number_of_cards") or difficulty)
            number_of_pairs = int(request.form.get("number_of_pairs") or (number_of_cards // 2))
            completion_time_seconds = max(0, int(request.form.get("completion_time_seconds") or request.form.get("time_taken_seconds") or 0))
            total_moves = max(0, int(request.form.get("total_moves") or request.form.get("moves") or 0))
            successful_matches = max(0, int(request.form.get("successful_matches") or request.form.get("correct_matches") or 0))
            total_pairs = max(0, int(request.form.get("total_pairs") or number_of_pairs))
            wrong_attempts = max(0, int(request.form.get("wrong_attempts") or 0))
        except ValueError:
            flash("Could not save score (invalid data). Please restart and try again.", "error")
            return redirect(url_for("memory_game", session_id=session_id) if session_id else url_for("memory_game"))

        if difficulty <= 0 or number_of_cards <= 0 or number_of_pairs <= 0:
            flash("Could not save score (invalid difficulty). Please restart and try again.", "error")
            return redirect(url_for("memory_game", session_id=session_id) if session_id else url_for("memory_game"))

        # Moves are pair-selection attempts in the game JavaScript. Accuracy therefore
        # remains successful pair attempts divided by total pair attempts.
        accuracy = (successful_matches / total_moves * 100.0) if total_moves else 0.0
        accuracy = max(0.0, min(100.0, accuracy))
        move_efficiency = (number_of_pairs / total_moves * 100.0) if total_moves else 0.0
        move_efficiency = max(0.0, min(100.0, move_efficiency))
        time_per_pair = completion_time_seconds / number_of_pairs
        # Time is normalized per pair, so different board sizes are comparable without
        # imposing an external or clinical time cutoff.
        time_efficiency = 100.0 / (1.0 + time_per_pair)
        time_efficiency = max(0.0, min(100.0, time_efficiency))
        # Published cognitive-testing research supports accuracy, errors, response time,
        # and efficiency as measurable variables. This weighted 0-100 composite is a
        # project-specific performance index, not a clinical diagnostic measure.
        final_score = max(0.0, min(100.0, (
            0.50 * accuracy + 0.30 * move_efficiency + 0.20 * time_efficiency
        )))

        db = get_db()
        user_id = int(session.get("user_id"))
        prior_sessions = db.execute(
            """
            SELECT final_memory_match_score, accuracy, move_efficiency, time_per_pair
            FROM game_results
            WHERE user_id = ? AND game_name = 'Memory Match Game' AND difficulty = ?
            ORDER BY created_at ASC
            """,
            (user_id, difficulty),
        ).fetchall()
        session_number = len(prior_sessions) + 1
        baseline_score = None
        baseline_accuracy = None
        baseline_move_efficiency = None
        baseline_time_per_pair = None
        change_percent = 0.0
        trend_status = "Baseline not established"
        if prior_sessions:
            baseline_score = sum(float(row["final_memory_match_score"] or 0.0) for row in prior_sessions) / len(prior_sessions)
            baseline_accuracy = sum(float(row["accuracy"] or 0.0) for row in prior_sessions) / len(prior_sessions)
            baseline_move_efficiency = sum(float(row["move_efficiency"] or 0.0) for row in prior_sessions) / len(prior_sessions)
            baseline_time_per_pair = sum(float(row["time_per_pair"] or 0.0) for row in prior_sessions) / len(prior_sessions)
            if baseline_score > 0:
                change_percent = ((final_score - baseline_score) / baseline_score) * 100.0
                # This is a project-defined trend threshold, not a medical cutoff.
                if abs(change_percent) <= 5:
                    trend_status = "Stable compared with baseline"
                elif change_percent > 0:
                    trend_status = "Improved compared with baseline"
                else:
                    trend_status = "Declined compared with baseline"

        db.execute(
            """
            INSERT INTO game_results
              (user_id, game_name, score, accuracy, time_taken_seconds, moves,
               correct_matches, wrong_attempts, session_id, difficulty,
               number_of_cards, number_of_pairs, completion_time_seconds,
               total_moves, successful_matches, total_pairs, move_efficiency, time_efficiency,
               time_per_pair, final_memory_match_score, baseline_memory_match_score,
               baseline_accuracy, baseline_move_efficiency, baseline_time_per_pair,
               session_number, change_from_baseline_percent, trend_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                "Memory Match Game",
                round(final_score, 2),
                accuracy,
                completion_time_seconds,
                total_moves,
                successful_matches,
                wrong_attempts,
                session_id,
                difficulty,
                number_of_cards,
                number_of_pairs,
                completion_time_seconds,
                total_moves,
                successful_matches,
                total_pairs,
                move_efficiency,
                time_efficiency,
                time_per_pair,
                round(final_score, 2),
                baseline_score,
                baseline_accuracy,
                baseline_move_efficiency,
                baseline_time_per_pair,
                session_number,
                change_percent,
                trend_status,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        db.commit()

        # Cursor/interaction metrics for this activity.
        db.execute(
            """
            INSERT INTO cursor_results
              (user_id, session_id, game_name, reaction_time, click_accuracy, correct_clicks,
               incorrect_clicks, missed_clicks, double_click_errors, cursor_distance, cursor_speed,
               hesitation_time, target_selection_accuracy, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_id,
                "Memory Match Game",
                None,
                accuracy,
                successful_matches,
                wrong_attempts,
                None,
                None,
                None,
                None,
                None,
                move_efficiency,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        db.commit()

        _finish_activity(get_db(), int(session.get("user_id")), session_id, "memory_game")
        flash(f"Memory Match Game score saved successfully: {final_score:.2f}/100.", "success")
        return redirect(get_next_activity_url("memory_game", session_id))

    # ------------------------------------------------------------------
    # Voice Cognitive Game (new: separate from the memory games)
    # ------------------------------------------------------------------

    # Word/sentence/category content now lives in voice_content.py, keyed by
    # language (English, Tamil, Kannada, Hindi) -- see that module for the
    # full vocabularies. LEVEL3_ROUNDS etc. stay here as game-pacing config.
    LEVEL3_ROUNDS = 6
    LEVEL3_PREP_SECONDS = 5
    LEVEL3_SPEAK_SECONDS = 15
    LEVEL1_SPEAK_SECONDS = 5
    LEVEL2_SPEAK_SECONDS = 8

    @app.get("/voice-game")
    def voice_game():
        if not session.get("user_id"):
            flash("Please login to play the voice game.", "error")
            return redirect(url_for("login"))

        session_id = request.args.get("session_id")
        if session_id:
            _start_activity(get_db(), int(session["user_id"]), session_id, "voice_game")
        language = voice_content.normalize_language(request.args.get("lang"))
        level1_word = voice_content.level1_word(language)
        level2_sentence = voice_content.level2_sentence(language)
        level3_category_key = voice_content.level3_category_key(language)
        level3_category = voice_content.level3_category_label(language, level3_category_key)

        return render_template(
            "voice_game.html",
            app_name="AI Cognitive Rehabilitation System",
            year=datetime.now().year,
            session_id=session_id,
            instruction=session_manager.ACTIVITY_INSTRUCTIONS["voice_game"],
            next_url=get_next_activity_url("voice_game", session_id),
            language=language,
            languages=voice_content.SUPPORTED_LANGUAGES,
            speech_locale=voice_content.speech_locale(language),
            level1_word=level1_word,
            level2_sentence=level2_sentence,
            level3_category=level3_category,
            level3_category_key=level3_category_key,
            level3_rounds=LEVEL3_ROUNDS,
            level3_prep_seconds=LEVEL3_PREP_SECONDS,
            level3_speak_seconds=LEVEL3_SPEAK_SECONDS,
            level1_speak_seconds=LEVEL1_SPEAK_SECONDS,
            level2_speak_seconds=LEVEL2_SPEAK_SECONDS,
        )

    @app.post("/voice-game/save")
    def save_voice_result():
        if not session.get("user_id"):
            return jsonify(message="Not authenticated"), 401

        payload = request.get_json(silent=True) or {}
        user_id = int(session["user_id"])
        session_id = payload.get("session_id")
        level = int(payload.get("level") or 0)
        round_number = int(payload.get("round_number") or 1)
        task_type = payload.get("task_type") or f"Level {level}"
        recognized_text = payload.get("recognized_text") or ""
        language = voice_content.normalize_language(payload.get("language"))
        response_time_ms = payload.get("response_time_ms")
        speech_duration_ms = payload.get("speech_duration_ms")
        pause_count = payload.get("pause_count")
        pause_duration_ms = payload.get("pause_duration_ms")

        speech_rate = None
        word_count = len([w for w in recognized_text.split() if w])
        if speech_duration_ms and speech_duration_ms > 0:
            speech_rate = round(word_count / (speech_duration_ms / 60000.0), 2)

        row = {
            "task_type": task_type,
            "category": None,
            "expected_text": None,
            "recognized_text": recognized_text,
            "correct": None,
            "correct_words": None,
            "incorrect_words": None,
            "missing_words": None,
            "extra_words": None,
            "unique_words": None,
            "repeated_words": None,
            "valid_word_count": None,
            "invalid_word_count": None,
            "recall_accuracy": None,
            "trend": None,
            "filler_count": 0,
        }

        if level == 1:
            expected_word = payload.get("expected_text") or ""
            result = voice_analysis.score_single_word_recall(expected_word, recognized_text, language)
            row.update(
                expected_text=expected_word,
                correct=1 if result["correct"] else 0,
                recall_accuracy=100.0 if result["correct"] else 0.0,
                filler_count=result["filler_count"],
            )
        elif level == 2:
            expected_sentence = payload.get("expected_text") or ""
            result = voice_analysis.score_sentence_recall(expected_sentence, recognized_text, language)
            row.update(
                expected_text=expected_sentence,
                correct_words=result["correct_words"],
                missing_words=result["missing_words"],
                extra_words=result["extra_words"],
                repeated_words=result["repeated_words"],
                recall_accuracy=result["recall_accuracy"],
                filler_count=result["filler_count"],
            )
        elif level == 3:
            # Level 3 sends the language-neutral category KEY (e.g. "animals"),
            # not the translated display label, so vocabulary lookup works
            # regardless of the selected language.
            category = payload.get("category") or ""
            result = voice_analysis.score_category_round(category, recognized_text, language)
            row.update(
                category=category,
                unique_words=result["unique_valid_word_count"],
                repeated_words=result["repeated_words"],
                valid_word_count=result["valid_word_count"],
                invalid_word_count=result["invalid_word_count"],
                filler_count=result["filler_count"],
            )
            # Trend across this level's rounds, computed once we have this
            # round's valid word count alongside prior rounds in the session.
            db = get_db()
            prior_rounds = db.execute(
                """
                SELECT valid_word_count FROM voice_results
                WHERE user_id = ? AND session_id = ? AND level = 3
                ORDER BY round_number ASC
                """,
                (user_id, session_id),
            ).fetchall()
            round_scores = [int(r["valid_word_count"] or 0) for r in prior_rounds] + [result["valid_word_count"]]
            row["trend"] = voice_analysis.compute_round_trend(round_scores)

        db = get_db()
        db.execute(
            """
            INSERT INTO voice_results
              (user_id, session_id, level, round_number, task_type, category, expected_text,
               recognized_text, correct, correct_words, incorrect_words, missing_words, extra_words,
               unique_words, repeated_words, valid_word_count, invalid_word_count, response_time_ms,
               speech_duration_ms, speech_rate, pause_count, pause_duration_ms, filler_count,
               recall_accuracy, trend, language, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, session_id, level, round_number, row["task_type"], row["category"],
                row["expected_text"], row["recognized_text"], row["correct"], row["correct_words"],
                row["incorrect_words"], row["missing_words"], row["extra_words"], row["unique_words"],
                row["repeated_words"], row["valid_word_count"], row["invalid_word_count"],
                response_time_ms, speech_duration_ms, speech_rate, pause_count, pause_duration_ms,
                row["filler_count"], row["recall_accuracy"], row["trend"], language,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        db.commit()

        return jsonify(status="saved", result=row, speech_rate=speech_rate)

    @app.get("/session/<session_id>/report.pdf")
    def session_report_pdf(session_id: str):
        """Generate a paginated, wrapped PDF report from persisted session records."""
        if not session.get("user_id"):
            return redirect(url_for("login"))
        db = get_db()
        user_id = int(session["user_id"])
        row = session_manager.get_session(db, session_id, user_id)
        if row is None:
            flash("Session report not found.", "error")
            return redirect(url_for("dashboard"))

        participant = db.execute(
            "SELECT id, full_name FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        analysis = performance_analysis.analyze_session(db, user_id, session_id)
        activities = db.execute(
            "SELECT * FROM session_activities WHERE session_id=? AND user_id=? ORDER BY started_at",
            (session_id, user_id),
        ).fetchall()
        monitoring = db.execute(
            "SELECT * FROM webcam_results WHERE session_id=? AND user_id=?",
            (session_id, user_id),
        ).fetchone()
        cursor_rows = db.execute(
            "SELECT * FROM cursor_results WHERE session_id=? AND user_id=? ORDER BY created_at",
            (session_id, user_id),
        ).fetchall()
        events = db.execute(
            "SELECT event_type, severity, flag_number, message, created_at FROM monitoring_events WHERE session_id=? AND user_id=? ORDER BY created_at",
            (session_id, user_id),
        ).fetchall()
        voice_rows = db.execute(
            "SELECT level, round_number, task_type, category, expected_text, recognized_text, correct, recall_accuracy, valid_word_count, unique_words, repeated_words, invalid_word_count, speech_rate, pause_count, pause_duration_ms, response_time_ms, speech_duration_ms, filler_count, created_at FROM voice_results WHERE session_id=? AND user_id=? ORDER BY level, round_number",
            (session_id, user_id),
        ).fetchall()
        game_rows = db.execute(
            "SELECT game_name, score, accuracy, time_taken_seconds, moves, correct_matches, wrong_attempts, correct_responses, wrong_responses, total_responses, created_at FROM game_results WHERE session_id=? AND user_id=? ORDER BY created_at",
            (session_id, user_id),
        ).fetchall()

        buffer = BytesIO()
        page_width, page_height = landscape(A4)
        doc = SimpleDocTemplate(
            buffer, pagesize=landscape(A4), rightMargin=28, leftMargin=28,
            topMargin=34, bottomMargin=32, title="Cognitive Rehabilitation Session Report",
            author="AI Cognitive Rehabilitation System",
        )
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"],
                                  alignment=TA_CENTER, fontSize=17, leading=21, spaceAfter=4))
        styles.add(ParagraphStyle(name="ReportSubTitle", parent=styles["Heading2"],
                                  alignment=TA_CENTER, fontSize=11, leading=14, spaceAfter=10))
        styles.add(ParagraphStyle(name="ReportCell", parent=styles["BodyText"],
                                  fontName="Helvetica", fontSize=7, leading=9, spaceAfter=0,
                                  wordWrap="CJK"))
        styles.add(ParagraphStyle(name="ReportHeader", parent=styles["ReportCell"],
                                  fontName="Helvetica-Bold", textColor=colors.white))
        styles.add(ParagraphStyle(name="ReportSmall", parent=styles["BodyText"],
                                  fontSize=8, leading=10, spaceAfter=3))

        def para(value, header=False):
            if value is None or value == "":
                value = "—"
            return Paragraph(escape(str(value)).replace("\n", "<br/>"),
                             styles["ReportHeader" if header else "ReportCell"])

        def make_table(data, widths, font_size=7):
            wrapped = [[para(cell, header=(ri == 0)) for cell in line]
                       for ri, line in enumerate(data)]
            table = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24527a")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c5d1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f3f7fa")]),
            ]))
            return table

        def draw_page(canvas, document):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#526273"))
            canvas.drawString(28, 18, "AI Cognitive Rehabilitation System · Session Report")
            canvas.drawRightString(page_width - 28, 18, f"Page {document.page}")
            canvas.restoreState()

        story = [
            Paragraph("AI Cognitive Rehabilitation System", styles["ReportTitle"]),
            Paragraph("Guided Cognitive Session Report", styles["ReportSubTitle"]),
        ]
        participant_name = participant["full_name"] if participant else session.get("full_name", "User")
        story.append(Paragraph(
            f"<b>Participant:</b> {escape(str(participant_name))} &nbsp;&nbsp; "
            f"<b>Participant ID:</b> {user_id} &nbsp;&nbsp; "
            f"<b>Session ID:</b> {escape(str(session_id))}", styles["ReportSmall"]))
        story.append(Paragraph(
            f"<b>Start:</b> {escape(str(row['start_time'] or '—'))} &nbsp;&nbsp; "
            f"<b>End:</b> {escape(str(row['end_time'] or 'In progress'))} &nbsp;&nbsp; "
            f"<b>Status:</b> {escape(str(row['status'] or '—'))} &nbsp;&nbsp; "
            f"<b>Duration:</b> {str(row['duration'])+' seconds' if row['duration'] is not None else 'Not completed'}",
            styles["ReportSmall"]))
        score = analysis.get("overall_score")
        story.append(Paragraph(
            f"<b>Overall performance index:</b> {f'{score:.1f}/100' if score is not None else 'Not available'}"
            f" &nbsp;&nbsp; <b>Strongest area:</b> {escape(str(analysis.get('strongest_area') or '—'))}"
            f" &nbsp;&nbsp; <b>Needs practice:</b> {escape(str(analysis.get('weakest_area') or '—'))}",
            styles["ReportSmall"]))
        story.append(Paragraph(
            f"<b>Games completed / recorded:</b> {len(game_rows)} &nbsp;&nbsp; "
            f"<b>Monitoring flags:</b> {len(events)}",
            styles["ReportSmall"]))
        story.append(Paragraph(
            f"<b>Summary:</b> {escape(str(analysis.get('recommendation') or 'No recommendation available.'))}",
            styles["ReportSmall"]))

        story.append(Paragraph("Domain Performance", styles["Heading2"]))
        domain_data = [["Domain", "Score / 100", "Interpretation"]]
        for domain in performance_analysis.DOMAINS:
            val = analysis.get("domain_scores", {}).get(domain)
            domain_data.append([domain, f"{val:.1f}" if val is not None else "Not attempted",
                                "Measured" if val is not None else "No recorded attempt"])
        story.append(make_table(domain_data, [180, 130, 300]))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Games / Recorded Performance", styles["Heading2"]))
        game_data = [["Game", "Score", "Accuracy", "Duration (s)", "Moves", "Correct", "Incorrect", "Responses", "Recorded"]]
        for g_row in game_rows:
            correct = g_row["correct_responses"] if "correct_responses" in g_row.keys() else None
            wrong = g_row["wrong_responses"] if "wrong_responses" in g_row.keys() else None
            total = g_row["total_responses"] if "total_responses" in g_row.keys() else None
            if correct is None: correct = g_row["correct_matches"] or 0
            if wrong is None: wrong = g_row["wrong_attempts"] or 0
            game_data.append([
                g_row["game_name"], g_row["score"], f"{float(g_row['accuracy'] or 0):.1f}%",
                g_row["time_taken_seconds"], g_row["moves"], correct, wrong,
                total if total is not None else "—", g_row["created_at"] or "—"
            ])
        if len(game_data) == 1:
            game_data.append(["No played game records", "—", "—", "—", "—", "—", "—", "—", "—"])
        story.append(make_table(game_data, [125, 48, 58, 65, 45, 48, 52, 60, 135]))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Assessment Timeline", styles["Heading2"]))
        activity_data = [["Activity", "Start", "End", "Duration (s)", "Completion status"]]
        for a in activities:
            activity_data.append([
                a["activity_key"].replace("_", " ").title(), a["started_at"] or "—",
                a["ended_at"] or "—", a["duration_seconds"] if a["duration_seconds"] is not None else "—",
                a["status"] or "—"
            ])
        if len(activity_data) == 1:
            activity_data.append(["No activity records", "—", "—", "—", "—"])
        story.append(make_table(activity_data, [155, 185, 185, 85, 115]))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Webcam Monitoring Metrics", styles["Heading2"]))
        if monitoring:
            m = monitoring
            keys = m.keys()
            metric_rows = [
                ["Metric", "Recorded value"],
                ["Monitoring duration", f"{m['monitoring_seconds'] or 0} seconds"],
                ["Identity checks / successful matches / mismatches",
                 f"{m['face_check_attempts'] or 0} / {m['face_verified_checks'] or 0} / {m['face_mismatch_events'] or 0}"],
                ["No-face incidents", str(m["face_missing_events"] or 0)],
                ["Multiple-face incidents", str(m["multiple_face_events"] or 0)],
                ["Blink count", str(m["blink_count"] or 0)],
                ["Eye-closure duration", f"{round((m['eyes_closed_ms'] or 0)/1000, 1)} seconds"],
                ["Gaze center / left / right / away",
                 f"{m['gaze_center_count'] or 0} / {m['gaze_left_count'] or 0} / {m['gaze_right_count'] or 0} / {m['gaze_away_count'] or 0}"],
                ["Gaze / distraction / attention events",
                 f"{m['gaze_away_count'] or 0} / {m['distraction_events'] or 0} / {m['attention_events'] or 0}"],
                ["Yawn events", str(m["yawn_events"] or 0)],
                ["Head turned left / right (informational)", f"{m['head_turned_left_events'] if 'head_turned_left_events' in keys else 0} / {m['head_turned_right_events'] if 'head_turned_right_events' in keys else 0}"],
                ["Total flags", str(m["total_flags"] or 0)],
            ]
            story.append(make_table(metric_rows, [300, 420]))
        else:
            story.append(Paragraph("No webcam monitoring metrics were recorded.", styles["ReportSmall"]))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Voice Cognitive Game Results", styles["Heading2"]))
        voice_data = [["Level", "Round", "Task", "Expected / category", "Recognized response", "Result", "Response ms", "Speech ms", "Rate WPM", "Pauses", "Recorded"]]
        for v in voice_rows:
            if v["level"] == 1:
                result = ("Correct" if v["correct"] else "Incorrect") + f" ({float(v['recall_accuracy'] or 0):.1f}%)"
            elif v["level"] == 2:
                result = f"Recall {float(v['recall_accuracy'] or 0):.1f}%"
            else:
                result = f"{v['valid_word_count'] or 0} valid; {v['unique_words'] or 0} unique"
            voice_data.append([
                v["level"], v["round_number"], v["task_type"], v["expected_text"] or v["category"] or "—",
                v["recognized_text"] or "—", result, v["response_time_ms"] or 0,
                v["speech_duration_ms"] or 0, v["speech_rate"] or 0,
                f"{v['pause_count'] or 0} / {v['pause_duration_ms'] or 0} ms", v["created_at"] or "—"
            ])
        if len(voice_data) == 1:
            voice_data.append(["No voice results", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"])
        story.append(make_table(voice_data, [30, 30, 65, 105, 125, 65, 50, 50, 42, 58, 145], 6.5))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Cursor / Mouse Interaction Signals", styles["Heading2"]))
        cursor_data = [["Game", "Reaction ms", "Click accuracy", "Click errors", "Target accuracy", "Hesitation ms", "Missed", "Double-click errors", "Distance px", "Speed px/s", "Recorded"]]
        for c in cursor_rows:
            cursor_data.append([
                c["game_name"], f"{c['reaction_time']:.0f}" if c["reaction_time"] is not None else "—",
                f"{c['click_accuracy']:.1f}%" if c["click_accuracy"] is not None else "—",
                c["incorrect_clicks"] or 0,
                f"{c['target_selection_accuracy']:.1f}%" if c["target_selection_accuracy"] is not None else "—",
                f"{c['hesitation_time']:.0f}" if c["hesitation_time"] is not None else "—",
                c["missed_clicks"] or 0, c["double_click_errors"] or 0,
                round(c["cursor_distance"] or 0),
                f"{c['cursor_speed']:.1f}" if c["cursor_speed"] is not None else "—",
                c["created_at"] or "—"
            ])
        if len(cursor_data) == 1:
            cursor_data.append(["No cursor measurements", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"])
        story.append(make_table(cursor_data, [100, 55, 65, 55, 70, 65, 40, 70, 55, 55, 100], 6.5))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Monitoring Flag Events", styles["Heading2"]))
        event_data = [["Timestamp", "Event type", "Description", "Severity", "Flag number"]]
        for e in events:
            event_data.append([
                e["created_at"] or "—", e["event_type"] or "—",
                e["message"] or e["event_type"] or "—", e["severity"] or "—",
                e["flag_number"] if e["flag_number"] is not None else "—"
            ])
        if len(event_data) == 1:
            event_data.append(["No flags recorded", "—", "—", "—", "—"])
        story.append(make_table(event_data, [135, 130, 340, 75, 60]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "Scores are project-defined performance indices for progress tracking and are not clinical or diagnostic measures.",
            styles["ReportSmall"]))
        doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
        buffer.seek(0)
        filename = f"Cognitive_Rehabilitation_Session_{session_id[:8]}.pdf"
        return send_file(buffer, as_attachment=True, download_name=filename, mimetype="application/pdf")

    @app.get("/voice-game/next-round")
    def voice_game_next_round():
        """Serve a fresh prompt for the next Level 1/2 word/sentence or Level 3 category."""
        level = request.args.get("level", type=int) or 1
        language = voice_content.normalize_language(request.args.get("lang"))
        if level == 1:
            return jsonify(text=voice_content.level1_word(language))
        if level == 2:
            return jsonify(text=voice_content.level2_sentence(language))
        category_key = voice_content.level3_category_key(language)
        return jsonify(category=category_key, label=voice_content.level3_category_label(language, category_key))

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
