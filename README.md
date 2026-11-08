# AI Cognitive Rehabilitation System

Flask + SQLite cognitive-training application with face enrollment and verification, Memory Test, Reaction Test, Memory Match Game, Voice Cognitive Game, session history, webcam monitoring, and PDF session reports.

## Windows installation

Requirements: 64-bit Python 3.11, the Python launcher (`py`), and internet access to install Python packages and load the browser-side MediaPipe model from its CDN.

From PowerShell in the extracted `game` folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Alternatively, run `setup_windows.bat` once and then `run_windows.bat`. Open `http://127.0.0.1:5000` in Chrome or Edge and allow camera access. The bundled FaceNet VGGFace2 checkpoint is at `models/facenet/facenet_vggface2.pt`; the setup script checks it and downloads it only if missing. Browser-side MediaPipe Tasks Vision is loaded from a CDN, so the browser needs network access for that model.

## Main components

- `app.py` — Flask routes, authentication, SQLite schema/migrations, game saves, monitoring events, session report and PDF export.
- `face_verification.py` — MTCNN detection and FaceNet 512-dimensional embeddings. Login and monitoring compare against the authenticated account's saved face embeddings. Monitoring checks return a match result for every detected face; detection alone never authorizes a face.
- `session_manager.py` — session lifecycle, activity ordering, and session completion.
- `performance_analysis.py` — domain summaries and project-defined performance indices.
- `voice_analysis.py` — voice response analysis.
- `static/js/webcam-monitor.js` — live webcam panel, face boxes, identity status, eye closure timing, event debouncing, and monitoring summary.
- `static/js/cursor-monitor.js`, `static/js/voice-recorder.js` — interaction and voice metrics.
- `templates/` and `static/` — game screens, dashboard, reports, styles and browser scripts.
- `rehab.db` — existing SQLite database and records.
- `dataset/lfw/` — bundled LFW dataset.
- `models/facenet/facenet_vggface2.pt` — bundled FaceNet checkpoint.

## Monitoring behavior

- The logged-in name and session owner are populated from the authenticated Flask session. The server uses `session['user_id']` and verifies that the requested active session belongs to that user.
- `VERIFIED USER` is shown only after the FaceNet embedding matches the authenticated account's enrolled embedding. A different person, including another registered account, is `UNKNOWN PERSON` for this session.
- Multiple faces produce `MULTIPLE PEOPLE DETECTED`; each face is independently compared with the current account's enrollment, so a matching face can have a green box and other faces red boxes.
- No face, identity mismatch, multiple faces, and verifier failure use separate statuses. Verifier failures show `VERIFICATION UNAVAILABLE`, not a green verified state.
- The current cosine-distance threshold is 0.30. This is a conservative project setting, not a claim of independently calibrated accuracy; validate it with representative genuine and impostor samples before real-world deployment.
- Eye closure is flagged after five uninterrupted seconds; reopening resets the timer. Normal blinking is not a flag. Gaze direction, head pose, cursor movement and informational focus messages do not independently create flags.
- Monitoring events are stored against the authenticated user and active session, with short server-side debounce for duplicate writes. The existing three-flag limit remains configurable in `WebcamMonitor`.

## Reports and scores

The PDF export uses landscape A4, wrapped Paragraph table cells, repeated table headers, margins, and page numbers. It includes participant/session details, attempted game results, activity timeline, monitoring metrics, voice and cursor results, and timestamped flag descriptions. Unplayed games are not inserted as failed attempts. Scores and domain indices are project-defined progress metrics, not clinical or diagnostic measures.

The Reaction Test score uses the stated 60/20/20 weighting once (speed, successful rounds, early-click control). Accuracy calculations use recorded attempts and are bounded to valid ranges.

## Checks performed for this revision

- Python source files parsed successfully.
- `static/js/webcam-monitor.js` passed Node.js syntax checking.
- Jinja templates parsed successfully.
- Existing `rehab.db` passed SQLite `PRAGMA integrity_check`.
- Route references in source/templates were checked; only Flask's built-in `static` endpoint is outside the user-defined route list.
- A ReportLab layout smoke test with long wrapped rows and repeated headers produced a readable multi-page PDF.
- The original ZIP passed its archive integrity test.

## Runtime tests still required

The current build environment does not have Flask or `facenet-pytorch` installed and cannot reach package indexes, and it has no camera/browser session. Therefore a full Flask test-client run, real FaceNet embedding comparison, live webcam test, and actual `/session/<id>/report.pdf` route test were not run here. After installing requirements on Windows, test with a real camera and a few enrolled identities:

1. Log in and confirm the correct account name and session owner appear.
2. Show the enrolled user alone; verify a green box only after a successful match.
3. Replace them with a different person and confirm `UNKNOWN PERSON`, red box, and one identity-mismatch flag per incident.
4. Show two people; confirm `MULTIPLE PEOPLE DETECTED`, per-face results, and one multiple-person flag for the incident.
5. Keep eyes continuously closed for five seconds; confirm one flag. Reopen eyes and verify the timer resets. Normal blinks must not flag.
6. Look away, turn the head, move the cursor, or trigger an informational focus message; confirm the flag count does not increase from those actions alone.
7. Complete one or more games; verify only attempted/completed games and recorded response metrics appear in the session summary.
8. Download the PDF with long names and multiple flag events; confirm all rows, repeated headers, and page numbers are visible.
9. Deny camera permission or interrupt network/model loading; confirm the panel reports unavailable status and never remains green without a successful verification.


## Updated face enrollment and voice prompts

- New face enrollments collect 10 samples across five poses: front, slight left, slight right, chin up, and chin down (two samples per pose). Re-enroll existing accounts from the Face Enrollment page to benefit from multi-pose matching.
- Monitoring compares each detected face against the authenticated account's saved enrollment samples. The cosine-distance threshold is configurable in `face_verification.py` and should be calibrated with genuine and impostor samples for the deployment camera and lighting.
- Voice Level 3 now includes six rounds with a larger category pool; word and sentence prompt pools have also been expanded.
- Webcam summary saving is idempotent across overlapping stop callbacks, logs HTTP failures, and permits an owned session's final summary to arrive immediately after termination.
