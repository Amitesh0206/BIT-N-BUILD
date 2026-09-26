import cv2
from datetime import datetime, timedelta
from html import escape
import mediapipe as mp
from pathlib import Path
import time


ON_SCREEN = "ON SCREEN"
ATTENTION_DIVERTED = "ATTENTION DIVERTED"
FACE_NOT_DETECTED = "FACE NOT DETECTED"
DIVERSION_START_SECONDS = 1.0
RETURN_TO_SCREEN_SECONDS = 0.4
GAZE_DIVERSION_START_SECONDS = 1.0
GAZE_RETURN_TO_CENTER_SECONDS = 0.3
GAZE_CALIBRATION_SECONDS = 1.0
GAZE_DIVERSION_THRESHOLD = 0.10
GAZE_CENTER_RETURN_THRESHOLD = 0.06
GAZE_CENTER = "GAZE CENTER"
GAZE_DIVERTED = "GAZE DIVERTED"
GAZE_UNAVAILABLE = "GAZE UNAVAILABLE"


def estimate_head_turn(landmarks):
    nose = landmarks[1]
    left_eye_corner = landmarks[33]
    right_eye_corner = landmarks[263]
    eye_width = abs(left_eye_corner.x - right_eye_corner.x)

    if eye_width == 0:
        return None

    eye_midpoint = (left_eye_corner.x + right_eye_corner.x) / 2
    return (nose.x - eye_midpoint) / eye_width


def estimate_gaze_position(landmarks):
    eye_pairs = ((33, 133, range(468, 473)), (362, 263, range(473, 478)))
    eye_positions = []

    for first_corner, second_corner, iris_indices in eye_pairs:
        first = landmarks[first_corner]
        second = landmarks[second_corner]
        if first.x <= second.x:
            eye_left, eye_right = first, second
        else:
            eye_left, eye_right = second, first

        eye_dx = eye_right.x - eye_left.x
        eye_dy = eye_right.y - eye_left.y
        eye_length_squared = eye_dx * eye_dx + eye_dy * eye_dy
        if eye_length_squared <= 0:
            return None

        iris_x = sum(landmarks[index].x for index in iris_indices) / 5
        iris_y = sum(landmarks[index].y for index in iris_indices) / 5
        iris_position = (
            (iris_x - eye_left.x) * eye_dx + (iris_y - eye_left.y) * eye_dy
        ) / eye_length_squared
        eye_positions.append(iris_position)

    return tuple(eye_positions)


class GazeMonitor:
    def __init__(self):
        self.status = GAZE_UNAVAILABLE
        self.pending_direction = None
        self.pending_since = None
        self.center_since = None
        self.diverted_since = None
        self.direction = None
        self.calibration_started = None
        self.calibration_samples = []
        self.center_position = None

    def update(self, face_found, eye_positions, now):
        messages = []

        if not face_found or eye_positions is None:
            self.pending_direction = None
            self.pending_since = None
            self.center_since = None
            if self.center_position is None:
                self.calibration_started = None
                self.calibration_samples.clear()

            if self.status == GAZE_DIVERTED:
                duration = now - self.diverted_since
                messages.append(
                    f"GAZE DIVERTED - ended after {duration:.1f} seconds (gaze unavailable)"
                )
                self.diverted_since = None
                self.direction = None
            if self.status != GAZE_UNAVAILABLE:
                self.status = GAZE_UNAVAILABLE
                messages.append(GAZE_UNAVAILABLE)

            return self.status, self.direction, messages

        if self.center_position is None:
            if self.calibration_started is None:
                self.calibration_started = now
            self.calibration_samples.append(eye_positions)
            if now - self.calibration_started < GAZE_CALIBRATION_SECONDS:
                self.status = GAZE_CENTER
                return self.status, "CENTER", messages

            self.center_position = tuple(
                sum(sample[index] for sample in self.calibration_samples)
                / len(self.calibration_samples)
                for index in range(2)
            )
            self.calibration_samples.clear()
            self.status = GAZE_CENTER
            messages.append("Gaze baseline calibrated; keep head facing the screen.")
            messages.append(GAZE_CENTER)

        eye_offsets = tuple(
            eye_positions[index] - self.center_position[index]
            for index in range(2)
        )
        gaze_offset = sum(eye_offsets) / 2

        if self.status == GAZE_UNAVAILABLE:
            self.status = GAZE_CENTER
            messages.append(GAZE_CENTER)

        if gaze_offset <= -GAZE_DIVERSION_THRESHOLD:
            observed_direction = "LEFT"
        elif gaze_offset >= GAZE_DIVERSION_THRESHOLD:
            observed_direction = "RIGHT"
        elif abs(gaze_offset) <= GAZE_CENTER_RETURN_THRESHOLD:
            observed_direction = "CENTER"
        else:
            observed_direction = None

        if self.status == GAZE_DIVERTED:
            if observed_direction == "CENTER":
                self.pending_direction = None
                self.pending_since = None
                if self.center_since is None:
                    self.center_since = now
                elif now - self.center_since >= GAZE_RETURN_TO_CENTER_SECONDS:
                    duration = now - self.diverted_since
                    self.status = GAZE_CENTER
                    self.diverted_since = None
                    self.direction = None
                    self.center_since = None
                    messages.append(
                        f"GAZE DIVERTED - ended after {duration:.1f} seconds"
                    )
                    messages.append(GAZE_CENTER)
            else:
                self.center_since = None
                if observed_direction in ("LEFT", "RIGHT"):
                    if observed_direction != self.direction:
                        if self.pending_direction != observed_direction:
                            self.pending_direction = observed_direction
                            self.pending_since = now
                        elif now - self.pending_since >= GAZE_DIVERSION_START_SECONDS:
                            self.direction = observed_direction
                            self.pending_direction = None
                            self.pending_since = None
                            messages.append(
                                f"GAZE DIVERTED - direction changed to {self.direction}"
                            )
                else:
                    self.pending_direction = None
                    self.pending_since = None
        elif observed_direction in ("LEFT", "RIGHT"):
            if self.pending_direction != observed_direction:
                self.pending_direction = observed_direction
                self.pending_since = now
            elif now - self.pending_since >= GAZE_DIVERSION_START_SECONDS:
                self.status = GAZE_DIVERTED
                self.direction = observed_direction
                self.diverted_since = now
                self.pending_direction = None
                self.pending_since = None
                messages.append(f"GAZE DIVERTED - {self.direction} started")
        else:
            self.pending_direction = None
            self.pending_since = None

        return self.status, observed_direction or "CENTER", messages


class AttentionMonitor:
    def __init__(self):
        self.status = FACE_NOT_DETECTED
        self.away_since = None
        self.screen_since = None
        self.diverted_since = None
        self.head_is_turned = False

    def update(self, face_found, head_turn, now):
        messages = []

        if not face_found:
            self.away_since = None
            self.screen_since = None
            self.head_is_turned = False

            if self.status == ATTENTION_DIVERTED:
                duration = now - self.diverted_since
                messages.append(
                    f"ATTENTION DIVERTED - ended after {duration:.1f} seconds"
                )
                self.diverted_since = None
            if self.status != FACE_NOT_DETECTED:
                self.status = FACE_NOT_DETECTED
                messages.append(FACE_NOT_DETECTED)

            return self.status, 0.0, messages

        if self.status == FACE_NOT_DETECTED:
            self.status = ON_SCREEN
            messages.append(ON_SCREEN)

        if head_turn is not None:
            if abs(head_turn) >= 0.20:
                self.head_is_turned = True
            elif abs(head_turn) <= 0.14:
                self.head_is_turned = False

        if self.head_is_turned:
            self.screen_since = None
            if self.away_since is None:
                self.away_since = now

            if (
                self.status != ATTENTION_DIVERTED
                and now - self.away_since >= DIVERSION_START_SECONDS
            ):
                self.status = ATTENTION_DIVERTED
                self.diverted_since = now
                messages.append("ATTENTION DIVERTED - started")
        else:
            self.away_since = None
            if self.status == ATTENTION_DIVERTED:
                if self.screen_since is None:
                    self.screen_since = now
                elif now - self.screen_since >= RETURN_TO_SCREEN_SECONDS:
                    duration = now - self.diverted_since
                    self.status = ON_SCREEN
                    self.diverted_since = None
                    self.screen_since = None
                    messages.append(
                        f"ATTENTION DIVERTED - ended after {duration:.1f} seconds"
                    )

        duration = (
            now - self.diverted_since
            if self.status == ATTENTION_DIVERTED
            else 0.0
        )
        return self.status, duration, messages


class SessionLogger:
    FACE_MISSING_PERSISTENCE_SECONDS = 1.0

    def __init__(self, started_at=None, wall_started_at=None):
        self.started_at = started_at if started_at is not None else time.monotonic()
        self.wall_started_at = wall_started_at or datetime.now()
        self.last_update = self.started_at
        self.last_face_found = True
        self.last_head_diverted = False
        self.last_gaze_diverted = False
        self.on_screen_seconds = 0.0
        self.attention_diverted_seconds = 0.0
        self.face_not_detected_seconds = 0.0
        self.events = []
        self.active_events = {}
        self.face_missing_since = None

    @staticmethod
    def _format_timestamp(value):
        return value.strftime("%Y-%m-%d %H:%M:%S")

    def _wall_time_at(self, monotonic_time, wall_now, now):
        return wall_now - timedelta(seconds=now - monotonic_time)

    def _start_event(self, event_type, now, wall_now, start_time=None):
        event_start = now if start_time is None else start_time
        self.active_events[event_type] = {
            "type": event_type,
            "start_monotonic": event_start,
            "start_timestamp": self._format_timestamp(
                self._wall_time_at(event_start, wall_now, now)
            ),
        }

    def _end_event(self, event_type, now, wall_now):
        event = self.active_events.pop(event_type, None)
        if event is None:
            return
        event["end_timestamp"] = self._format_timestamp(wall_now)
        event["duration_seconds"] = max(0.0, now - event["start_monotonic"])
        self.events.append(event)
        print(
            f"{event_type} - ended after "
            f"{event['duration_seconds']:.1f} seconds"
        )

    def _account_time(self, now):
        elapsed = max(0.0, now - self.last_update)
        if self.last_face_found:
            if self.last_head_diverted or self.last_gaze_diverted:
                self.attention_diverted_seconds += elapsed
            else:
                self.on_screen_seconds += elapsed
        else:
            self.face_not_detected_seconds += elapsed
        self.last_update = now

    def update(self, now, wall_now, face_found, head_diverted, gaze_diverted):
        self._account_time(now)

        if head_diverted and not self.last_head_diverted:
            self._start_event("Head Direction Diverted", now, wall_now)
            print("Head Direction Diverted - started")
        elif self.last_head_diverted and not head_diverted:
            self._end_event("Head Direction Diverted", now, wall_now)

        if gaze_diverted and not self.last_gaze_diverted:
            self._start_event("Gaze Diverted", now, wall_now)
            print("Gaze Diverted - started")
        elif self.last_gaze_diverted and not gaze_diverted:
            self._end_event("Gaze Diverted", now, wall_now)

        if not face_found:
            if self.face_missing_since is None:
                self.face_missing_since = now
            elif (
                "Face Not Detected" not in self.active_events
                and now - self.face_missing_since
                >= self.FACE_MISSING_PERSISTENCE_SECONDS
            ):
                missing_start = self.face_missing_since
                self._start_event(
                    "Face Not Detected", now, wall_now, start_time=missing_start
                )
                print("Face Not Detected - started")
        else:
            self.face_missing_since = None
            if "Face Not Detected" in self.active_events:
                self._end_event("Face Not Detected", now, wall_now)

        self.last_face_found = face_found
        self.last_head_diverted = head_diverted
        self.last_gaze_diverted = gaze_diverted

    def summary(self, ended_at=None):
        ended_at = ended_at if ended_at is not None else time.monotonic()
        duration = max(0.0, ended_at - self.started_at)
        on_screen_percent = (
            self.on_screen_seconds / duration * 100 if duration else 0.0
        )
        diverted_percent = (
            self.attention_diverted_seconds / duration * 100 if duration else 0.0
        )
        diversion_events = [
            event
            for event in self.events
            if event["type"] in ("Head Direction Diverted", "Gaze Diverted")
        ]
        return {
            "session_duration": duration,
            "on_screen_seconds": self.on_screen_seconds,
            "attention_diverted_seconds": self.attention_diverted_seconds,
            "face_not_detected_seconds": self.face_not_detected_seconds,
            "on_screen_percent": on_screen_percent,
            "attention_diverted_percent": diverted_percent,
            "head_diversion_count": sum(
                event["type"] == "Head Direction Diverted"
                for event in self.events
            ),
            "gaze_diversion_count": sum(
                event["type"] == "Gaze Diverted" for event in self.events
            ),
            "face_not_detected_count": sum(
                event["type"] == "Face Not Detected" for event in self.events
            ),
            "longest_diversion": max(
                (event["duration_seconds"] for event in diversion_events),
                default=0.0,
            ),
        }

    @staticmethod
    def _format_duration(seconds):
        total_seconds = int(seconds)
        minutes, remaining_seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def render_html(self, summary):
        rows = []
        for event in sorted(
            self.events, key=lambda item: item["start_monotonic"]
        ):
            rows.append(
                "<tr>"
                f"<td>{escape(event['start_timestamp'])}</td>"
                f"<td>{escape(event['end_timestamp'])}</td>"
                f"<td>{escape(event['type'])}</td>"
                f"<td>{event['duration_seconds']:.1f} s</td>"
                "</tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="4" class="empty">No confirmed events recorded.</td></tr>')

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interview Integrity Report</title>
  <style>
    :root {{ color-scheme: light; --ink: #172033; --muted: #64748b; --line: #e2e8f0; --blue: #2563eb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f1f5f9; color: var(--ink); font: 15px/1.5 Arial, sans-serif; }}
    main {{ max-width: 980px; margin: 40px auto; padding: 0 20px 40px; }}
    header {{ padding: 30px; background: #14233b; border-radius: 18px; color: white; }}
    header p {{ margin: 8px 0 0; color: #cbd5e1; }}
    h1 {{ margin: 0; font-size: 30px; }}
    h2 {{ margin: 0 0 16px; font-size: 19px; }}
    section {{ margin-top: 22px; padding: 24px; background: white; border: 1px solid var(--line); border-radius: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ padding: 18px; background: #f8fafc; border: 1px solid var(--line); border-radius: 12px; }}
    .label {{ color: var(--muted); font-size: 13px; }}
    .value {{ display: block; margin-top: 4px; font-size: 25px; font-weight: 700; }}
    .counts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .count {{ padding: 13px 15px; border-radius: 10px; background: #eff6ff; }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid var(--line); }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
    .empty {{ color: var(--muted); text-align: center; }}
    .note {{ color: var(--muted); font-size: 13px; }}
    footer {{ margin-top: 20px; color: var(--muted); font-size: 13px; text-align: center; }}
    @media (max-width: 600px) {{ main {{ margin-top: 18px; padding: 0 12px 24px; }} section {{ padding: 17px; overflow-x: auto; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Interview Integrity Report</h1>
      <p>Behavioral attention signals for transparent human review</p>
      <p>Session started: {escape(self._format_timestamp(self.wall_started_at))}</p>
    </header>
    <section>
      <h2>Session Summary</h2>
      <div class="cards">
        <div class="card"><span class="label">Session duration</span><span class="value">{self._format_duration(summary['session_duration'])}</span></div>
        <div class="card"><span class="label">Approximately on-screen</span><span class="value">{summary['on_screen_percent']:.1f}%</span></div>
        <div class="card"><span class="label">Attention diverted</span><span class="value">{summary['attention_diverted_percent']:.1f}%</span></div>
        <div class="card"><span class="label">Longest diversion</span><span class="value">{summary['longest_diversion']:.1f} s</span></div>
      </div>
      <p class="note">Percentages are calculated from total session duration. Face-not-detected time is reported separately; overlapping head and gaze signals count once toward attention-diverted time.</p>
    </section>
    <section>
      <h2>Behavioral Event Counts</h2>
      <div class="counts">
        <div class="count">Head Direction Diverted: <strong>{summary['head_diversion_count']}</strong></div>
        <div class="count">Gaze Diverted: <strong>{summary['gaze_diversion_count']}</strong></div>
        <div class="count">Face Not Detected: <strong>{summary['face_not_detected_count']}</strong></div>
      </div>
    </section>
    <section>
      <h2>Event Timeline</h2>
      <table>
        <thead><tr><th>Start</th><th>End</th><th>Behavioral attention signal</th><th>Duration</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    <footer>Approximate behavioral observations only; not a determination of misconduct.</footer>
  </main>
</body>
</html>
"""

    def finish(self, report_path, now=None, wall_now=None):
        now = time.monotonic() if now is None else now
        wall_now = datetime.now() if wall_now is None else wall_now
        self._account_time(now)
        for event_type in tuple(self.active_events):
            self._end_event(event_type, now, wall_now)

        self.events.sort(key=lambda item: item["start_monotonic"])
        result = self.summary(now)
        report_path.write_text(self.render_html(result), encoding="utf-8")

        print("\nINTERVIEW INTEGRITY REPORT")
        print(f"Session duration: {self._format_duration(result['session_duration'])}")
        print(
            f"Approximately on-screen: {result['on_screen_seconds']:.1f} seconds "
            f"({result['on_screen_percent']:.1f}%)"
        )
        print(
            f"Attention diverted: {result['attention_diverted_seconds']:.1f} seconds "
            f"({result['attention_diverted_percent']:.1f}%)"
        )
        print(f"Face not detected: {result['face_not_detected_seconds']:.1f} seconds")
        print(f"Head diversion events: {result['head_diversion_count']}")
        print(f"Gaze diversion events: {result['gaze_diversion_count']}")
        print(f"Face-not-detected events: {result['face_not_detected_count']}")
        print(f"Longest diversion: {result['longest_diversion']:.1f} seconds")
        print(f"Recruiter report saved to: {report_path}")
        return result


def main():
    print("Starting camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open the webcam. Check that it is connected and not being used by another app.")

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    monitor = AttentionMonitor()
    gaze_monitor = GazeMonitor()
    session_logger = SessionLogger()
    report_path = Path(__file__).with_name("recruiter_report.html")
    print("Look toward the screen for the first second to calibrate gaze.")
    print("Session started. Press Q in the webcam window to finish and save the report.")
    print(FACE_NOT_DETECTED)
    next_debug_time = 0.0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            faces = results.multi_face_landmarks
            face_found = bool(faces)
            head_turn = None
            eye_positions = None
            left_iris_position = None
            right_iris_position = None

            if face_found:
                landmarks = faces[0]
                head_turn = estimate_head_turn(landmarks.landmark)
                eye_positions = estimate_gaze_position(landmarks.landmark)
                if eye_positions is not None:
                    left_iris_position, right_iris_position = eye_positions
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(100, 100, 100), thickness=1
                    ),
                )
                gaze_landmark_color = (
                    (0, 200, 255)
                    if gaze_monitor.status == GAZE_DIVERTED
                    else (0, 220, 0)
                )
                for eye_connections in (
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_face_mesh.FACEMESH_LEFT_EYE,
                    mp_face_mesh.FACEMESH_RIGHT_EYE,
                    mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                    mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                    mp_face_mesh.FACEMESH_LEFT_IRIS,
                    mp_face_mesh.FACEMESH_RIGHT_IRIS,
                ):
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=landmarks,
                        connections=eye_connections,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=gaze_landmark_color, thickness=1
                        ),
                    )
                for iris_center in (468, 473):
                    point = landmarks.landmark[iris_center]
                    cv2.circle(
                        frame,
                        (int(point.x * frame.shape[1]), int(point.y * frame.shape[0])),
                        4,
                        (0, 255, 255),
                        -1,
                    )

            now = time.monotonic()
            status, diverted_duration, messages = monitor.update(
                face_found, head_turn, now
            )
            for message in messages:
                print(message)
            gaze_status, gaze_direction, gaze_messages = gaze_monitor.update(
                face_found, eye_positions, now
            )
            for message in gaze_messages:
                print(message)
            session_logger.update(
                now,
                datetime.now(),
                face_found,
                status == ATTENTION_DIVERTED,
                gaze_status == GAZE_DIVERTED,
            )

            if not face_found:
                head_direction = "N/A"
                gaze_direction = "N/A"
                display_status = FACE_NOT_DETECTED
            else:
                head_direction = (
                    "LEFT" if head_turn < 0 else "RIGHT"
                ) if monitor.head_is_turned and head_turn is not None else "CENTER"
                gaze_direction = gaze_direction or "CENTER"
                if status == ATTENTION_DIVERTED:
                    display_status = ATTENTION_DIVERTED
                elif gaze_status == GAZE_DIVERTED:
                    display_status = GAZE_DIVERTED
                else:
                    display_status = ON_SCREEN

            status_color = {
                ON_SCREEN: (60, 220, 60),
                GAZE_DIVERTED: (0, 200, 255),
                ATTENTION_DIVERTED: (0, 120, 255),
                FACE_NOT_DETECTED: (180, 180, 180),
            }[display_status]
            panel = frame.copy()
            cv2.rectangle(panel, (10, 10), (500, 180), (18, 24, 32), -1)
            cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
            head_label = head_direction if face_found else "N/A"
            cv2.putText(frame, f"Head: {head_label}", (22, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
            gaze_label = (
                "DIVERTED " + gaze_direction
                if gaze_status == GAZE_DIVERTED
                else gaze_direction
            )
            cv2.putText(frame, f"Gaze: {gaze_label}", (22, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            cv2.putText(frame, f"Status: {display_status}", (22, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            session_duration = now - session_logger.started_at
            cv2.putText(
                frame,
                f"Session time: {SessionLogger._format_duration(session_duration)}",
                (22, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )
            if status == ATTENTION_DIVERTED:
                cv2.putText(
                    frame,
                    f"Head diversion: {diverted_duration:.1f}s",
                    (250, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    status_color,
                    1,
                )

            if now >= next_debug_time:
                if face_found and eye_positions is not None:
                    print(
                        f"Head direction: {head_direction}; "
                        f"Left iris position: {left_iris_position:.2f}; "
                        f"Right iris position: {right_iris_position:.2f}; "
                        f"Gaze direction: {gaze_direction}; "
                        f"Gaze diverted: {'YES' if gaze_status == GAZE_DIVERTED else 'NO'}"
                    )
                else:
                    print(
                        f"Head direction: N/A; Left iris position: N/A; "
                        f"Right iris position: N/A; Gaze direction: N/A; "
                        f"Gaze diverted: NO"
                    )
                next_debug_time = now + 1.0

            cv2.putText(
                frame,
                "Head and gaze are approximate attention signals, not cheating verdicts.",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
            )

            cv2.imshow("Interview Integrity - Attention Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()

    session_logger.finish(report_path)


if __name__ == "__main__":
    main()
