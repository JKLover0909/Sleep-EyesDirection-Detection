"""
FQC operator attention engine.

- Sleep / nodding off: Eye Aspect Ratio (EAR) + rolling PERCLOS + head-down pitch
- Eye direction / looking at screen: iris position ratios inside eye lids

Algorithms adapted from:
- e-candeloro/Driver-State-Detection (EAR, PERCLOS timers) — MIT
- MohamedASAK/Gaze-Detection (iris horizontal/vertical ratios)

Uses MediaPipe Tasks FaceLandmarker (0.10.33+; solutions.FaceMesh removed).
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from numpy.linalg import norm

from database import SessionLocal, create_event

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(BASE_DIR, "data", "sample_videos")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "snapshots")
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

LEFT_EYE = [33, 133, 160, 144, 158, 153]
RIGHT_EYE = [362, 263, 385, 380, 387, 373]
LEFT_IRIS = 468
RIGHT_IRIS = 473


@dataclass
class FrameMetrics:
    ear: float | None = None
    gaze_direction: str = "Unknown"
    gaze_h: float | None = None
    gaze_v: float | None = None
    looking_at_screen: bool = False
    asleep: bool = False
    tired: bool = False
    looking_away: bool = False
    perclos: float = 0.0
    pitch_proxy: float | None = None
    face_detected: bool = False
    fps: float = 0.0
    alerts: list[str] = field(default_factory=list)


class AttentionScorer:
    def __init__(
        self,
        ear_thresh: float = 0.21,
        ear_time_thresh: float = 2.0,
        gaze_away_time_thresh: float = 2.0,
        perclos_thresh: float = 0.25,
        perclos_window_sec: float = 30.0,
        decay: float = 0.88,
    ):
        self.ear_thresh = ear_thresh
        self.ear_time_thresh = ear_time_thresh
        self.gaze_away_time_thresh = gaze_away_time_thresh
        self.perclos_thresh = perclos_thresh
        self.perclos_window_sec = perclos_window_sec
        self.decay = decay
        self.closure_time = 0.0
        self.away_time = 0.0
        self.last_t = time.time()
        self.timestamps = np.empty((0,), dtype=np.float64)
        self.closed_flags = np.empty((0,), dtype=bool)

    def update(self, t_now: float, ear: float | None, looking_at_screen: bool):
        elapsed = max(0.0, t_now - self.last_t)
        self.last_t = t_now

        eyes_closed = ear is not None and ear <= self.ear_thresh
        if eyes_closed:
            self.closure_time += elapsed
        else:
            self.closure_time *= self.decay

        if not looking_at_screen and ear is not None:
            self.away_time += elapsed
        else:
            self.away_time *= self.decay

        self.timestamps = np.concatenate((self.timestamps, [t_now]))
        self.closed_flags = np.concatenate((self.closed_flags, [eyes_closed]))
        valid = self.timestamps >= (t_now - self.perclos_window_sec)
        self.timestamps = self.timestamps[valid]
        self.closed_flags = self.closed_flags[valid]
        perclos = float(np.mean(self.closed_flags)) if self.timestamps.size else 0.0

        asleep = self.closure_time >= self.ear_time_thresh
        tired = perclos >= self.perclos_thresh and self.timestamps.size > 15
        looking_away = self.away_time >= self.gaze_away_time_thresh
        return asleep, tired, looking_away, perclos


def _ear_one(pts: np.ndarray) -> float:
    return (norm(pts[2] - pts[3]) + norm(pts[4] - pts[5])) / (
        2.0 * norm(pts[0] - pts[1]) + 1e-6
    )


def compute_ear(landmarks: np.ndarray) -> float:
    left = landmarks[LEFT_EYE, :2]
    right = landmarks[RIGHT_EYE, :2]
    return float((_ear_one(left) + _ear_one(right)) / 2.0)


def compute_gaze(landmarks: np.ndarray) -> tuple[str, float, float]:
    def iris_xy(center_idx: int) -> tuple[float, float]:
        # FaceLandmarker iris: center + 4 surrounding; use center landmark
        return float(landmarks[center_idx, 0]), float(landmarks[center_idx, 1])

    rix, riy = iris_xy(RIGHT_IRIS)
    lix, liy = iris_xy(LEFT_IRIS)

    right_horz = (rix - landmarks[33, 0]) / (landmarks[133, 0] - landmarks[33, 0] + 1e-6)
    left_horz = (lix - landmarks[362, 0]) / (landmarks[263, 0] - landmarks[362, 0] + 1e-6)
    avg_h = (right_horz + left_horz) / 2

    right_top = min(landmarks[159, 1], landmarks[145, 1])
    right_bot = max(landmarks[159, 1], landmarks[145, 1])
    left_top = min(landmarks[386, 1], landmarks[374, 1])
    left_bot = max(landmarks[386, 1], landmarks[374, 1])
    right_vert = (riy - right_top) / (right_bot - right_top + 1e-6)
    left_vert = (liy - left_top) / (left_bot - left_top + 1e-6)
    avg_v = (right_vert + left_vert) / 2

    if avg_h < 0.40:
        horz = "Right"
    elif avg_h > 0.60:
        horz = "Left"
    else:
        horz = "Center"

    if avg_v < 0.35:
        vert = "Up"
    elif avg_v > 0.65:
        vert = "Down"
    else:
        vert = "Center"

    if horz == "Center" and vert == "Center":
        direction = "Center"
    elif horz == "Center":
        direction = vert
    elif vert == "Center":
        direction = horz
    else:
        direction = f"{vert}-{horz}"
    return direction, float(avg_h), float(avg_v)


def head_pitch_proxy(landmarks: np.ndarray) -> float:
    nose = landmarks[1, :2]
    left_eye = landmarks[33, :2]
    right_eye = landmarks[263, :2]
    eye_mid = (left_eye + right_eye) / 2
    chin = landmarks[152, :2]
    face_h = max(abs(chin[1] - eye_mid[1]), 1e-6)
    return float((nose[1] - eye_mid[1]) / face_h)


def _build_landmarker() -> vision.FaceLandmarker:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}. Download face_landmarker.task into models/."
        )
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


class FQCProcessor:
    def __init__(self, source: str | int | None = None):
        self.source = source
        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.frame_version = 0
        self.metrics = FrameMetrics()
        self.clients = 0
        self.overlays = {
            "landmarks": True,
            "gaze_arrow": True,
            "hud": True,
        }
        self.gaze_history: deque[str] = deque(maxlen=8)
        self.scorer = AttentionScorer()
        self._alert_cooldown = {"Sleep": 0.0, "LookingAway": 0.0, "Tired": 0.0}
        self._alert_gap = 8.0
        self.landmarker: vision.FaceLandmarker | None = None
        self._fps_ema = 0.0
        self._last_fps_t = time.time()
        self._frame_count = 0
        self._video_ts_ms = 0

    def list_videos(self) -> list[str]:
        if not os.path.isdir(VIDEOS_DIR):
            return []
        return sorted(
            f
            for f in os.listdir(VIDEOS_DIR)
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm"))
        )

    def set_source(self, name: str | None):
        if name in (None, "", "webcam", "0"):
            self.source = 0
        else:
            path = name if os.path.isabs(name) else os.path.join(VIDEOS_DIR, name)
            self.source = path

    def set_overlays(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if k in self.overlays:
                    self.overlays[k] = bool(v)

    def register_stream_client(self):
        with self.lock:
            self.clients += 1

    def unregister_stream_client(self):
        with self.lock:
            self.clients = max(0, self.clients - 1)

    def get_jpeg(self, last_version: int, timeout: float = 1.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if self.jpeg is not None and self.frame_version != last_version:
                    return self.jpeg, self.frame_version
            time.sleep(0.02)
        with self.lock:
            return self.jpeg, self.frame_version

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            m = self.metrics
            return {
                "face_detected": m.face_detected,
                "ear": m.ear,
                "gaze_direction": m.gaze_direction,
                "looking_at_screen": m.looking_at_screen,
                "asleep": m.asleep,
                "tired": m.tired,
                "looking_away": m.looking_away,
                "perclos": round(m.perclos, 3),
                "pitch_proxy": m.pitch_proxy,
                "fps": round(m.fps, 1),
                "alerts": list(m.alerts),
                "source": str(self.source),
                "overlays": dict(self.overlays),
                "system": "Online" if self.running else "Offline",
            }

    def start(self):
        if self.running:
            return
        if self.source is None:
            videos = self.list_videos()
            self.source = os.path.join(VIDEOS_DIR, videos[0]) if videos else 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="fqc-engine", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3.0)
            self.thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

    def _open_capture(self):
        if self.cap:
            self.cap.release()
        src = self.source if self.source is not None else 0
        self.cap = cv2.VideoCapture(src)
        if isinstance(src, str):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self._video_ts_ms = 0

    def _smooth_gaze(self, direction: str) -> str:
        self.gaze_history.append(direction)
        return Counter(self.gaze_history).most_common(1)[0][0]

    def _maybe_alert(self, frame, event_type: str, message: str, severity: str):
        now = time.time()
        if now - self._alert_cooldown.get(event_type, 0) < self._alert_gap:
            return
        self._alert_cooldown[event_type] = now
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{event_type}_{stamp}.jpg"
        path = os.path.join(SNAPSHOTS_DIR, fname)
        cv2.imwrite(path, frame)
        rel = f"/snapshots/{fname}"
        db = SessionLocal()
        try:
            create_event(db, event_type, message, severity=severity, snapshot_path=rel)
        finally:
            db.close()

    def _draw(self, frame, landmarks_np: np.ndarray, metrics: FrameMetrics):
        h, w = frame.shape[:2]
        overlays = self.overlays

        if overlays.get("landmarks"):
            for i in LEFT_EYE + RIGHT_EYE:
                x, y = int(landmarks_np[i, 0] * w), int(landmarks_np[i, 1] * h)
                cv2.circle(frame, (x, y), 1, (0, 180, 255), -1)
            if landmarks_np.shape[0] > RIGHT_IRIS:
                for i in (LEFT_IRIS, RIGHT_IRIS):
                    x, y = int(landmarks_np[i, 0] * w), int(landmarks_np[i, 1] * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        if (
            overlays.get("gaze_arrow")
            and metrics.gaze_direction != "Unknown"
            and landmarks_np.shape[0] > RIGHT_IRIS
        ):
            for i in (LEFT_IRIS, RIGHT_IRIS):
                x, y = int(landmarks_np[i, 0] * w), int(landmarks_np[i, 1] * h)
                dx = dy = 0
                g = metrics.gaze_direction
                if "Left" in g:
                    dx = -36
                if "Right" in g:
                    dx = 36
                if "Up" in g:
                    dy = -36
                if "Down" in g:
                    dy = 36
                if dx or dy:
                    cv2.arrowedLine(
                        frame, (x, y), (x + dx, y + dy), (80, 255, 80), 2, tipLength=0.35
                    )

        if overlays.get("hud"):
            if metrics.asleep:
                banner, color = "SLEEP / NODDING OFF", (40, 40, 220)
            elif metrics.looking_away:
                banner, color = "NOT LOOKING AT SCREEN", (0, 140, 255)
            elif metrics.tired:
                banner, color = "TIRED (HIGH PERCLOS)", (0, 200, 255)
            elif metrics.looking_at_screen:
                banner, color = "ATTENTIVE — LOOKING AT SCREEN", (60, 180, 60)
            else:
                banner, color = "MONITORING", (180, 180, 180)

            cv2.rectangle(frame, (0, 0), (w, 42), color, -1)
            cv2.putText(
                frame, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA
            )

            ear_txt = f"{metrics.ear:.3f}" if metrics.ear is not None else "—"
            lines = [
                f"EAR: {ear_txt}   PERCLOS: {metrics.perclos*100:.1f}%",
                f"Gaze: {metrics.gaze_direction}   Screen: {'YES' if metrics.looking_at_screen else 'NO'}",
                f"FPS: {metrics.fps:.1f}",
            ]
            y0 = h - 70
            cv2.rectangle(frame, (0, y0 - 10), (w, h), (20, 20, 20), -1)
            for i, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (12, y0 + 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
        return frame

    def _loop(self):
        self.landmarker = _build_landmarker()
        self._open_capture()
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._open_capture()
                time.sleep(0.2)
                continue

            ok, frame = self.cap.read()
            if not ok:
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._video_ts_ms = 0
                    # Recreate landmarker after seek to reset timestamp monotonicity
                    if self.landmarker:
                        self.landmarker.close()
                    self.landmarker = _build_landmarker()
                    continue
                time.sleep(0.05)
                continue

            h0, w0 = frame.shape[:2]
            if w0 > 960:
                scale = 960 / w0
                frame = cv2.resize(frame, (960, int(h0 * scale)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._video_ts_ms += 33
            result = self.landmarker.detect_for_video(mp_image, self._video_ts_ms)

            metrics = FrameMetrics()
            t_now = time.time()
            self._frame_count += 1
            if t_now - self._last_fps_t >= 1.0:
                inst = self._frame_count / (t_now - self._last_fps_t)
                self._fps_ema = inst if self._fps_ema == 0 else 0.7 * self._fps_ema + 0.3 * inst
                self._frame_count = 0
                self._last_fps_t = t_now
            metrics.fps = self._fps_ema

            if result.face_landmarks:
                face = result.face_landmarks[0]
                lm_np = np.array([[p.x, p.y, p.z] for p in face], dtype=np.float64)
                metrics.face_detected = True
                metrics.ear = compute_ear(lm_np)

                if lm_np.shape[0] > RIGHT_IRIS:
                    direction, gh, gv = compute_gaze(lm_np)
                else:
                    direction, gh, gv = "Center", 0.5, 0.5
                direction = self._smooth_gaze(direction)
                metrics.gaze_direction = direction
                metrics.gaze_h = gh
                metrics.gaze_v = gv
                metrics.looking_at_screen = direction == "Center"
                metrics.pitch_proxy = head_pitch_proxy(lm_np)

                nodding = (
                    metrics.pitch_proxy is not None
                    and metrics.pitch_proxy > 0.35
                    and metrics.ear is not None
                    and metrics.ear < 0.28
                )

                asleep, tired, looking_away, perclos = self.scorer.update(
                    t_now, metrics.ear, metrics.looking_at_screen and not nodding
                )
                if nodding:
                    asleep = True
                metrics.asleep = asleep
                metrics.tired = tired
                metrics.looking_away = looking_away and not asleep
                metrics.perclos = perclos

                alerts = []
                if asleep:
                    alerts.append("Sleep")
                    self._maybe_alert(
                        frame,
                        "Sleep",
                        "Operator may be nodding off (eyes closed / head down)",
                        "Critical",
                    )
                elif looking_away:
                    alerts.append("LookingAway")
                    self._maybe_alert(
                        frame,
                        "LookingAway",
                        f"Operator not looking at screen (gaze={direction})",
                        "High",
                    )
                elif tired:
                    alerts.append("Tired")
                    self._maybe_alert(
                        frame,
                        "Tired",
                        f"Elevated PERCLOS ({perclos*100:.0f}%) — possible fatigue",
                        "Medium",
                    )
                metrics.alerts = alerts
                frame = self._draw(frame, lm_np, metrics)
            else:
                self.scorer.update(t_now, None, False)
                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

            ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok_enc:
                with self.lock:
                    self.jpeg = buf.tobytes()
                    self.frame_version += 1
                    self.metrics = metrics

            if isinstance(self.source, str):
                time.sleep(0.03)

        if self.cap:
            self.cap.release()
            self.cap = None
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None


processor = FQCProcessor()
