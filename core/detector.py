"""
Camera capture and dart detection via OpenCV.

Detection strategy (top-down camera):
  1. Maintain a background reference of the empty board.
  2. On each frame, compute absolute diff against the background.
  3. Threshold + morphology to isolate new objects (dart barrels/shafts).
  4. For each significant contour, find the point closest to the board centre
     → that is the estimated dart-tip position.
  5. Map pixel position → board score via core.board.pixel_to_score.

Calibration:
  - Automatic: Hough circle transform on the board image.
  - Manual:    User supplies centre (x, y) + radius in pixels (or via a
               browser click that POSTs to /api/camera/calibrate).
"""
import math
import threading
import time
from typing import List, Optional, Tuple

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from .board import pixel_to_score, score_to_label


class DartDetector:
    """Thread-safe camera capture + dart detection."""

    # Minimum contour area (px²) considered a dart
    MIN_CONTOUR_AREA = 40
    # Background diff threshold (0–255)
    DIFF_THRESHOLD = 28

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        if not _CV2_AVAILABLE:
            raise RuntimeError(
                "opencv-python n'est pas installé. "
                "Lancez: pip install opencv-python"
            )

        self.camera_id = camera_id
        self.width = width
        self.height = height

        self._cap: Optional[object] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Frame buffers
        self._current_frame: Optional[object] = None  # np.ndarray
        self._background: Optional[object] = None

        # Board calibration (pixel space)
        self.board_center: Optional[Tuple[int, int]] = None
        self.board_radius: Optional[int] = None

        # Last detected dart positions (for overlay)
        self._dart_positions: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Impossible d'ouvrir la caméra {self.camera_id}. "
                "Vérifiez que la caméra est branchée et non utilisée."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()

    def is_available(self) -> bool:
        return (
            self._cap is not None
            and self._cap.isOpened()
            and self._current_frame is not None
        )

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._current_frame = frame
            time.sleep(0.033)  # ~30 fps cap

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_current_frame(self, annotated: bool = True):
        """Return the latest frame (optionally with calibration overlay)."""
        with self._lock:
            if self._current_frame is None:
                return None
            frame = self._current_frame.copy()

        if annotated and self.board_center and self.board_radius:
            cx, cy = self.board_center
            r = self.board_radius
            # Cercle vert — juste avant les chiffres (bord anneau double ≈ 87 % du rayon total)
            cv2.circle(frame, (cx, cy), int(r * 0.87), (0, 255, 0), 2)
            # Cercle rouge — bord extérieur de la cible, après les chiffres
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), 2)
            # Point central bleu
            cv2.circle(frame, (cx, cy), 7, (255, 80, 0), -1)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)
            # Positions des fléchettes détectées
            for (dx, dy) in self._dart_positions:
                cv2.circle(frame, (int(dx), int(dy)), 9, (255, 60, 0), -1)
                cv2.circle(frame, (int(dx), int(dy)), 9, (255, 255, 255), 2)

        return frame

    def get_jpeg_frame(self, quality: int = 70) -> Optional[bytes]:
        """Return the latest frame encoded as JPEG bytes."""
        frame = self.get_current_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def set_background(self) -> bool:
        """Capture current frame as background (board without darts)."""
        with self._lock:
            if self._current_frame is None:
                return False
            self._background = self._current_frame.copy()
        return True

    def calibrate_auto(self) -> dict:
        """
        Auto-detect the dartboard using Hough circle transforms with
        progressively lenient thresholds, then an ellipse-fitting fallback
        for cameras that are not perfectly perpendicular to the board.
        Returns a dict with 'status', 'center', 'radius' or 'error'.
        """
        with self._lock:
            if self._current_frame is None:
                return {"error": "Aucune image disponible."}
            frame = self._current_frame.copy()

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w    = gray.shape
        min_r   = min(h, w) // 6
        max_r   = min(h, w) // 2

        # ── 1. Hough circles — essai avec plusieurs seuils (du plus strict au plus laxiste)
        for param2 in (50, 38, 28):
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=min(h, w) // 2,
                param1=100,
                param2=param2,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                best = max(circles[0], key=lambda c: c[2])
                self.board_center = (int(best[0]), int(best[1]))
                self.board_radius = int(best[2])
                return {
                    "status": "ok",
                    "center": list(self.board_center),
                    "radius": self.board_radius,
                }

        # ── 2. Fallback : détection d'ellipse pour caméra non perpendiculaire ────
        # La cible peut apparaître elliptique si la caméra est de biais.
        # On cherche le plus grand contour fermé approchant une ellipse.
        edges = cv2.Canny(blurred, 40, 120)
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, dil_k, iterations=2)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        best_ellipse = None
        best_r       = 0
        min_area     = math.pi * min_r * min_r * 0.4

        for cnt in contours:
            if len(cnt) < 50:
                continue
            if cv2.contourArea(cnt) < min_area:
                continue
            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                continue

            (ex, ey), (axis_a, axis_b), _angle = ellipse
            if axis_b < 1:
                continue
            aspect = axis_a / axis_b          # >= 1 (major/minor)
            if aspect > 2.2:                  # trop déformé, pas une cible
                continue

            r_avg = (axis_a + axis_b) / 4.0   # demi-axe moyen
            if not (min_r <= r_avg <= max_r):
                continue

            # Préférer l'ellipse la plus grande qui reste dans le cadre
            if r_avg > best_r:
                best_r       = r_avg
                best_ellipse = ellipse

        if best_ellipse is not None:
            (ex, ey), (axis_a, axis_b), _ = best_ellipse
            self.board_center = (int(ex), int(ey))
            self.board_radius = int((axis_a + axis_b) / 4.0)
            return {
                "status": "ok",
                "center": list(self.board_center),
                "radius": self.board_radius,
            }

        return {
            "error": (
                "Aucun cercle détecté. Essayez d'améliorer l'éclairage "
                "ou utilisez la calibration manuelle."
            )
        }

    def calibrate_manual(self, cx: float, cy: float, radius: float) -> None:
        """Set board region from user-provided coordinates."""
        self.board_center = (int(cx), int(cy))
        self.board_radius = int(radius)

    # ------------------------------------------------------------------
    # Dart detection
    # ------------------------------------------------------------------

    def detect_darts(self) -> List[dict]:
        """
        Compare current frame with background and return detected dart hits.

        Returns a list of dicts:
            { 'score': int, 'label': str, 'x_norm': float, 'y_norm': float }
        x_norm / y_norm are normalised board coordinates (–1..1).
        """
        if (
            self._background is None
            or self.board_center is None
            or self.board_radius is None
        ):
            return []

        with self._lock:
            if self._current_frame is None:
                return []
            current = self._current_frame.copy()

        # --- Difference image ---
        diff = cv2.absdiff(current, self._background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # --- Find contours ---
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cx, cy = self.board_center
        r = self.board_radius
        dart_positions = []

        for cnt in contours:
            if cv2.contourArea(cnt) < self.MIN_CONTOUR_AREA:
                continue

            # Dart tip = contour point closest to board centre
            distances = [
                (float(np.hypot(pt[0][0] - cx, pt[0][1] - cy)), pt[0])
                for pt in cnt
            ]
            dist, tip = min(distances, key=lambda x: x[0])

            # Ignore tips outside 110 % of board radius (noise)
            if dist > r * 1.1:
                continue

            dart_positions.append((float(tip[0]), float(tip[1])))

        self._dart_positions = dart_positions

        # --- Convert positions to scores ---
        results = []
        for (tip_x, tip_y) in dart_positions:
            score, zone, mult = pixel_to_score(tip_x, tip_y, cx, cy, r)
            label = score_to_label(score, zone, mult)
            x_norm = (tip_x - cx) / r
            y_norm = (tip_y - cy) / r
            results.append(
                {"score": score, "label": label, "x_norm": x_norm, "y_norm": y_norm}
            )

        return results
