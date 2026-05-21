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

from .board import score_from_norm, score_to_label


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
        # If an ellipse was fitted (camera not perpendicular), stores
        # OpenCV ellipse format: ((cx, cy), (MA, ma), angle_deg)
        # MA and ma are the FULL axis lengths (semi = /2).
        self.board_ellipse: Optional[tuple] = None

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

            if self.board_ellipse is not None:
                (ecx, ecy), (ma, mi), angle = self.board_ellipse
                center = (int(ecx), int(ecy))
                sa, sb = int(ma / 2), int(mi / 2)   # semi-axes
                # Ellipse verte — avant les chiffres (≈ 87 % des axes)
                cv2.ellipse(frame, center, (int(sa * 0.87), int(sb * 0.87)),
                            angle, 0, 360, (0, 255, 0), 2)
                # Ellipse rouge — bord extérieur (après les chiffres)
                cv2.ellipse(frame, center, (sa, sb), angle, 0, 360, (0, 0, 255), 2)
            else:
                r = self.board_radius
                # Cercle vert — avant les chiffres
                cv2.circle(frame, (cx, cy), int(r * 0.87), (0, 255, 0), 2)
                # Cercle rouge — bord extérieur
                cv2.circle(frame, (cx, cy), r, (0, 0, 255), 2)

            # Point central bleu (commun aux deux modes)
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
        Auto-detect the dartboard using Hough circle transforms then an
        ellipse-fitting fallback.  Candidates are ranked by their internal
        edge density: the dartboard has far more internal structure (segment
        wires, rings) than any other region, so the highest-density candidate
        is almost always the board regardless of its position in the frame.
        """
        with self._lock:
            if self._current_frame is None:
                return {"error": "Aucune image disponible."}
            frame = self._current_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # CLAHE : améliore le contraste local (utile en conditions d'éclairage inégal)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred  = cv2.GaussianBlur(enhanced, (9, 9), 2)

        # Carte de contours utilisée pour scorer les candidats
        edges = cv2.Canny(blurred, 40, 120)

        min_r = min(h, w) // 8
        max_r = min(h, w) // 2

        def edge_density(cx_, cy_, r_):
            """Fraction de pixels de contour à l'intérieur du cercle candidat."""
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (int(cx_), int(cy_)), int(r_), 255, -1)
            inside = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
            area   = math.pi * r_ * r_
            return inside / area if area > 0 else 0.0

        # ── 1. Hough circles — plusieurs seuils, on garde le + dense ─────────
        for param2 in (50, 38, 28):
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=min(h, w) // 3,   # autoriser plusieurs candidats
                param1=80,
                param2=param2,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                best = max(
                    circles[0],
                    key=lambda c: edge_density(c[0], c[1], c[2]),
                )
                self.board_center = (int(best[0]), int(best[1]))
                self.board_radius = int(best[2])
                return {
                    "status": "ok",
                    "center": list(self.board_center),
                    "radius": self.board_radius,
                }

        # ── 2. Fallback ellipse (caméra de biais) ─────────────────────────────
        dil_k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dil = cv2.dilate(edges, dil_k, iterations=2)
        contours, _ = cv2.findContours(
            edges_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        best_ellipse = None
        best_density = -1.0
        min_area     = math.pi * min_r * min_r * 0.4

        for cnt in contours:
            if len(cnt) < 50 or cv2.contourArea(cnt) < min_area:
                continue
            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                continue

            (ex, ey), (axis_a, axis_b), _ = ellipse
            if axis_b < 1 or axis_a / axis_b > 2.2:
                continue
            r_avg = (axis_a + axis_b) / 4.0
            if not (min_r <= r_avg <= max_r):
                continue

            # Densité de contours dans l'ellipse
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, ellipse, 255, -1)
            n_inside = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
            n_area   = cv2.countNonZero(mask)
            density  = n_inside / n_area if n_area > 0 else 0.0

            if density > best_density:
                best_density = density
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
        """Set board region from user-provided coordinates (always circle mode)."""
        self.board_center  = (int(cx), int(cy))
        self.board_radius  = int(radius)
        self.board_ellipse = None

    def _to_board_norm(self, px: float, py: float) -> Tuple[float, float]:
        """
        Transform a pixel position to normalised board coordinates
        (radius 1.0 = outer double ring edge).

        If an ellipse was fitted, applies the inverse ellipse transform to
        correct perspective distortion before scoring.
        """
        if self.board_ellipse is not None:
            (cx, cy), (ma, mi), angle_deg = self.board_ellipse
            a = ma / 2.0          # semi-major axis
            b = mi / 2.0          # semi-minor axis
            theta  = math.radians(angle_deg)
            dx = px - cx
            dy = py - cy
            cos_t =  math.cos(-theta)
            sin_t =  math.sin(-theta)
            dx_r  =  dx * cos_t - dy * sin_t
            dy_r  =  dx * sin_t + dy * cos_t
            return dx_r / a, dy_r / b
        else:
            cx, cy = self.board_center
            r = float(self.board_radius)
            return (px - cx) / r, (py - cy) / r

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
            _dist, tip = min(distances, key=lambda x: x[0])

            # Ignore tips outside 110 % of board boundary
            tip_x, tip_y = float(tip[0]), float(tip[1])
            x_n, y_n = self._to_board_norm(tip_x, tip_y)
            if math.sqrt(x_n * x_n + y_n * y_n) > 1.1:
                continue

            dart_positions.append((tip_x, tip_y))

        self._dart_positions = dart_positions

        # --- Convert positions to scores (with ellipse correction) ---
        results = []
        for (tip_x, tip_y) in dart_positions:
            x_n, y_n = self._to_board_norm(tip_x, tip_y)
            score, zone, mult = score_from_norm(x_n, y_n)
            label = score_to_label(score, zone, mult)
            results.append(
                {"score": score, "label": label, "x_norm": x_n, "y_norm": y_n}
            )

        return results
