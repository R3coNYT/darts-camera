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
            r      = self.board_radius

            # Draw a ring at `frac` of the board radius (circle or ellipse).
            def draw_ring(frac: float, color: tuple, thickness: int = 1) -> None:
                if self.board_ellipse is not None:
                    (ecx, ecy), (ma, mi), ang = self.board_ellipse
                    axes = (
                        max(1, int(ma / 2 * frac)),
                        max(1, int(mi / 2 * frac)),
                    )
                    cv2.ellipse(frame, (int(ecx), int(ecy)), axes,
                                ang, 0, 360, color, thickness)
                else:
                    cv2.circle(frame, (cx, cy), max(1, int(r * frac)),
                               color, thickness)

            # Double ring  (rouge) — bords intérieur et extérieur
            draw_ring(1.000, (0, 0, 220), 2)   # outer double
            draw_ring(0.953, (0, 0, 220), 1)   # inner double
            # Triple ring  (vert)
            draw_ring(0.629, (0, 220, 0), 2)   # outer triple
            draw_ring(0.582, (0, 220, 0), 1)   # inner triple
            # Outer bull   (jaune)
            draw_ring(0.094, (0, 200, 255), 2)
            # Bull (centre bleu)
            cv2.circle(frame, (cx, cy), max(1, int(r * 0.037)), (255, 160, 0), -1)
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
        Calibration auto basée sur les anneaux rouge/vert de la cible.
        On détecte le bord extérieur du double, pas la bordure noire avec les numéros.
        """
        with self._lock:
            if self._current_frame is None:
                return {"error": "Aucune image disponible."}
            frame = self._current_frame.copy()

        ell = self._fit_scoring_ellipse_from_colors(frame)

        if ell is None:
            return {
                "error": (
                    "Impossible de détecter correctement les anneaux rouge/vert. "
                    "Essaie avec plus de lumière ou utilise la calibration manuelle."
                )
            }

        (ex, ey), (axis_a, axis_b), angle = ell

        self.board_ellipse = ell
        self.board_center = (int(ex), int(ey))

        # Rayon moyen uniquement pour l'affichage / fallback.
        # Pour le scoring, _to_board_norm utilise directement board_ellipse.
        self.board_radius = int((axis_a + axis_b) / 4.0)

        return {
            "status": "ok",
            "center": list(self.board_center),
            "radius": self.board_radius,
            "ellipse": {
                "axes": [int(axis_a), int(axis_b)],
                "angle": float(angle),
            },
        }

    def _fit_scoring_ellipse_from_colors(self, frame):
        """
        Détecte les zones rouges/vertes de la cible, récupère les points les plus
        externes par angle, puis fit une ellipse correspondant au bord extérieur
        du double.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # Masque rouge : deux plages car le rouge est coupé autour de 0/180 en HSV
        red_1 = cv2.inRange(hsv, (0, 35, 25), (12, 255, 230))
        red_2 = cv2.inRange(hsv, (165, 35, 25), (180, 255, 230))

        # Masque vert de la cible
        green = cv2.inRange(hsv, (35, 30, 20), (95, 255, 230))

        mask = cv2.bitwise_or(red_1, red_2)
        mask = cv2.bitwise_or(mask, green)

        # Nettoyage du masque
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pts_list = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 20:
                continue
            pts_list.append(c.reshape(-1, 2))

        if not pts_list:
            return None

        pts = np.vstack(pts_list).astype(np.float32)

        # Centre approximatif avec les pixels rouge/vert
        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))

        dx = pts[:, 0] - cx0
        dy = pts[:, 1] - cy0

        dist2 = dx * dx + dy * dy
        angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

        # Pour chaque angle, on garde les points les plus éloignés :
        # ils correspondent normalement au double extérieur.
        outer_pts = []
        bins = 180  # 2 degrés par bin

        for b in range(bins):
            a_min = b * (360.0 / bins)
            a_max = (b + 1) * (360.0 / bins)

            idx = np.where((angles >= a_min) & (angles < a_max))[0]
            if len(idx) == 0:
                continue

            # On garde les 2 points les plus externes pour stabiliser l'ellipse
            best = idx[np.argsort(dist2[idx])[-2:]]
            outer_pts.extend(pts[best])

        outer_pts = np.array(outer_pts, dtype=np.float32)

        if len(outer_pts) < 40:
            return None

        try:
            ell = cv2.fitEllipse(outer_pts.reshape(-1, 1, 2))
        except cv2.error:
            return None

        (ex, ey), (axis_a, axis_b), angle = ell

        small_axis = min(axis_a, axis_b)
        big_axis = max(axis_a, axis_b)

        # Validation basique
        if small_axis < min(h, w) * 0.20:
            return None

        if big_axis > min(h, w) * 1.05:
            return None

        if big_axis / small_axis > 2.2:
            return None

        if not (0 <= ex < w and 0 <= ey < h):
            return None

        # Légère marge pour être bien sur le bord extérieur du double
        # Si ton cercle est un peu trop grand/petit, ajuste 1.00 à 1.03.
        SCALE = 1.015
        ell = ((ex, ey), (axis_a * SCALE, axis_b * SCALE), angle)

        return ell

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
