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
        Detect the dartboard and store its ellipse for perspective-aware scoring.

        Strategy
        --------
        1. Hough circles across a broad radius range (15 %–82 % of frame) so
           that boards which fill most of the frame are found correctly.
        2. All candidates from all Hough parameter sweeps are scored with the
           ring-structure metric and the globally best one is kept.
        3. A tight-annulus (90 %–110 %) ellipse fit refines the circle to an
           ellipse that captures camera angle, validated against centre drift.
        """
        with self._lock:
            if self._current_frame is None:
                return {"error": "Aucune image disponible."}
            frame = self._current_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        img_cx, img_cy = w // 2, h // 2
        max_center_dist = min(h, w) * 0.50   # board centre may be up to half-width off

        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred  = cv2.GaussianBlur(enhanced, (9, 9), 2)
        edges    = cv2.Canny(blurred, 30, 100)

        # Allow boards from 15 % up to 82 % of the shorter frame dimension.
        # 82 % at 720 p ≈ 590 px  →  covers boards that nearly fill the frame.
        min_r = int(min(h, w) * 0.15)
        max_r = int(min(h, w) * 0.82)

        # ── Ring-structure scorer ──────────────────────────────────────────────
        _FRACS = (0.09, 0.57, 0.63, 0.95, 1.00)
        _HW    = 0.05

        def ring_score(cx_, cy_, r_):
            log_sum = 0.0
            n_valid = 0
            for f in _FRACS:
                rr   = max(3, int(r_ * f))
                dr   = max(2, int(r_ * _HW))
                ann  = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(ann, (int(cx_), int(cy_)), rr + dr, 255, -1)
                cv2.circle(ann, (int(cx_), int(cy_)), max(0, rr - dr), 0, -1)
                area = cv2.countNonZero(ann)
                if area == 0:
                    continue
                n    = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=ann))
                log_sum += math.log(max(n, 1) / area)
                n_valid += 1
            return log_sum / n_valid if n_valid else float('-inf')

        # ── Tight-annulus ellipse refinement ──────────────────────────────────
        def fit_rim_ellipse(cx_, cy_, r_):
            inner = max(1, int(r_ * 0.90))
            outer = int(r_ * 1.10)
            ann   = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(ann, (int(cx_), int(cy_)), outer, 255, -1)
            cv2.circle(ann, (int(cx_), int(cy_)), inner, 0, -1)
            rim  = cv2.bitwise_and(edges, edges, mask=ann)
            cnts, _ = cv2.findContours(rim, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            valid = [c for c in cnts if len(c) >= 5]
            if not valid:
                return None
            all_pts = np.vstack(valid)
            if len(all_pts) < 60:
                return None
            try:
                ell = cv2.fitEllipse(all_pts)
            except cv2.error:
                return None
            (_, _), (ma, mi), _ = ell
            if mi < 1 or ma / mi > 2.0:
                return None
            return ell

        # ── Step 1: Hough circles – collect & score ALL candidates ────────────
        # Use dp=1 (full-resolution accumulator) and sweep param2 from strict
        # to lenient; score every unique candidate and keep the global best.
        candidates: list = []   # (ring_score, cx, cy, r)
        seen: list = []          # (cx, cy, r) already evaluated

        def already_seen(cx_, cy_, cr_):
            for scx, scy, scr in seen:
                if (abs(cx_ - scx) < 20 and abs(cy_ - scy) < 20
                        and abs(cr_ - scr) < 30):
                    return True
            return False

        for param2 in (60, 45, 35, 25, 18):
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=max(min_r, min(h, w) // 3),
                param1=80, param2=param2,
                minRadius=min_r, maxRadius=max_r,
            )
            if circles is None:
                continue
            for c in circles[0]:
                ccx, ccy, cr = float(c[0]), float(c[1]), float(c[2])
                if math.sqrt((ccx - img_cx) ** 2 + (ccy - img_cy) ** 2) > max_center_dist:
                    continue
                if already_seen(ccx, ccy, cr):
                    continue
                seen.append((ccx, ccy, cr))
                s = ring_score(ccx, ccy, cr)
                candidates.append((s, ccx, ccy, cr))

        if not candidates:
            return {
                "error": (
                    "Aucune cible détectée. "
                    "Améliorez l'éclairage ou utilisez la calibration manuelle."
                )
            }

        # Best ring-structure score wins.
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_rs, best_cx, best_cy, best_r = candidates[0]

        # ── Step 2: Ellipse refinement ────────────────────────────────────────
        ell = fit_rim_ellipse(best_cx, best_cy, best_r)
        if ell is not None:
            (ex, ey), (ma, mi), _ = ell
            drift = math.sqrt((ex - best_cx) ** 2 + (ey - best_cy) ** 2)
            if drift < best_r * 0.30:
                self.board_ellipse = ell
                self.board_center  = (int(ex), int(ey))
                self.board_radius  = int((ma + mi) / 4.0)
                return {
                    "status": "ok",
                    "center": list(self.board_center),
                    "radius": self.board_radius,
                }

        # ── Step 3: Plain circle fallback ─────────────────────────────────────
        self.board_ellipse = None
        self.board_center  = (int(best_cx), int(best_cy))
        self.board_radius  = int(best_r)
        return {
            "status": "ok",
            "center": list(self.board_center),
            "radius": self.board_radius,
        }

    def calibrate_manual(self, cx: float, cy: float, radius: float) -> bool:
        """
        Set board region from user-provided centre + radius.
        Automatically upgrades to an ellipse by fitting to the actual rim
        edges at the specified location (corrects camera tilt/perspective).
        Returns True if an ellipse was fitted, False if a plain circle is kept.
        """
        self.board_center  = (int(cx), int(cy))
        self.board_radius  = int(radius)
        self.board_ellipse = None

        ell = self._fit_ellipse_at(cx, cy, radius)
        if ell is not None:
            self.board_ellipse = ell
            (ex, ey), (ma, mi), _ = ell
            self.board_center = (int(ex), int(ey))
            self.board_radius = int((ma + mi) / 4.0)
            return True
        return False

    def _fit_ellipse_at(self, cx: float, cy: float, r: float):
        """
        Fit an ellipse to the board rim by collecting Canny edge pixels in a
        wide annulus (80 %–120 % of r) centred at (cx, cy).
        Returns an OpenCV ellipse tuple, or None if the fit is poor.
        """
        with self._lock:
            if self._current_frame is None:
                return None
            frame = self._current_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred  = cv2.GaussianBlur(enhanced, (9, 9), 2)
        edges    = cv2.Canny(blurred, 30, 100)

        inner = max(1, int(r * 0.82))
        outer = min(int(r * 1.18), min(h, w))
        ann   = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ann, (int(cx), int(cy)), outer, 255, -1)
        cv2.circle(ann, (int(cx), int(cy)), inner, 0, -1)

        rim     = cv2.bitwise_and(edges, edges, mask=ann)
        cnts, _ = cv2.findContours(rim, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        valid   = [c for c in cnts if len(c) >= 5]
        if not valid:
            return None
        all_pts = np.vstack(valid)
        if len(all_pts) < 50:
            return None
        try:
            ell = cv2.fitEllipse(all_pts)
        except cv2.error:
            return None

        (ex, ey), (ma, mi), _ = ell
        if mi < 1 or ma / mi > 2.2:
            return None
        # Reject if the fitted centre drifted too far from the user's click
        if math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2) > r * 0.25:
            return None
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
