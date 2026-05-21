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
        Both approaches score candidates with a **ring-structure metric**:
        a real dartboard has strong edges at specific fractional radii
        (bull ≈9 %, triple ≈57-63 %, double ≈95-100 %).
        The geometric mean of those ring densities is very high for a
        dartboard and near-zero for random textured backgrounds.

        1. Contour → convex-hull → fitEllipse (perspective-aware, primary).
        2. Hough circles scored by ring-structure (fallback if 1 fails).
        3. After locating the board, fit a refined ellipse on the rim edges.
        """
        with self._lock:
            if self._current_frame is None:
                return {"error": "Aucune image disponible."}
            frame = self._current_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # The board is always roughly centred in the frame.
        # Any candidate whose centre is farther than this from the image
        # centre is rejected as a false positive.
        img_cx, img_cy = w // 2, h // 2
        max_center_dist = min(h, w) * 0.40

        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred  = cv2.GaussianBlur(enhanced, (9, 9), 2)
        edges    = cv2.Canny(blurred, 40, 120)

        min_r = min(h, w) // 6
        max_r = int(min(h, w) * 0.72)

        # ── Ring-structure scorer ──────────────────────────────────────────────
        # Measures edge density in thin annuli at the known ring positions of a
        # standard dartboard (fractions of outer double-ring radius).
        # Geometric mean → a single missing ring drives the score to 0.
        _FRACS = (0.09, 0.57, 0.63, 0.95, 1.00)
        _HW    = 0.05   # annulus half-width as fraction of r

        def ring_score(cx_, cy_, r_):
            log_sum = 0.0
            for f in _FRACS:
                rr  = max(3, int(r_ * f))
                dr  = max(2, int(r_ * _HW))
                ann = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(ann, (int(cx_), int(cy_)), rr + dr, 255, -1)
                cv2.circle(ann, (int(cx_), int(cy_)), max(0, rr - dr), 0, -1)
                n    = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=ann))
                area = cv2.countNonZero(ann)
                log_sum += math.log(n / area if area > 0 else 1e-9)
            return log_sum / len(_FRACS)   # log geometric mean (monotone ↔ OK to compare)

        # ── Helper: fit ellipse on the board rim edge pixels ──────────────────
        def fit_rim_ellipse(cx_, cy_, r_):
            ann = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(ann, (int(cx_), int(cy_)), int(r_ * 1.15), 255, -1)
            cv2.circle(ann, (int(cx_), int(cy_)), int(r_ * 0.85), 0, -1)
            rim = cv2.bitwise_and(edges, edges, mask=ann)
            pts_list, _ = cv2.findContours(rim, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            valid = [c for c in pts_list if len(c) >= 5]
            if not valid:
                return None
            all_pts = np.vstack(valid)
            if len(all_pts) < 50:
                return None
            try:
                ell = cv2.fitEllipse(all_pts)
            except cv2.error:
                return None
            (_, _), (ma, mi), _ = ell
            if mi < 1 or ma / mi > 2.5:
                return None
            return ell

        # ── Step 1 : contour → convex hull → fitEllipse (no circle assumption) ─
        k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        best_ell      = None
        best_rs       = float('-inf')
        min_cnt_area  = math.pi * min_r * min_r * 0.4

        for cnt in contours:
            if len(cnt) < 50 or cv2.contourArea(cnt) < min_cnt_area:
                continue
            hull = cv2.convexHull(cnt)
            if len(hull) < 5:
                continue
            # Reject if the raw contour fills < 30 % of its convex hull
            # (avoids fitting ellipses to thin curvy lines)
            if cv2.contourArea(cnt) / (cv2.contourArea(hull) + 1e-6) < 0.30:
                continue
            try:
                ell = cv2.fitEllipse(hull)
            except cv2.error:
                continue
            (ex, ey), (ma, mi), _ = ell
            if mi < 1 or ma / mi > 2.5:
                continue
            r_avg = (ma + mi) / 4.0
            if not (min_r <= r_avg <= max_r):
                continue
            # Board centre must be near the image centre
            if math.sqrt((ex - img_cx) ** 2 + (ey - img_cy) ** 2) > max_center_dist:
                continue
            s = ring_score(ex, ey, r_avg)
            if s > best_rs:
                best_rs  = s
                best_ell = ell

        # ── Step 2 : Hough fallback (if no good contour candidate) ────────────
        hough_cx = hough_cy = hough_r = None
        if best_ell is None:
            hough_best_rs = float('-inf')
            for param2 in (50, 38, 28):
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT,
                    dp=1, minDist=min(h, w) // 3,
                    param1=80, param2=param2,
                    minRadius=min_r, maxRadius=max_r,
                )
                if circles is not None:
                    for c in circles[0]:
                        ccx, ccy, cr = int(c[0]), int(c[1]), int(c[2])
                        # Reject candidates too far from image centre
                        if math.sqrt((ccx - img_cx) ** 2 + (ccy - img_cy) ** 2) > max_center_dist:
                            continue
                        s = ring_score(ccx, ccy, cr)
                        if s > hough_best_rs:
                            hough_best_rs = s
                            hough_cx, hough_cy, hough_r = ccx, ccy, cr
                    break   # stop at first param2 that detects anything

            if hough_cx is not None:
                # Try to upgrade the Hough circle to an accurate ellipse
                ell = fit_rim_ellipse(hough_cx, hough_cy, hough_r)
                if ell is not None:
                    best_ell = ell
                else:
                    # No good ellipse fit — store as plain circle
                    self.board_ellipse = None
                    self.board_center  = (hough_cx, hough_cy)
                    self.board_radius  = hough_r
                    return {
                        "status": "ok",
                        "center": [hough_cx, hough_cy],
                        "radius": hough_r,
                    }

        if best_ell is None:
            return {
                "error": (
                    "Aucune cible détectée. Améliorez l'éclairage "
                    "ou utilisez la calibration manuelle."
                )
            }

        # ── Refine: fit the ellipse on the rim edges for better precision ──────
        (ex0, ey0), (ma0, mi0), _ = best_ell
        r0 = (ma0 + mi0) / 4.0
        refined = fit_rim_ellipse(ex0, ey0, r0)
        if refined is not None:
            best_ell = refined

        (ex, ey), (ma, mi), _ = best_ell
        aspect = ma / mi if mi > 0 else 1.0
        self.board_center  = (int(ex), int(ey))
        self.board_radius  = int((ma + mi) / 4.0)
        self.board_ellipse = best_ell if aspect >= 1.05 else None
        return {
            "status": "ok",
            "center": list(self.board_center),
            "radius": self.board_radius,
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
