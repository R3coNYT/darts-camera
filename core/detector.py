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
import os
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
        
        self.board_homography = None
        self.board_homography_inv = None

        # Last detected dart positions (for overlay)
        self._dart_positions: List[Tuple[float, float]] = []
        # Calibration debug: outer edge points used to fit the ellipse
        self._cal_debug_pts: Optional[object] = None
        # YOLO pose model (chargé en lazy la première fois qu'on en a besoin)
        self._yolo_model = None

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
                if self.board_homography is not None:
                    self._draw_projected_ring(frame, frac, color, thickness)

                elif self.board_ellipse is not None:
                    (ecx, ecy), (ma, mi), ang = self.board_ellipse
                    axes = (
                        max(1, int(ma / 2 * frac)),
                        max(1, int(mi / 2 * frac)),
                    )
                    cv2.ellipse(
                        frame,
                        (int(ecx), int(ecy)),
                        axes,
                        ang,
                        0,
                        360,
                        color,
                        thickness,
                    )

                else:
                    cv2.circle(
                        frame,
                        (cx, cy),
                        max(1, int(r * frac)),
                        color,
                        thickness,
                    )

            # Double ring  (rouge) — bords intérieur et extérieur
            draw_ring(1.000, (0, 0, 220), 2)   # outer double
            draw_ring(0.953, (0, 0, 220), 1)   # inner double
            # Triple ring  (vert)
            draw_ring(0.629, (0, 220, 0), 2)   # outer triple
            draw_ring(0.582, (0, 220, 0), 1)   # inner triple
            # Outer bull   (jaune)
            draw_ring(0.094, (0, 200, 255), 2)
            # Bull (centre bleu)
            if self.board_homography is not None:
                self._draw_projected_disk(frame, 0.037, (255, 160, 0))
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)
            else:
                cv2.circle(frame, (cx, cy), max(1, int(r * 0.037)), (255, 160, 0), -1)
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)
            # Positions des fléchettes détectées
            for (dx, dy) in self._dart_positions:
                cv2.circle(frame, (int(dx), int(dy)), 9, (255, 60, 0), -1)
                cv2.circle(frame, (int(dx), int(dy)), 9, (255, 255, 255), 2)

            # Points de calibration debug (intersections barres détectées)
            if self._cal_debug_pts is not None:
                for pt in self._cal_debug_pts:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

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
        Calibration auto basée d'abord sur les barres métalliques de la cible.
        Les barres radiales servent à trouver le vrai centre, puis leurs extrémités
        servent à retrouver le bord extérieur de la zone de score.
        """
        with self._lock:
            if self._current_frame is None:
                return {"error": "Aucune image disponible."}
            frame = self._current_frame.copy()

        # Méthode 0 : YOLO11n-pose (si le modèle entraîné est présent)
        if self._fit_board_from_yolo(frame):
            return {
                "status": "ok",
                "method": "yolo_pose",
                "center": list(self.board_center),
                "radius": self.board_radius,
            }

        # Méthode 1 : intersections des barres métalliques (HoughLinesP)
        ell = self._fit_board_from_intersections(frame)

        if ell is not None:
            (ex, ey), (axis_a, axis_b), angle = ell

            self.board_ellipse = ell
            self.board_center = (int(ex), int(ey))
            self.board_radius = int((axis_a + axis_b) / 4.0)

            return {
                "status": "ok",
                "method": "intersections",
                "center": list(self.board_center),
                "radius": self.board_radius,
                "ellipse": {
                    "axes": [int(axis_a), int(axis_b)],
                    "angle": float(angle),
                },
            }

        # Méthode 2 : fallback rouge/vert
        ell = self._fit_scoring_ellipse_from_colors(frame)

        if ell is not None:
            (ex, ey), (axis_a, axis_b), angle = ell

            self.board_ellipse = ell
            self.board_center = (int(ex), int(ey))
            self.board_radius = int((axis_a + axis_b) / 4.0)

            return {
                "status": "ok",
                "method": "color_ellipse",
                "center": list(self.board_center),
                "radius": self.board_radius,
                "ellipse": {
                    "axes": [int(axis_a), int(axis_b)],
                    "angle": float(angle),
                },
            }

        # Méthode 3 : fallback cercle extérieur
        fallback = self._fit_outer_board_circle_hough(frame)

        if fallback is None:
            return {
                "error": (
                    "Impossible de détecter la cible. "
                    "Essaie sans fléchettes, avec plus de lumière, ou utilise la calibration manuelle."
                )
            }

        cx, cy, outer_r = fallback

        SCORE_RATIO = 0.76

        self.board_ellipse = None
        self.board_center = (int(cx), int(cy))
        self.board_radius = int(outer_r * SCORE_RATIO)

        return {
            "status": "ok",
            "method": "hough_fallback",
            "center": list(self.board_center),
            "radius": self.board_radius,
            "outer_radius": int(outer_r),
        }

    # ------------------------------------------------------------------
    # YOLO pose calibration (méthode 0)
    # ------------------------------------------------------------------

    def _fit_board_from_yolo(self, frame) -> bool:
        """
        Utilise un modèle YOLO11n-pose custom entraîné sur la cible de fléchettes.
        Détecte 5 keypoints :
          0 : centre (bull)
          1 : haut du double extérieur
          2 : droite du double extérieur
          3 : bas du double extérieur
          4 : gauche du double extérieur

        Ces 5 points sont passés à calibrate_perspective_manual() qui calcule
        l'homographie et met à jour board_center / board_radius / board_homography.

        Retourne True si la calibration a réussi.
        Le modèle est chargé en lazy depuis models/dartboard-pose.pt.
        Si le fichier n'existe pas, retourne False silencieusement.
        """
        # Cherche le modèle dans <projet>/models/dartboard-pose.pt
        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'dartboard-pose.pt'
        )
        model_path = os.path.normpath(model_path)

        if not os.path.exists(model_path):
            return False  # modèle pas encore entraîné → on passe à la méthode suivante

        # Chargement lazy (une seule fois)
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO  # importé ici pour ne pas bloquer si absent
                self._yolo_model = YOLO(model_path)
            except Exception:
                self._yolo_model = None
                return False

        try:
            result = self._yolo_model(frame, verbose=False)[0]
        except Exception:
            return False

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return False

        kpts = result.keypoints.xy[0].cpu().numpy()   # (5, 2)

        # Confiances disponibles ?
        if result.keypoints.conf is not None:
            confs = result.keypoints.conf[0].cpu().numpy()
        else:
            confs = np.ones(len(kpts), dtype=np.float32)

        if len(kpts) < 5:
            return False

        # Rejet si confiance < 30 % sur n'importe quel keypoint
        if np.any(confs[:5] < 0.30):
            return False

        # YOLO retourne (0, 0) pour les keypoints non détectés
        if np.any((kpts[:5, 0] == 0) & (kpts[:5, 1] == 0)):
            return False

        center = (float(kpts[0, 0]), float(kpts[0, 1]))
        top    = (float(kpts[1, 0]), float(kpts[1, 1]))
        right  = (float(kpts[2, 0]), float(kpts[2, 1]))
        bottom = (float(kpts[3, 0]), float(kpts[3, 1]))
        left   = (float(kpts[4, 0]), float(kpts[4, 1]))

        # Sauvegarde les 5 points pour l'overlay debug (cyan)
        self._cal_debug_pts = np.array(
            [center, top, right, bottom, left], dtype=np.float32
        )

        return self.calibrate_perspective_manual(center, top, right, bottom, left)

    def _fit_board_from_intersections(self, frame):
        """
        Détecte les intersections RÉELLES des barres métalliques :
        HoughLinesP trouve les segments de fil, puis on calcule uniquement
        les croisements où deux segments se coupent physiquement (vérification
        que le point est dans la boîte englobante des deux segments).
        Le grain du bois est quasi-parallèle → peu/pas d'intersections parasites.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        min_dim = min(h, w)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 90)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=15,
            maxLineGap=6,
        )

        if lines is None or len(lines) < 10:
            self._cal_debug_pts = None
            return None

        # Garder les 400 segments les plus longs (limite la complexité O(N²))
        raw = []
        for l in lines[:, 0]:
            x1, y1, x2, y2 = float(l[0]), float(l[1]), float(l[2]), float(l[3])
            raw.append((math.hypot(x2 - x1, y2 - y1), x1, y1, x2, y2))
        raw.sort(reverse=True)
        raw = raw[:400]

        segs = []
        for (_, x1, y1, x2, y2) in raw:
            dx, dy = x2 - x1, y2 - y1
            angle = math.atan2(dy, dx) % math.pi
            A, B = dy, -dx
            n = math.hypot(A, B)
            if n == 0:
                continue
            A /= n; B /= n
            C = -(A * x1 + B * y1)   # équation : A*x + B*y + C = 0
            segs.append({
                'angle': angle, 'A': A, 'B': B, 'C': C,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            })

        TOL = 14.0  # tolérance pixels au-delà des extrémités

        def on_seg(px, py, s):
            return (min(s['x1'], s['x2']) - TOL <= px <= max(s['x1'], s['x2']) + TOL
                    and min(s['y1'], s['y2']) - TOL <= py <= max(s['y1'], s['y2']) + TOL)

        intersections = []
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                s1, s2 = segs[i], segs[j]
                diff = abs(s1['angle'] - s2['angle'])
                diff = min(diff, math.pi - diff)
                if diff < math.radians(20):   # ignorer segments quasi-parallèles
                    continue
                det = s1['A'] * s2['B'] - s2['A'] * s1['B']
                if abs(det) < 1e-6:
                    continue
                # Intersection de deux droites (forme A*x + B*y + C = 0)
                px = (s1['B'] * s2['C'] - s2['B'] * s1['C']) / det
                py = (s1['C'] * s2['A'] - s2['C'] * s1['A']) / det
                if not (0 <= px < w and 0 <= py < h):
                    continue
                # On ne garde que les vrais croisements (pas les prolongements)
                if on_seg(px, py, s1) and on_seg(px, py, s2):
                    intersections.append((px, py))

        if len(intersections) < 15:
            self._cal_debug_pts = None
            return None

        pts = np.array(intersections, dtype=np.float32)
        self._cal_debug_pts = pts

        # Centre via densité (bin 2 % de min_dim)
        bin_size = max(12, int(min_dim * 0.02))
        bins_arr = np.floor(pts / bin_size).astype(np.int32)
        unique_b, counts = np.unique(bins_arr, axis=0, return_counts=True)
        best = unique_b[np.argmax(counts)]
        cguess = (best.astype(np.float32) + 0.5) * bin_size
        d_guess = np.linalg.norm(pts - cguess, axis=1)
        cluster = pts[d_guess < bin_size * 5]
        if len(cluster) < 8:
            return None
        cx0 = float(np.median(cluster[:, 0]))
        cy0 = float(np.median(cluster[:, 1]))

        dx_arr = pts[:, 0] - cx0
        dy_arr = pts[:, 1] - cy0
        dist_arr = np.sqrt(dx_arr ** 2 + dy_arr ** 2)
        ang_arr  = (np.degrees(np.arctan2(dy_arr, dx_arr)) + 360.0) % 360.0

        # Bord externe : point le plus éloigné par tranche de 9°
        outer_pts = []
        for b in range(40):
            a_min = b * 9.0
            idx = np.where((ang_arr >= a_min) & (ang_arr < a_min + 9.0))[0]
            if len(idx) == 0:
                continue
            outer_pts.append(pts[idx[np.argmax(dist_arr[idx])]])

        if len(outer_pts) < 12:
            return None

        outer_pts = np.array(outer_pts, dtype=np.float32)
        try:
            ell = cv2.fitEllipse(outer_pts.reshape(-1, 1, 2))
        except cv2.error:
            return None

        (ex, ey), (axis_a, axis_b), angle = ell
        sm = min(axis_a, axis_b)
        bg = max(axis_a, axis_b)

        if sm < min_dim * 0.20:               return None
        if bg > min_dim * 1.10:               return None
        if bg / sm > 2.5:                      return None
        if not (0 <= ex < w and 0 <= ey < h):  return None

        return ell

    def _fit_outer_board_circle_hough(self, frame):
        """
        Détecte le cercle extérieur noir de la cible.
        Retourne (cx, cy, radius) ou None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)

        min_r = int(min(h, w) * 0.25)
        max_r = int(min(h, w) * 0.80)

        candidates = []

        for param2 in (80, 65, 50, 38, 28, 20):
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min(h, w) // 3,
                param1=90,
                param2=param2,
                minRadius=min_r,
                maxRadius=max_r,
            )

            if circles is None:
                continue

            for c in circles[0]:
                cx, cy, r = float(c[0]), float(c[1]), float(c[2])

                # Rejette les cercles trop collés aux bords
                if cx - r < -50 or cy - r < -50 or cx + r > w + 50 or cy + r > h + 50:
                    continue

                # Score simple : on préfère une cible assez proche du centre de l'image
                img_cx, img_cy = w / 2.0, h / 2.0
                center_dist = math.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)

                score = -center_dist + r * 0.2
                candidates.append((score, cx, cy, r))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        _score, cx, cy, r = candidates[0]

        return cx, cy, r

    def _fit_scoring_ellipse_from_colors(self, frame):
        """
        Détecte les zones rouges/vertes de la cible, récupère les points les plus
        externes par angle, puis fit une ellipse correspondant au bord extérieur
        du double.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # Masque rouge : deux plages car le rouge est coupé autour de 0/180 en HSV
        red_1 = cv2.inRange(hsv, (0, 55, 20), (12, 255, 255))
        red_2 = cv2.inRange(hsv, (165, 55, 20), (180, 255, 255))

        green = cv2.inRange(hsv, (38, 45, 20), (92, 255, 255))

        mask = cv2.bitwise_or(red_1, red_2)
        mask = cv2.bitwise_or(mask, green)

        # Nettoyage du masque
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self._cal_debug_mask = mask  # save for live overlay

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
        self._cal_debug_pts = outer_pts  # save for overlay display

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

    def calibrate_perspective_manual(
        self,
        center: Tuple[float, float],
        top: Tuple[float, float],
        right: Tuple[float, float],
        bottom: Tuple[float, float],
        left: Tuple[float, float],
    ) -> bool:
        """
        Calibration perspective avec 5 points :
        - center = centre de la bulle
        - top/right/bottom/left = bord extérieur du double
        """
        src = np.array([
            [0.0, 0.0],    # centre
            [0.0, -1.0],   # haut
            [1.0, 0.0],    # droite
            [0.0, 1.0],    # bas
            [-1.0, 0.0],   # gauche
        ], dtype=np.float32)

        dst = np.array([
            center,
            top,
            right,
            bottom,
            left,
        ], dtype=np.float32)

        H, _ = cv2.findHomography(src, dst, method=0)

        if H is None:
            return False

        self.board_homography = H
        self.board_homography_inv = np.linalg.inv(H)

        self.board_center = (int(center[0]), int(center[1]))

        distances = [
            math.hypot(top[0] - center[0], top[1] - center[1]),
            math.hypot(right[0] - center[0], right[1] - center[1]),
            math.hypot(bottom[0] - center[0], bottom[1] - center[1]),
            math.hypot(left[0] - center[0], left[1] - center[1]),
        ]

        self.board_radius = int(sum(distances) / len(distances))

        self.board_ellipse = None

        return True


    def _project_board_points(self, points):
        """
        Convertit des points normalisés de la cible vers l'image caméra.
        """
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, self.board_homography)
        return projected.reshape(-1, 2).astype(np.int32)


    def _draw_projected_ring(self, frame, frac: float, color: tuple, thickness: int = 1):
        """
        Dessine un anneau corrigé par homographie.
        """
        points = []

        for i in range(240):
            a = 2.0 * math.pi * i / 240
            x = math.cos(a) * frac
            y = math.sin(a) * frac
            points.append((x, y))

        projected = self._project_board_points(points)
        cv2.polylines(frame, [projected], True, color, thickness)


    def _draw_projected_disk(self, frame, frac: float, color: tuple):
        """
        Dessine un disque rempli corrigé par homographie.
        """
        points = []

        for i in range(120):
            a = 2.0 * math.pi * i / 120
            x = math.cos(a) * frac
            y = math.sin(a) * frac
            points.append((x, y))

        projected = self._project_board_points(points)
        cv2.fillPoly(frame, [projected], color)

    def _to_board_norm(self, px: float, py: float) -> Tuple[float, float]:
        """
        Transform a pixel position to normalised board coordinates
        (radius 1.0 = outer double ring edge).

        If an ellipse was fitted, applies the inverse ellipse transform to
        correct perspective distortion before scoring.
        """

        if self.board_homography_inv is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            norm = cv2.perspectiveTransform(pt, self.board_homography_inv)[0][0]
            return float(norm[0]), float(norm[1])
        
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
