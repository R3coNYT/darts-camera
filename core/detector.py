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
        
        self.board_homography = None
        self.board_homography_inv = None

        # Last detected dart positions (for overlay)
        self._dart_positions: List[Tuple[float, float]] = []
        # Calibration debug: outer edge points used to fit the ellipse
        self._cal_debug_pts: Optional[object] = None
        # Calibration debug: HSV color mask (red+green zones detected)
        self._cal_debug_mask: Optional[object] = None

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

            # Points de calibration debug (bord extérieur rouge/vert détectés)
            if self._cal_debug_pts is not None:
                for pt in self._cal_debug_pts:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

            # Masque couleur calibration (zones rouge/vert détectées) en overlay cyan
            if self._cal_debug_mask is not None:
                overlay = frame.copy()
                overlay[self._cal_debug_mask > 0] = (0, 255, 255)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

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

        # Méthode 1 : barres métalliques
        ell = self._fit_board_from_wires(frame)

        if ell is not None:
            (ex, ey), (axis_a, axis_b), angle = ell

            self.board_ellipse = ell
            self.board_center = (int(ex), int(ey))
            self.board_radius = int((axis_a + axis_b) / 4.0)

            return {
                "status": "ok",
                "method": "metal_wires",
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

    def _fit_board_from_wires(self, frame):
        """
        Détecte les barres métalliques radiales.
        Les barres servent à trouver le centre, puis le rayon est estimé
        en testant les anneaux connus de la cible.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        min_dim = min(h, w)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 140)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=55,
            minLineLength=int(min_dim * 0.08),
            maxLineGap=22,
        )

        if lines is None:
            return None

        detected_lines = []

        def make_line(x1, y1, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)

            if length < min_dim * 0.08:
                return None

            A = dy
            B = -dx
            C = dx * y1 - dy * x1

            norm = math.hypot(A, B)
            if norm == 0:
                return None

            A /= norm
            B /= norm
            C /= norm

            angle = math.atan2(dy, dx) % math.pi

            return {
                "p1": (float(x1), float(y1)),
                "p2": (float(x2), float(y2)),
                "length": length,
                "angle": angle,
                "A": A,
                "B": B,
                "C": C,
            }

        for l in lines[:, 0]:
            line = make_line(l[0], l[1], l[2], l[3])
            if line is not None:
                detected_lines.append(line)

        if len(detected_lines) < 6:
            return None

        def intersect(l1, l2):
            A1, B1, C1 = l1["A"], l1["B"], l1["C"]
            A2, B2, C2 = l2["A"], l2["B"], l2["C"]

            det = A1 * B2 - A2 * B1
            if abs(det) < 1e-6:
                return None

            x = (B1 * C2 - B2 * C1) / det
            y = (C1 * A2 - C2 * A1) / det

            return x, y

        intersections = []

        for i in range(len(detected_lines)):
            for j in range(i + 1, len(detected_lines)):
                l1 = detected_lines[i]
                l2 = detected_lines[j]

                diff = abs(l1["angle"] - l2["angle"])
                diff = min(diff, math.pi - diff)

                # Ignore les lignes presque parallèles
                if diff < math.radians(15):
                    continue

                p = intersect(l1, l2)
                if p is None:
                    continue

                x, y = p

                if 0 <= x < w and 0 <= y < h:
                    intersections.append((x, y))

        if len(intersections) < 10:
            return None

        intersections = np.array(intersections, dtype=np.float32)

        # On cherche la zone où les intersections sont les plus concentrées.
        bin_size = max(14, int(min_dim * 0.025))
        bins = np.floor(intersections / bin_size).astype(np.int32)

        unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
        best_bin = unique_bins[np.argmax(counts)]

        center_guess = (best_bin.astype(np.float32) + 0.5) * bin_size
        distances = np.linalg.norm(intersections - center_guess, axis=1)

        cluster = intersections[distances < bin_size * 3.0]

        if len(cluster) < 8:
            return None

        cx, cy = np.median(cluster, axis=0)

        # On garde seulement les vraies lignes radiales qui passent près du centre.
        radial_lines = []
        max_center_dist = max(10, int(min_dim * 0.035))

        for l in detected_lines:
            dist_to_center = abs(l["A"] * cx + l["B"] * cy + l["C"])
            if dist_to_center <= max_center_dist:
                radial_lines.append(l)

        if len(radial_lines) < 5:
            return None

        # Le rayon n'est PAS pris avec les extrémités des lignes.
        # On le recalcule en cherchant le meilleur alignement des anneaux.
        radius = self._estimate_radius_from_known_rings(frame, float(cx), float(cy))

        if radius is None:
            return None

        # On retourne une ellipse circulaire propre.
        # Si plus tard tu veux gérer une caméra très inclinée, on pourra remplacer
        # ça par une vraie homographie.
        return (
            (float(cx), float(cy)),
            (float(radius * 2), float(radius * 2)),
            0.0,
        )

    def _estimate_radius_from_known_rings(self, frame, cx: float, cy: float):
        """
        Estime le rayon extérieur de la zone de score.
        On teste plusieurs rayons et on garde celui dont les anneaux connus
        tombent le mieux sur des contours détectés.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        min_dim = min(h, w)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
        edges = cv2.Canny(blurred, 30, 100)

        # Proportions de ta cible déjà utilisées dans get_current_frame()
        ring_fracs = (0.094, 0.582, 0.629, 0.953, 1.000)

        min_r = int(min_dim * 0.22)
        max_r = int(min_dim * 0.55)

        # Évite de tester des rayons impossibles par rapport à la position du centre.
        max_by_bounds = int(min(
            max(cx, w - cx),
            max(cy, h - cy),
            min_dim * 0.58,
        ))

        max_r = min(max_r, max_by_bounds)

        best_score = -1
        best_r = None

        for r in range(min_r, max_r + 1):
            total_score = 0.0
            valid_rings = 0

            for frac in ring_fracs:
                rr = max(3, int(r * frac))
                dr = max(2, int(r * 0.012))

                annulus = np.zeros((h, w), dtype=np.uint8)

                cv2.circle(
                    annulus,
                    (int(cx), int(cy)),
                    rr + dr,
                    255,
                    -1,
                )

                cv2.circle(
                    annulus,
                    (int(cx), int(cy)),
                    max(0, rr - dr),
                    0,
                    -1,
                )

                area = cv2.countNonZero(annulus)
                if area == 0:
                    continue

                edge_count = cv2.countNonZero(
                    cv2.bitwise_and(edges, edges, mask=annulus)
                )

                density = edge_count / area

                total_score += density
                valid_rings += 1

            if valid_rings == 0:
                continue

            score = total_score / valid_rings

            if score > best_score:
                best_score = score
                best_r = r

        if best_r is None:
            return None

        # Sécurité : si le score est vraiment trop faible, on considère que ça a raté.
        if best_score < 0.025:
            return None

        return int(best_r)

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
