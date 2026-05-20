"""
Dartboard geometry and score calculation.

Coordinate system:
  - Angles: 0° = top (12 o'clock), positive clockwise
  - Radii: normalized to board scoring radius (1.0 = outer double ring)

Official WDF/BDO measurements used for zone radii.
"""
import math
from typing import Tuple, Optional

# Segment order clockwise from top (12 o'clock) – standard dartboard layout
SEGMENTS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Cricket numbers (standard)
CRICKET_NUMBERS = [15, 16, 17, 18, 19, 20, 25]

# Zone radii as fractions of board scoring radius (official measurements)
BULL_INNER_R  = 6.35  / 170.0   # Inner bullseye (50 pts) ≈ 0.0374
BULL_OUTER_R  = 15.9  / 170.0   # Outer bull     (25 pts) ≈ 0.0935
TRIPLE_INNER_R = 99.0  / 170.0  # Inner edge of triple ring ≈ 0.5824
TRIPLE_OUTER_R = 107.0 / 170.0  # Outer edge of triple ring ≈ 0.6294
DOUBLE_INNER_R = 162.0 / 170.0  # Inner edge of double ring ≈ 0.9529
DOUBLE_OUTER_R = 1.0             # Outer edge of double ring (board edge)

# Canvas drawing constants – segment half-angle in degrees
SEGMENT_HALF_ANGLE = 9.0  # 360 / 20 / 2


def angle_to_segment(angle_deg: float) -> int:
    """Return the segment number for a given board angle (0°=top, clockwise)."""
    idx = int((angle_deg % 360 + SEGMENT_HALF_ANGLE) / (SEGMENT_HALF_ANGLE * 2)) % 20
    return SEGMENTS[idx]


def pixel_to_score(
    px: float, py: float,
    center_x: float, center_y: float,
    board_pixel_radius: float,
) -> Tuple[int, str, int]:
    """
    Convert pixel coordinates to a dart score.

    Returns:
        (score_value, zone_label, multiplier)
        zone_label: 'BULL' | 'OUTER_BULL' | 'SINGLE' | 'TRIPLE' | 'DOUBLE' | 'MISS'
    """
    dx = px - center_x
    dy = py - center_y
    r_norm = math.sqrt(dx * dx + dy * dy) / board_pixel_radius

    if r_norm <= BULL_INNER_R:
        return 50, "BULL", 1

    if r_norm <= BULL_OUTER_R:
        return 25, "OUTER_BULL", 1

    # Angle from top, clockwise (atan2 with x as sin, -y as cos)
    angle = math.degrees(math.atan2(dx, -dy)) % 360
    segment = angle_to_segment(angle)

    if r_norm <= TRIPLE_INNER_R:
        return segment, "SINGLE", 1

    if r_norm <= TRIPLE_OUTER_R:
        return segment * 3, "TRIPLE", 3

    if r_norm <= DOUBLE_INNER_R:
        return segment, "SINGLE", 1

    if r_norm <= DOUBLE_OUTER_R:
        return segment * 2, "DOUBLE", 2

    return 0, "MISS", 0


def score_to_label(score: int, zone: str, multiplier: int) -> str:
    """Return a human-readable dart label: 'T20', 'D16', 'S5', 'B', 'DB', 'MISS'."""
    if zone == "BULL":
        return "DB"
    if zone == "OUTER_BULL":
        return "B"
    if zone == "MISS":
        return "MISS"
    base = score // multiplier
    return f"{['S','D','T'][multiplier - 1]}{base}"


def label_to_value(label: str) -> Optional[int]:
    """Convert 'T20', 'D16', 'S5', 'B', 'DB', 'MISS' → integer point value."""
    if label == "DB":
        return 50
    if label == "B":
        return 25
    if label == "MISS":
        return 0
    try:
        mult = {"S": 1, "D": 2, "T": 3}[label[0]]
        return mult * int(label[1:])
    except (KeyError, ValueError, IndexError):
        return None


def segment_center_angle(segment_number: int) -> float:
    """Return the center angle (°, 0=top, clockwise) of the given segment."""
    return (SEGMENTS.index(segment_number) * 18) % 360


def label_to_board_coords(
    label: str,
    cx: float = 0.0,
    cy: float = 0.0,
    board_radius: float = 1.0,
) -> Optional[Tuple[float, float]]:
    """
    Return the (x, y) canvas position for the center of a scored zone.
    Useful for highlighting checkout targets on the board.
    Returns None for MISS.
    """
    if label == "DB":
        return cx, cy
    if label == "B":
        return cx, cy - board_radius * (BULL_OUTER_R + BULL_INNER_R) / 2
    if label == "MISS":
        return None

    prefix = label[0]
    try:
        number = int(label[1:])
    except ValueError:
        return None

    angle_deg = segment_center_angle(number)
    # Convert "0=top, clockwise" to math convention (0=right, counter-clockwise):
    # canvas_angle = angle_deg - 90  (in degrees, x right, y down)
    angle_rad = math.radians(angle_deg - 90)

    radii = {
        "D": (DOUBLE_INNER_R + DOUBLE_OUTER_R) / 2,
        "T": (TRIPLE_INNER_R + TRIPLE_OUTER_R) / 2,
        "S": (BULL_OUTER_R + TRIPLE_INNER_R) / 2,
    }
    r_norm = radii.get(prefix, radii["S"])
    r = r_norm * board_radius

    return cx + math.cos(angle_rad) * r, cy + math.sin(angle_rad) * r
