"""
Checkout path suggestions for x01 game modes (501, 301).

Rules:
  - Last dart MUST be a double (D1–D20) or double bull (DB = 50).
  - Maximum 3 darts per checkout.
  - Score range: 2–170.

The module pre-computes all reachable checkouts at import time using an
exhaustive (but fast, ~50 000 iterations) search, sorted to prefer high-value
first darts so that the suggestion is as practical as possible.
"""
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# All single-dart scoring possibilities
# ---------------------------------------------------------------------------
_ALL_DARTS: List[tuple] = []
for _n in range(1, 21):
    _ALL_DARTS.append((_n,      f"S{_n}"))
    _ALL_DARTS.append((_n * 2,  f"D{_n}"))
    _ALL_DARTS.append((_n * 3,  f"T{_n}"))
_ALL_DARTS.append((25, "B"))    # Outer bull (25)
_ALL_DARTS.append((50, "DB"))   # Double bull (50)

# Preferred dart order: highest value first for better suggestions
_ALL_DARTS_DESC = sorted(_ALL_DARTS, key=lambda x: -x[0])

# Valid finishes: doubles only (D1–D20) + DB
_FINISHES: Dict[int, str] = {n * 2: f"D{n}" for n in range(1, 21)}
_FINISHES[50] = "DB"

# ---------------------------------------------------------------------------
# Build the checkout table once
# ---------------------------------------------------------------------------
_TABLE: Dict[int, List[str]] = {}


def _build() -> None:
    # 1-dart checkouts
    for val, lbl in _FINISHES.items():
        _TABLE[val] = [lbl]

    # 2-dart checkouts
    for v1, l1 in _ALL_DARTS_DESC:
        for fin_val, fin_lbl in sorted(_FINISHES.items(), key=lambda x: -x[0]):
            total = v1 + fin_val
            if 2 <= total <= 170 and total not in _TABLE:
                _TABLE[total] = [l1, fin_lbl]

    # 3-dart checkouts
    for v1, l1 in _ALL_DARTS_DESC:
        for v2, l2 in _ALL_DARTS_DESC:
            for fin_val, fin_lbl in sorted(_FINISHES.items(), key=lambda x: -x[0]):
                total = v1 + v2 + fin_val
                if 2 <= total <= 170 and total not in _TABLE:
                    _TABLE[total] = [l1, l2, fin_lbl]


_build()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_checkout(score: int) -> Optional[List[str]]:
    """
    Return the checkout path for *score*, or None if unreachable.

    Example:
        get_checkout(170) → ['T20', 'T20', 'DB']
        get_checkout(40)  → ['D20']
        get_checkout(169) → None
    """
    return _TABLE.get(score)


def format_checkout(score: int) -> str:
    """
    Human-readable checkout string, e.g. 'T20 → T20 → DB'.
    Returns a French message when no checkout exists.
    """
    path = get_checkout(score)
    if path is None:
        if score > 170:
            return "Hors portée (> 170)"
        if score == 1:
            return "Bust (1 restant)"
        return "Sortie impossible"
    return " → ".join(path)


def checkout_darts_needed(score: int) -> Optional[int]:
    """Return the number of darts needed to check out, or None."""
    path = get_checkout(score)
    return len(path) if path else None
