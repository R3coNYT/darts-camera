"""
Game logic for Darts Camera.

Supported modes:
  501 / 301  – classic x01 with double-out rule
  Cricket    – close numbers 15-20 + Bull, score on open numbers
  Around the Clock – hit 1 → 2 → … → 20 → Bull in order
  Free       – no rules, just record throws
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .board import CRICKET_NUMBERS
from .checkout import get_checkout


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class GameMode(str, Enum):
    MODE_501       = "501"
    MODE_301       = "301"
    CRICKET        = "cricket"
    AROUND_CLOCK   = "around_the_clock"
    FREE           = "free"


@dataclass
class DartThrow:
    score: int          # Point value of this dart
    label: str          # e.g. 'T20', 'D16', 'B', 'MISS'
    x_norm: Optional[float] = None   # Board-normalized coords (–1..1)
    y_norm: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "x": self.x_norm,
            "y": self.y_norm,
        }


@dataclass
class Turn:
    player_name: str
    darts: List[DartThrow] = field(default_factory=list)
    scored: int = 0        # Points counted in this turn
    busted: bool = False

    def add(self, throw: DartThrow) -> None:
        self.darts.append(throw)
        if not self.busted:
            self.scored += throw.score


@dataclass
class Player:
    name: str
    team: Optional[str] = None

    # x01 state
    remaining: int = 0

    # Cricket state
    cricket_hits: Dict[int, int] = field(default_factory=dict)
    cricket_score: int = 0

    # Around-the-clock state  (stores current target number 1–20, then 25)
    atc_target: int = 1

    # Stats
    total_darts: int = 0
    total_counted: int = 0   # Points that actually counted (not busted)
    rounds_played: int = 0
    turns: List[Turn] = field(default_factory=list)

    @property
    def avg_per_dart(self) -> float:
        return round(self.total_counted / self.total_darts, 2) if self.total_darts else 0.0

    @property
    def avg_per_round(self) -> float:
        return round(self.total_counted / self.rounds_played, 2) if self.rounds_played else 0.0

    def cricket_closed(self, number: int) -> bool:
        return self.cricket_hits.get(number, 0) >= 3

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "team": self.team,
            "remaining": self.remaining,
            "cricket_hits": self.cricket_hits,
            "cricket_score": self.cricket_score,
            "atc_target": self.atc_target,
            "total_darts": self.total_darts,
            "total_counted": self.total_counted,
            "rounds_played": self.rounds_played,
            "avg_per_dart": self.avg_per_dart,
            "avg_per_round": self.avg_per_round,
        }


# ---------------------------------------------------------------------------
# Main Game class
# ---------------------------------------------------------------------------

class Game:
    def __init__(
        self,
        mode: GameMode,
        player_names: List[str],
        double_in: bool = False,
        double_out: bool = True,
    ):
        if not player_names:
            raise ValueError("Au moins un joueur requis.")

        self.mode = GameMode(mode)
        self.double_in = double_in
        self.double_out = double_out
        self.players: List[Player] = [Player(name=n) for n in player_names]
        self.current_player_idx = 0
        self.round_num = 1
        self.current_turn: Optional[Turn] = None
        self.game_over = False
        self.winner: Optional[str] = None
        self.started_at = time.time()

        # Per-mode initialization
        start = 501 if self.mode == GameMode.MODE_501 else 301
        for p in self.players:
            if self.mode in (GameMode.MODE_501, GameMode.MODE_301):
                p.remaining = start
            elif self.mode == GameMode.CRICKET:
                p.cricket_hits = {n: 0 for n in CRICKET_NUMBERS}
                p.cricket_score = 0
            elif self.mode == GameMode.AROUND_CLOCK:
                p.atc_target = 1

        self._start_turn()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    def _start_turn(self) -> None:
        self.current_turn = Turn(player_name=self.current_player.name)

    def _parse_label(self, label: str):
        """
        Parse a label like 'T20', 'D16', 'S5', 'B', 'DB', 'MISS'.
        Returns (multiplier, number) or (0, 0) for MISS.
        Special: B → (1, 25),  DB → (2, 25).
        """
        if label == "MISS":
            return 0, 0
        if label == "B":
            return 1, 25
        if label == "DB":
            return 2, 25
        mult_map = {"S": 1, "D": 2, "T": 3}
        try:
            return mult_map[label[0]], int(label[1:])
        except (KeyError, ValueError, IndexError):
            return 0, 0

    def _is_valid_double(self, label: str) -> bool:
        """True if the label represents a valid finishing double."""
        if label == "DB":
            return True
        if label.startswith("D"):
            try:
                int(label[1:])
                return True
            except ValueError:
                pass
        return False

    # ------------------------------------------------------------------
    # Public: throw a dart
    # ------------------------------------------------------------------

    def throw_dart(
        self,
        score: int,
        label: str,
        x_norm: float = None,
        y_norm: float = None,
    ) -> dict:
        """
        Record one dart throw for the current player.
        Returns a result dict consumed by the Flask layer.
        """
        if self.game_over:
            return {"error": "La partie est terminée."}
        if len(self.current_turn.darts) >= 3:
            return {"error": "Ce tour a déjà 3 fléchettes."}

        throw = DartThrow(score=score, label=label, x_norm=x_norm, y_norm=y_norm)
        self.current_player.total_darts += 1

        result: dict = {
            "throw": throw.to_dict(),
            "player": self.current_player.name,
            "dart_number": len(self.current_turn.darts) + 1,
            "busted": False,
            "won": False,
            "checkout_suggestion": None,
            "remaining": None,
        }

        if self.mode in (GameMode.MODE_501, GameMode.MODE_301):
            result.update(self._x01_throw(throw))
        elif self.mode == GameMode.CRICKET:
            result.update(self._cricket_throw(throw))
        elif self.mode == GameMode.AROUND_CLOCK:
            result.update(self._atc_throw(throw))
        else:  # FREE
            self.current_turn.add(throw)
            self.current_player.total_counted += score

        # Checkout suggestion after throw (x01 only, if not yet won/busted)
        if (
            self.mode in (GameMode.MODE_501, GameMode.MODE_301)
            and not result.get("won")
            and not result.get("busted")
        ):
            rem = self.current_player.remaining
            result["remaining"] = rem
            result["checkout_suggestion"] = get_checkout(rem)

        return result

    # ------------------------------------------------------------------
    # Mode-specific throw handlers
    # ------------------------------------------------------------------

    def _x01_throw(self, throw: DartThrow) -> dict:
        player = self.current_player
        new_rem = player.remaining - throw.score

        # Bust: below 0, lands on 1, or 0 without valid double
        if new_rem < 0 or new_rem == 1:
            self.current_turn.busted = True
            self.current_turn.darts.append(throw)
            return {"busted": True, "remaining": player.remaining}

        if new_rem == 0:
            if self.double_out and not self._is_valid_double(throw.label):
                self.current_turn.busted = True
                self.current_turn.darts.append(throw)
                return {"busted": True, "remaining": player.remaining}
            # Valid finish
            self.current_turn.add(throw)
            player.remaining = 0
            player.total_counted += throw.score
            self.winner = player.name
            self.game_over = True
            return {"won": True, "remaining": 0}

        # Normal hit
        self.current_turn.add(throw)
        player.remaining = new_rem
        player.total_counted += throw.score
        return {"remaining": new_rem, "checkout_suggestion": get_checkout(new_rem)}

    def _cricket_throw(self, throw: DartThrow) -> dict:
        player = self.current_player
        mult, number = self._parse_label(throw.label)
        self.current_turn.add(throw)
        result: dict = {}

        if number in CRICKET_NUMBERS and mult > 0:
            current_hits = player.cricket_hits.get(number, 0)
            total_hits = current_hits + mult

            if current_hits < 3:
                player.cricket_hits[number] = min(3, total_hits)
                excess = total_hits - 3
            else:
                excess = mult  # Already closed → all hits score

            # Score excess hits if ANY opponent has not closed this number
            if excess > 0 and player.cricket_hits[number] >= 3:
                opponents_open = any(
                    p.cricket_hits.get(number, 0) < 3
                    for p in self.players
                    if p.name != player.name
                )
                if opponents_open:
                    pts = excess * (50 if number == 25 else number)
                    player.cricket_score += pts
                    player.total_counted += pts
                    result["cricket_scored"] = pts

        result["cricket_hits"] = dict(player.cricket_hits)
        result["cricket_score"] = player.cricket_score

        if self._check_cricket_win(player):
            self.winner = player.name
            self.game_over = True
            result["won"] = True

        return result

    def _check_cricket_win(self, player: Player) -> bool:
        if not all(player.cricket_hits.get(n, 0) >= 3 for n in CRICKET_NUMBERS):
            return False
        return all(
            player.cricket_score >= p.cricket_score
            for p in self.players
            if p.name != player.name
        )

    def _atc_throw(self, throw: DartThrow) -> dict:
        player = self.current_player
        target = player.atc_target
        mult, number = self._parse_label(throw.label)
        self.current_turn.add(throw)

        hit = number == target and mult > 0
        result: dict = {"current_target": target}

        if hit:
            player.total_counted += 1
            if target == 25:
                self.winner = player.name
                self.game_over = True
                result["won"] = True
            else:
                player.atc_target = target + 1 if target < 20 else 25
                result["next_target"] = player.atc_target
        return result

    # ------------------------------------------------------------------
    # Public: end turn / undo
    # ------------------------------------------------------------------

    def end_turn(self) -> dict:
        """Finalise current turn and advance to next player."""
        if self.game_over:
            return {"error": "La partie est terminée."}

        player = self.current_player
        turn = self.current_turn

        # Update stats
        player.rounds_played += 1
        player.turns.append(turn)

        # Restore x01 score on bust
        if turn.busted and self.mode in (GameMode.MODE_501, GameMode.MODE_301):
            # remaining was never updated on busted throws – no action needed
            pass

        # Advance
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        if self.current_player_idx == 0:
            self.round_num += 1

        self._start_turn()
        return {
            "next_player": self.current_player.name,
            "round": self.round_num,
        }

    def undo_last_dart(self) -> dict:
        """Remove the last recorded dart from the current turn."""
        if not self.current_turn or not self.current_turn.darts:
            return {"error": "Aucune fléchette à annuler."}

        last = self.current_turn.darts.pop()
        player = self.current_player
        player.total_darts = max(0, player.total_darts - 1)

        if self.mode in (GameMode.MODE_501, GameMode.MODE_301):
            if self.current_turn.busted:
                self.current_turn.busted = False
            else:
                player.remaining += last.score
                player.total_counted = max(0, player.total_counted - last.score)
                self.current_turn.scored = max(0, self.current_turn.scored - last.score)
        elif self.mode == GameMode.CRICKET:
            # Simplified: re-subtract what was scored (cricket undo is best-effort)
            mult, number = self._parse_label(last.label)
            if number in CRICKET_NUMBERS and mult > 0:
                player.cricket_hits[number] = max(0, player.cricket_hits.get(number, 0) - mult)
            player.total_counted = max(0, player.total_counted - last.score)
            player.cricket_score = max(0, player.cricket_score - (last.score if last.score > 0 else 0))
        else:
            player.total_counted = max(0, player.total_counted - last.score)

        return {"undone": last.label, "dart_count": len(self.current_turn.darts)}

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        player = self.current_player
        checkout = None
        if self.mode in (GameMode.MODE_501, GameMode.MODE_301) and not self.game_over:
            checkout = get_checkout(player.remaining)

        return {
            "mode": self.mode.value,
            "round": self.round_num,
            "current_player": player.name,
            "current_player_idx": self.current_player_idx,
            "current_turn_darts": [d.to_dict() for d in self.current_turn.darts] if self.current_turn else [],
            "current_turn_busted": self.current_turn.busted if self.current_turn else False,
            "players": [p.to_dict() for p in self.players],
            "game_over": self.game_over,
            "winner": self.winner,
            "checkout_suggestion": checkout,
            "double_out": self.double_out,
            "double_in": self.double_in,
        }
