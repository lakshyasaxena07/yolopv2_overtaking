# modules/ttc_engine.py — TTC Engine + Safety Decision

from dataclasses import dataclass
from enum        import Enum
from collections import deque


class SafetyLevel(Enum):
    SAFE   = "SAFE"
    RISKY  = "RISKY"
    UNSAFE = "UNSAFE"


@dataclass
class SafetyDecision:
    level:         SafetyLevel
    ttc_min:       float
    num_threats:   int
    closest_dist:  float
    closing_speed: float
    reason:        str

    @property
    def color_bgr(self):
        return {
            SafetyLevel.SAFE:   (0,   220, 90),
            SafetyLevel.RISKY:  (0,   180, 255),
            SafetyLevel.UNSAFE: (0,   50,  255)
        }[self.level]

    @property
    def display_text(self):
        return self.level.value


class TTCEngine:
    """
    Time-To-Collision based safety decision engine.

    Decision Logic:
    1. Overtake feasibility check (road/barrier/vehicle)
    2. Oncoming threat TTC calculation
    3. Congestion check (too close vehicles)
    4. Stable decision (5-frame history)
    """

    SIZE_PENALTY = {
        "car":        1.0,
        "motorcycle": 0.9,
        "bus":        1.4,
        "truck":      1.4,
        "vehicle":    1.1
    }

    def __init__(self, cfg=None):
        self.TTC_SAFE  = 5.0
        self.TTC_RISKY = 2.5
        self._history  = deque(maxlen=5)

    def compute_ttc(self, distance_m, approach_rate_mps):
        if approach_rate_mps <= 0.1:
            return float("inf")
        return max(0.0, distance_m / approach_rate_mps)

    def evaluate(self, enriched_tracks,
                 overtake_feasibility=None):
        """
        Main evaluation function.

        Args:
            enriched_tracks     : list from estimator
            overtake_feasibility: dict from lane_detector

        Returns:
            SafetyDecision
        """

        # ── Check 1: Overtake physically possible? ────────────
        if overtake_feasibility is not None:
            if not overtake_feasibility["feasible"]:
                reason   = (f"NO OVERTAKE | "
                            f"{overtake_feasibility['reason']}")
                decision = SafetyDecision(
                    level         = SafetyLevel.UNSAFE,
                    ttc_min       = 999,
                    num_threats   = 0,
                    closest_dist  = 999.0,
                    closing_speed = 0.0,
                    reason        = reason
                )
                self._history.append(decision)
                return decision

        # ── Relevant tracks only ──────────────────────────────
        relevant = [
            t for t in enriched_tracks
            if t.get("is_relevant", True)
        ]

        # ── Check 2: Oncoming threats ─────────────────────────
        threats = [
            t for t in relevant
            if t.get("is_oncoming") and
               t.get("approach_rate", 0) > 0.3
        ]

        # ── Check 3: Too close vehicles ───────────────────────
        too_close_vehicles = [
            t for t in relevant
            if t.get("is_too_close")
        ]

        # ── Check 4: Overtake lane blocked? ───────────────────
        overtake_blocked = any(
            t["zone"] == "overtake_lane" and
            t["distance_m"] < 30
            for t in relevant
        )

        # ── All clear → SAFE ──────────────────────────────────
        if (not threats and
                not too_close_vehicles and
                not overtake_blocked):

            if (overtake_feasibility and
                    overtake_feasibility["feasible"]):
                safe_reason = (
                    f"SAFE | "
                    f"{overtake_feasibility['reason']}"
                )
            else:
                safe_reason = "Road clear — Safe to overtake"

            decision = SafetyDecision(
                level         = SafetyLevel.SAFE,
                ttc_min       = float("inf"),
                num_threats   = 0,
                closest_dist  = 999.0,
                closing_speed = 0.0,
                reason        = safe_reason
            )
            self._history.append(decision)
            return decision

        # ── TTC for oncoming threats ──────────────────────────
        ttc_list = []
        for t in threats:
            penalty      = self.SIZE_PENALTY.get(
                t["class_name"], 1.1)
            eff_approach = t["approach_rate"] * penalty
            ttc          = self.compute_ttc(
                t["distance_m"], eff_approach)
            ttc_list.append((ttc, t))

        if ttc_list:
            ttc_list.sort(key=lambda x: x[0])
            min_ttc, worst = ttc_list[0]
            closest_dist   = worst["distance_m"]
            closing_kph    = worst["approach_rate"] * 3.6
        else:
            min_ttc      = float("inf")
            closest_dist = 999.0
            closing_kph  = 0.0

        num_threats = len(threats)

        # ── UNSAFE conditions ─────────────────────────────────
        unsafe = (
            (min_ttc < self.TTC_RISKY) or
            (closest_dist < 40.0) or
            (num_threats >= 2) or
            (len(too_close_vehicles) >= 2)
        )

        # ── RISKY conditions ──────────────────────────────────
        risky = (
            (min_ttc < self.TTC_SAFE) or
            (closest_dist < 80.0) or
            overtake_blocked or
            (len(too_close_vehicles) >= 1)
        )

        if unsafe:
            level  = SafetyLevel.UNSAFE
            reason = self._build_reason(
                min_ttc, closest_dist, num_threats,
                too_close_vehicles, overtake_blocked,
                "CRITICAL"
            )
        elif risky:
            level  = SafetyLevel.RISKY
            reason = self._build_reason(
                min_ttc, closest_dist, num_threats,
                too_close_vehicles, overtake_blocked,
                "CAUTION"
            )
        else:
            level  = SafetyLevel.SAFE
            reason = "Road clear — Safe to overtake"

        decision = SafetyDecision(
            level         = level,
            ttc_min       = min_ttc if min_ttc != float("inf")
                                    else 999,
            num_threats   = num_threats,
            closest_dist  = closest_dist,
            closing_speed = closing_kph,
            reason        = reason
        )
        self._history.append(decision)
        return decision

    def get_stable_decision(self):
        """5-frame majority vote for stable display."""
        if not self._history:
            return None
        levels = [d.level for d in self._history]
        for level in [SafetyLevel.UNSAFE,
                      SafetyLevel.RISKY,
                      SafetyLevel.SAFE]:
            if levels.count(level) >= 2:
                for d in reversed(self._history):
                    if d.level == level:
                        return d
        return self._history[-1]

    def _build_reason(self, ttc, dist, threats,
                      too_close, blocked, tag):
        parts = [tag]
        if ttc < 999:
            parts.append(f"TTC:{ttc:.1f}s")
        if dist < 999:
            parts.append(f"Dist:{dist:.0f}m")
        if threats > 0:
            parts.append(f"Oncoming:{threats}")
        if too_close:
            parts.append(f"TooClose:{len(too_close)}")
        if blocked:
            parts.append("OvertakeLaneBlocked")
        return "  |  ".join(parts)