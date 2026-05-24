# modules/ttc_engine.py — Kinematic TTC Safety Engine (Production)
#
# SAFETY-CRITICAL REDESIGN:
# - Replaces static distance thresholds with physics-based TTC
# - TTC = Distance / RelativeVelocity (per-vehicle, per-direction)
# - Same-direction threshold: 4.0s (closing from behind)
# - Oncoming threshold: 6.0s (head-on approach — higher due to combined velocity)
# - Computes TTC for ALL relevant tracks, not just oncoming

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
    Kinematic Time-To-Collision Safety Engine.

    SAFETY-CRITICAL PHYSICS:
    TTC = Distance / max(RelativeVelocity, 0.1)

    At Ego=100km/h, Oncoming=80km/h:
      Relative = 50 m/s, Distance = 300m → TTC = 6.0s → UNSAFE threshold
      Old static gap (30m) would give 0.6s — lethal.

    At Ego=100km/h, Same-dir Target=40km/h:
      Relative = 16.67 m/s, Distance = 67m → TTC = 4.0s → UNSAFE threshold
      Old static gap (25m) would give 1.5s — insufficient for human reaction.

    Decision thresholds are direction-aware:
    - Oncoming: TTC < 6.0s → UNSAFE (combined velocity = very fast closure)
    - Same-direction: TTC < 4.0s → UNSAFE (lower relative velocity)
    """

    SIZE_PENALTY = {
        "car":        1.0,
        "motorcycle": 0.9,
        "bus":        1.4,
        "truck":      1.4,
        "vehicle":    1.1
    }

    def __init__(self, cfg=None):
        # SAFETY-CRITICAL: Use direction-aware thresholds from config
        self.TTC_UNSAFE_SAME_DIR = getattr(cfg, 'TTC_UNSAFE_SAME_DIR', 4.0)
        self.TTC_UNSAFE_ONCOMING = getattr(cfg, 'TTC_UNSAFE_ONCOMING', 6.0)
        self.TTC_RISKY_SAME_DIR  = self.TTC_UNSAFE_SAME_DIR + 2.0   # 6.0s
        self.TTC_RISKY_ONCOMING  = self.TTC_UNSAFE_ONCOMING + 2.0   # 8.0s
        self._history = deque(maxlen=5)

    def compute_ttc(self, distance_m, approach_rate_mps):
        """
        TTC = Distance / RelativeVelocity

        SAFETY-CRITICAL: Floor approach_rate at 0.1 m/s to prevent division by zero.
        Returns inf if vehicle is not approaching (stationary or moving away).
        """
        if approach_rate_mps <= 0.1:
            return float("inf")
        return max(0.0, distance_m / approach_rate_mps)

    def evaluate(self, enriched_tracks,
                 overtake_feasibility=None):
        """
        Main evaluation — computes per-vehicle TTC for ALL relevant tracks.

        SAFETY-CRITICAL CHANGE: Old engine only computed TTC for oncoming threats.
        Now computes TTC for same-direction vehicles too (rear-end collision risk).
        Uses direction-aware thresholds (4.0s same-dir, 6.0s oncoming).
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

        # ── Compute per-vehicle TTC (ALL directions) ──────────
        ttc_results = []
        for t in relevant:
            approach = t.get("approach_rate", 0)
            if approach <= 0.1:
                continue   # Not approaching — no TTC threat

            penalty = self.SIZE_PENALTY.get(t["class_name"], 1.1)
            eff_approach = approach * penalty
            ttc_val = self.compute_ttc(t["distance_m"], eff_approach)

            # Direction-aware threshold selection
            is_oncoming = t.get("is_oncoming", False)
            ttc_unsafe_thresh = self.TTC_UNSAFE_ONCOMING if is_oncoming else self.TTC_UNSAFE_SAME_DIR
            ttc_risky_thresh  = self.TTC_RISKY_ONCOMING if is_oncoming else self.TTC_RISKY_SAME_DIR

            ttc_results.append({
                "ttc": ttc_val,
                "track": t,
                "is_oncoming": is_oncoming,
                "ttc_unsafe_thresh": ttc_unsafe_thresh,
                "ttc_risky_thresh": ttc_risky_thresh,
            })

        # ── Check: Overtake lane blocked? ─────────────────────
        overtake_blocked = any(
            t["zone"] == "overtake_lane" and
            t["distance_m"] < 30
            for t in relevant
        )

        # ── All clear → SAFE ─────────────────────────────────
        if not ttc_results and not overtake_blocked:
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

        # ── Evaluate TTC threats ──────────────────────────────
        unsafe_threats = [r for r in ttc_results if r["ttc"] < r["ttc_unsafe_thresh"]]
        risky_threats  = [r for r in ttc_results if r["ttc"] < r["ttc_risky_thresh"]]

        # Find worst-case values for reporting
        if ttc_results:
            ttc_results.sort(key=lambda r: r["ttc"])
            min_ttc = ttc_results[0]["ttc"]
            worst = ttc_results[0]["track"]
            closest_dist = worst["distance_m"]
            closing_kph = worst.get("approach_rate", 0) * 3.6
        else:
            min_ttc = float("inf")
            closest_dist = 999.0
            closing_kph = 0.0

        num_threats = len(unsafe_threats) + len(risky_threats)

        # ── UNSAFE conditions (any single vehicle below UNSAFE threshold) ──
        if unsafe_threats or len(risky_threats) >= 3:
            level  = SafetyLevel.UNSAFE
            reason = self._build_reason(
                min_ttc, closest_dist, len(unsafe_threats),
                risky_threats, overtake_blocked,
                "CRITICAL"
            )
        elif risky_threats or overtake_blocked:
            level  = SafetyLevel.RISKY
            reason = self._build_reason(
                min_ttc, closest_dist, len(unsafe_threats),
                risky_threats, overtake_blocked,
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

    def _build_reason(self, ttc, dist, unsafe_count,
                      risky_threats, blocked, tag):
        parts = [tag]
        if ttc < 999:
            parts.append(f"TTC:{ttc:.1f}s")
        if dist < 999:
            parts.append(f"Dist:{dist:.0f}m")
        if unsafe_count > 0:
            parts.append(f"Threats:{unsafe_count}")
        if risky_threats:
            parts.append(f"Risky:{len(risky_threats)}")
        if blocked:
            parts.append("OvertakeLaneBlocked")
        return "  |  ".join(parts)