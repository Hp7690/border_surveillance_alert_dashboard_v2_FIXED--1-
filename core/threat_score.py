"""
threat_score.py
-----------------
Computes a unified 0-100 THREAT SCORE for every alert, combining
several factors into one number an operator can triage on at a glance
-- rather than only a 4-level severity label (LOW/MEDIUM/HIGH/CRITICAL).

Factors combined:
  1. Base severity weight       -- starting point from the existing
                                    severity tier (see alert_manager.PRIORITY)
  2. Event-type weight          -- a weapon is inherently worse than a
                                    generic person sighting, even at
                                    the same nominal severity tier
  3. Time-of-day                -- night-time events are weighted higher
                                    (harder for a human guard to spot,
                                    historically higher-risk window)
  4. Restricted-zone factor     -- an explicit virtual-fence hit adds weight
  5. Detection confidence       -- a low-confidence detection is
                                    discounted (less certain -> lower score)
  6. Repeat-behaviour bonus     -- handled separately in AlertManager
                                    (needs rolling state across alerts),
                                    added on top of this module's base score

This is intentionally a transparent, explainable weighted-sum model
(not a trained ML model) so operators can understand and, if needed,
retune WHY a given alert scored the way it did -- see the WEIGHTS dict
below, which is the only thing you'd need to touch to retune it.
"""

WEAPON_TYPES = {"weapon_detected", "gun_detected"}
INTRUSION_TYPES = {"intrusion", "night_intrusion"}
SUSPICIOUS_TYPES = {"crawling", "climbing", "sudden_approach", "erratic_movement", "loitering"}

BASE_SEVERITY_SCORE = {"CRITICAL": 70, "HIGH": 50, "MEDIUM": 30, "LOW": 12}

WEIGHTS = {
    "weapon_bonus": 18,
    "intrusion_bonus": 10,
    "suspicious_bonus": 8,
    "night_bonus": 8,
    "zone_bonus": 7,
    "low_confidence_penalty_max": 15,  # scaled by (1 - confidence)
}


def compute_base_score(event_type, severity, details=None, is_night=False):
    """Returns an int 0-100. Does NOT include the repeat-behaviour bonus
    (see AlertManager._repeat_bonus, which has the rolling state needed
    for that and adds it on top of this)."""
    details = details or {}
    score = BASE_SEVERITY_SCORE.get(severity, 20)

    if event_type in WEAPON_TYPES:
        score += WEIGHTS["weapon_bonus"]
    elif event_type in INTRUSION_TYPES:
        score += WEIGHTS["intrusion_bonus"]
    elif event_type in SUSPICIOUS_TYPES:
        score += WEIGHTS["suspicious_bonus"]

    if is_night:
        score += WEIGHTS["night_bonus"]

    if details.get("zone"):
        score += WEIGHTS["zone_bonus"]

    confidence = details.get("confidence")
    if confidence is not None:
        score -= round((1 - confidence) * WEIGHTS["low_confidence_penalty_max"])

    return max(0, min(100, score))


def score_band(score):
    """Human-readable band for a numeric score -- used for coloring in the UI."""
    if score >= 80:
        return "SEVERE"
    if score >= 60:
        return "ELEVATED"
    if score >= 35:
        return "MODERATE"
    return "MINIMAL"
