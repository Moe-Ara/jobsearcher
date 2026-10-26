"""Fit scoring: combines structured signals instead of trusting the LLM's overall number.

Each component is 0-10; the weighted average (weights from preferences.yaml) plus small bonuses /
penalties gives the fit score. Free to recompute at any time (no LLM call).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config_files import Preferences
from app.enums import FilterStatus, Recommendation


@dataclass
class ScoreResult:
    score: float | None
    recommendation: Recommendation
    breakdown: dict[str, Any] = field(default_factory=dict)


def _clamp(v: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, v))


def sector_score(sector: str | None, prefs: Preferences) -> float:
    s = prefs.scoring
    if not sector:
        return s.default_sector_score
    label = sector.lower()
    matches = [score for key, score in s.sector_scores.items() if key.lower() in label]
    return max(matches) if matches else s.default_sector_score


def role_family_score(role_family: str | None, prefs: Preferences) -> float:
    scores = prefs.scoring.role_family_scores
    if role_family and role_family in scores:
        return scores[role_family]
    if role_family:  # tolerate small naming drift from the model
        low = role_family.lower()
        for name, score in scores.items():
            if name.lower() in low or low in name.lower():
                return score
    return scores.get("Other", 3.0)


def location_score(country: str | None, location: str | None, remote: str | None, llm_location: float | None, prefs: Preferences) -> float:
    loc = prefs.locations
    if country and country in loc.country_scores:
        base = loc.country_scores[country]
    elif country:
        base = 2.0 if remote != "remote" else 6.0
    elif llm_location is not None:
        base = llm_location
    else:
        base = loc.unknown_country_score
    text = (location or "").lower()
    bonus = max((b for city, b in loc.preferred_cities.items() if city.lower() in text), default=0.0)
    return _clamp(base + bonus)


def compute_score(
    *,
    filter_status: str,
    analysis: dict[str, Any] | None,
    country: str | None,
    location: str | None,
    remote_status: str | None,
    seniority: str | None,
    best_source_type: str,
    target_priority: str | None,
    prefs: Preferences,
    years_required: float | None = None,
    candidate_years: float | None = None,
    location_override: float | None = None,
) -> ScoreResult:
    if filter_status == FilterStatus.REJECTED:
        return ScoreResult(None, Recommendation.REJECTED, {"reason": "hard filter"})
    if not analysis:
        return ScoreResult(None, Recommendation.PENDING, {"reason": "not analysed yet"})

    s = prefs.scoring
    # the job's own seniority (title / ATS label) is more reliable than a model's reading of it
    level = seniority if seniority in s.seniority_scores else (analysis.get("seniority") or "Unknown")
    seniority_fit = s.seniority_scores.get(level, s.seniority_scores.get("Unknown", 7.0))
    components = {
        "technical": float(analysis.get("technical_match", 0)),
        "experience": 0.7 * float(analysis.get("experience_match", 0)) + 0.3 * seniority_fit,
        "role_family": role_family_score(analysis.get("role_family"), prefs),
        "sector": sector_score(analysis.get("sector"), prefs),
        "location": location_override if location_override is not None
        else location_score(country, location, remote_status, analysis.get("location_match"), prefs),
        "career": float(analysis.get("career_match", 0)),
        "source": s.source_reliability.get(best_source_type, 7.0),
    }
    total_w = sum(w for k, w in s.weights.items() if k in components) or 1.0
    base = sum(components[k] * w for k, w in s.weights.items() if k in components) / total_w

    bonus = s.target_company_bonus.get(target_priority or "", 0.0)
    gaps = len(analysis.get("critical_gaps") or [])
    penalty = min(gaps * s.critical_gap_penalty, s.critical_gap_penalty_max)
    # seniority above the candidate's level, and required years beyond the candidate's experience
    seniority_pen = s.seniority_penalty.get(level, 0.0)
    years_required = years_required if years_required is not None else analysis.get("years_experience_required")
    years_pen = 0.0
    if years_required and candidate_years is not None:
        gap = float(years_required) - float(candidate_years) - s.years_gap_tolerance
        years_pen = min(max(0.0, gap) * s.years_gap_penalty_per_year, s.years_gap_penalty_max)
    score = round(_clamp(base + bonus - penalty - seniority_pen - years_pen), 2)

    if score >= s.thresholds.get("priority", 8.0):
        rec = Recommendation.PRIORITY
    elif score >= s.thresholds.get("review", 6.5):
        rec = Recommendation.REVIEW
    else:
        rec = Recommendation.LOW_PRIORITY
    breakdown = {
        "components": {k: round(v, 2) for k, v in components.items()},
        "weights": dict(s.weights),
        "base": round(base, 2),
        "target_bonus": bonus,
        "gap_penalty": round(penalty, 2),
        "seniority": level,
        "seniority_penalty": seniority_pen,
        "years_required": years_required,
        "years_penalty": round(years_pen, 2),
        "llm_overall": analysis.get("overall_score"),
    }
    return ScoreResult(score, rec, breakdown)
