"""Side jobs: part-time, mini-jobs and temporary work of any kind, near home or remote (preferences.yaml → side_jobs).

They live next to the career search (job.track == "side"): own filters — any occupation, but no student-only jobs
(unless your profile says enrolled_student) and nothing that needs a licence you don't have; the defence and untrusted-source rules still
apply — and an own 0-10 ranking without any LLM call:

  base 3.5 · drive time from home (≤15 min +2, ≤25 +1.5, ≤40 +1, ≤60 +0.5) · fully remote +1.5, some home office
  +0.5 · fits you: IT / software / data / AI work +3, related (tech, support, teaching, research, languages) +1.5 ·
  open to career changers +0.5 · pay stated +0.5 · flexible part-time slots +0.5 · temp agency −1
"""

from __future__ import annotations

import re
from typing import Any

from app.config_files import Preferences
from app.enums import FilterStatus, Recommendation
from app.filters import FilterOutcome, defence_check, untrusted_source_check


# "abgeschlossene Ausbildung als / zum / zur …": the job expects a trade qualification (ranked lower, not hidden)
_TRAINED_TRADE = re.compile(r"abgeschlossene[nr]?\s+(berufs)?ausbildung\s+(als|zum|zur|im|in)\b|ausgebildete[rn]?\s+\w+", re.I)


def side_filters(job, prefs: Preferences) -> FilterOutcome:
    out = FilterOutcome()
    from app.config_files import get_profile

    sp = prefs.side_jobs
    title = job.title or ""
    patterns = [*sp.excluded_title_patterns, *([] if get_profile().enrolled_student else sp.student_title_patterns)]
    for pattern in patterns:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            out.reject("side_student_only", f"Student, school or training format (“{m.group(0)}”) — not a side job for you")
            return out
    for pattern in sp.excluded_profession_patterns:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            out.reject("side_profession", f"Regulated profession / trade (“{m.group(0)}”) — needs a qualification you don't have")
            return out
    text = "\n".join(x for x in (title, job.description, job.requirements) if x)
    for pattern in sp.excluded_requirement_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            out.reject("side_requirement", f"Needs “{m.group(0)}”")
            return out
    if (job.raw_source_data or {}).get("nationwide_not_remote"):
        out.reject("side_location", "Found by a nationwide remote search, but the posting offers no home office")
        return out
    defence_check(job, prefs, out)
    if not out.rejected:
        untrusted_source_check(job, prefs, out)
    minutes = (job.raw_source_data or {}).get("drive_minutes")
    if not out.rejected and minutes is not None and minutes > sp.max_drive_minutes and (job.remote_status or "") != "remote":
        out.reject("side_distance", f"{minutes:.0f} min by car — more than {sp.max_drive_minutes} min from home")
    if not out.rejected and job.country and job.country != "Germany" and (job.remote_status or "") != "remote":
        out.reject("side_location", f"Side jobs are kept near home or remote (this one is in {job.country})")
    return out


def side_score(job, prefs: Preferences) -> tuple[float, dict[str, Any]]:
    sp = prefs.side_jobs
    raw = job.raw_source_data or {}
    parts: dict[str, float] = {"base": 3.5}
    minutes, distance = raw.get("drive_minutes"), raw.get("distance_km")
    if isinstance(minutes, (int, float)):
        parts["drive_time"] = (2.0 if minutes <= 15 else 1.5 if minutes <= 25 else 1.0 if minutes <= 40
                               else 0.5 if minutes <= sp.max_drive_minutes else -1.0)
    elif isinstance(distance, (int, float)):  # no coordinates: straight-line distance as a rough stand-in
        parts["distance"] = 2.0 if distance <= 8 else 1.0 if distance <= 25 else 0.0 if distance <= 50 else -1.0
    if (job.remote_status or "") == "remote":
        parts["remote"] = 1.5
    elif raw.get("homeoffice_partly") or (job.remote_status or "") == "hybrid":
        parts["some_home_office"] = 0.5
    # title and occupation say what the work is; the description is only a weak signal (company boilerplate)
    head = f" {job.title or ''} {raw.get('occupation') or ''} ".lower().replace("-", " ")
    body = f" {(job.description or '')[:1200]} ".lower().replace("-", " ")
    strong = [k for k in sp.strong_fit_keywords if re.search(rf"(?<![a-zäöü]){re.escape(k)}", head)]
    related = [k for k in sp.good_fit_keywords if re.search(rf"(?<![a-zäöü]){re.escape(k)}", head)]
    hits = strong + [k for k in related if k not in strong]
    if strong:
        parts["fits_you"] = 3.0
    elif related or any(re.search(rf"(?<![a-zäöü]){re.escape(k)}", body) for k in sp.strong_fit_keywords):
        parts["fits_you"] = 1.5
    if raw.get("career_changers"):
        parts["career_changers"] = 0.5
    if raw.get("pay_stated"):
        parts["pay_stated"] = 0.5
    if raw.get("flexible_slots"):
        parts["flexible_hours"] = 0.5
    if raw.get("temp_agency"):
        parts["temp_agency"] = -1.0
    if _TRAINED_TRADE.search(job.description or "") and not raw.get("career_changers"):
        parts["needs_vocational_training"] = -1.5
    score = round(max(0.0, min(10.0, sum(parts.values()))), 1)
    return score, {"side": True, "parts": parts, "fit_words": hits[:6]}


def apply_side(job, prefs: Preferences) -> None:
    """Filters + score for a side job (no LLM analysis — the analysis queue never picks these up)."""
    from app.enums import AnalysisStatus

    out = side_filters(job, prefs)
    job.filter_status = out.status
    job.filter_reasons = out.reasons
    job.analysis_status = AnalysisStatus.SKIPPED
    score, breakdown = side_score(job, prefs)
    job.fit_score = score if out.status != FilterStatus.REJECTED else min(score, 2.0)
    job.score_breakdown = breakdown
    job.recommendation = (Recommendation.PRIORITY if score >= 8 else Recommendation.REVIEW if score >= 6
                          else Recommendation.LOW_PRIORITY)
