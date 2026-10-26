"""Free, offline keyword analyzer used when no LLM is configured (or as a fallback).

Produces the same JobFitAnalysis shape with low confidence, so the rest of the pipeline is identical.
"""

from __future__ import annotations

import re

from app.config_files import CandidateProfile, Preferences
from app.core.normalize import infer_seniority
from app.llm.schemas import JobFitAnalysis, JobTriage, SkillMatch

HEURISTIC_MODEL = "heuristic-v1"

# Extra search terms for profile skills whose wording rarely appears verbatim in postings.
_SKILL_ALIASES: dict[str, list[str]] = {
    "c++": [r"c\+\+", r"\bcpp\b"],
    "ci/cd": [r"ci/cd", r"\bci\b", r"continuous integration", r"continuous delivery", r"github actions", r"gitlab ci"],
    "backend engineering": [r"back[- ]?end", r"rest(ful)? apis?", r"microservices?"],
    "infrastructure / devops": [r"devops", r"kubernetes", r"terraform", r"infrastructure as code", r"\bsre\b"],
    "data visualization": [r"data visuali[sz]ation", r"grafana", r"power bi", r"tableau", r"plotly"],
    "machine learning": [r"machine learning", r"\bml\b", r"deep learning", r"pytorch", r"tensorflow", r"scikit"],
    "llm integration": [r"\bllms?\b", r"large language model", r"genai", r"generative ai", r"openai", r"\brag\b", r"langchain"],
    "llm-based planning": [r"agentic", r"ai agents?", r"llm[- ]based planning", r"multi[- ]agent"],
    "fine-tuning open-source llms": [r"fine[- ]?tun\w* (llms?|models?)", r"hugging ?face", r"\blora\b", r"open[- ]source (llms?|models?)"],
    "classifiers / supervised learning": [r"classifiers?\b", r"classification models?", r"supervised learning"],
    "autonomous systems": [r"autonomous (systems?|driving|vehicles?|robots?|mobile)", r"self[- ]driving"],
    "robotics": [r"robot", r"\bros2?\b"],
    "security automation (siem alert response)": [r"\bsiem\b", r"\bsoar\b", r"security automation", r"\bsoc\b"],
    "prompt engineering": [r"prompt engineering", r"prompting"],
    "unity": [r"\bunity\b"],
    "game development": [r"game dev", r"gaming", r"game engine"],
    "docker": [r"docker", r"containeri[sz]", r"containers?\b"],
    "jenkins": [r"jenkins"],
    "python": [r"python"],
    "java": [r"\bjava\b"],
}


def _patterns_for(skill: str) -> list[str]:
    key = skill.lower()
    if key in _SKILL_ALIASES:
        return _SKILL_ALIASES[key]
    return [re.escape(key)]


def _role_family(text: str, prefs: Preferences) -> str:
    best, best_hits = "Other", 0
    for family, keywords in prefs.role_family_keywords.items():
        hits = sum(1 for kw in keywords if kw.lower() in text)
        if hits > best_hits:
            best, best_hits = family, hits
    if best == "Other" and re.search(r"software (engineer|developer)|entwickler", text):
        return "Backend Software Engineering"
    return best


def _sector(text: str, prefs: Preferences) -> str:
    for key in sorted(prefs.scoring.sector_scores, key=len, reverse=True):
        if len(key) > 3 and key in text:
            return key.title()
    return "Unknown"


def analyze(job, profile: CandidateProfile, prefs: Preferences) -> JobFitAnalysis:
    text = "\n".join(x for x in (job.title, job.description, job.requirements, job.responsibilities) if x).lower()
    matches: list[SkillMatch] = []
    for skill, evidence in profile.skills.all():
        if any(re.search(p, text) for p in _patterns_for(skill)):
            matches.append(SkillMatch(skill=skill, evidence=evidence, profile_evidence=skill))
    n = len(matches)
    n_prof = sum(1 for m in matches if m.evidence == "direct_professional")
    # professional evidence counts more than academic/project evidence
    technical = min(10, round(1 + 1.0 * n_prof + 0.7 * (n - n_prof))) if text else 3
    seniority = job.seniority if job.seniority in {"Intern", "Junior", "Mid", "Senior", "Lead", "Principal", "Executive"} else (infer_seniority(job.title) or "Unknown")
    # Without an LLM, experience/career fit can't really be judged: never let them outrun the technical overlap.
    experience = min({"Junior": 7, "Mid": 7, "Senior": 5, "Lead": 4, "Principal": 2, "Executive": 1, "Intern": 2}.get(seniority, 6), technical + 3)
    family = _role_family(text, prefs) if n else "Other"
    career = min(round(prefs.scoring.role_family_scores.get(family, 3) * 0.8), technical + 2)
    return JobFitAnalysis(
        role_family=family,
        sector=_sector(text, prefs),
        seniority=seniority,
        technical_match=technical,
        experience_match=experience,
        career_match=career,
        location_match=7,
        overall_score=round((technical + experience + career) / 3, 1),
        confidence=0.3 if len(text) > 500 else 0.15,
        strong_matches=[m for m in matches if m.evidence == "direct_professional"],
        partial_matches=[m for m in matches if m.evidence != "direct_professional"],
        critical_gaps=[],
        nice_to_have_gaps=[],
        why_good=f"Keyword overlap with {n} profile skills." if n else "",
        why_bad="Heuristic estimate only — configure an LLM for a real analysis.",
        summary=f"Keyword match: {', '.join(m.skill for m in matches[:5]) or 'no profile skills found'} (heuristic, no LLM).",
        recommendation="REVIEW" if n >= 4 else "LOW_PRIORITY",
        defence_related=False,
        defence_reason="",
        years_experience_required=int(job.years_experience_required) if job.years_experience_required else None,
        language_requirements=list(job.language_requirements or []),
        remote_status=job.remote_status or "unknown",
        suspicious_instructions=False,
    )


def triage(job, prefs: Preferences) -> JobTriage:
    text = f"{job.title}\n{job.description or ''}".lower()
    return JobTriage(
        relevant=True, profession_unrelated=False, defence_related=False, defence_reason="",
        role_family=_role_family(text, prefs), confidence=0.2, reason="heuristic",
    )
