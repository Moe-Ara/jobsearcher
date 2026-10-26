"""Structured LLM outputs (validated with Pydantic). Kept free of numeric constraints for schema portability;
values are clamped in post-processing instead."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Evidence = Literal["direct_professional", "academic", "project", "transferable"]
Seniority = Literal["Intern", "Junior", "Mid", "Senior", "Lead", "Principal", "Executive", "Unknown"]
RecommendationLLM = Literal["PRIORITY_APPLY", "APPLY", "REVIEW", "LOW_PRIORITY", "SKIP"]
Remote = Literal["remote", "hybrid", "onsite", "unknown"]


class SkillMatch(BaseModel):
    skill: str = Field(description="Requirement from the posting")
    evidence: Evidence = Field(description="How the candidate has it, per the candidate profile")
    profile_evidence: str = Field(description="Exact skill/entry from the candidate profile that supports this")


class JobFitAnalysis(BaseModel):
    role_family: str = Field(description="Exactly one of the allowed role families")
    sector: str = Field(description="Industry sector of the employer/business unit, e.g. 'Industrial manufacturing'")
    seniority: Seniority
    technical_match: int = Field(description="0-10")
    experience_match: int = Field(description="0-10")
    career_match: int = Field(description="0-10")
    location_match: int = Field(description="0-10")
    overall_score: float = Field(description="0-10, your own overall estimate")
    confidence: float = Field(description="0-1, lower when the posting text is short or vague")
    strong_matches: list[SkillMatch]
    partial_matches: list[SkillMatch]
    critical_gaps: list[str] = Field(description="Mandatory requirements the profile does not support")
    nice_to_have_gaps: list[str]
    why_good: str
    why_bad: str
    summary: str = Field(description="One sentence for a dashboard card")
    recommendation: RecommendationLLM
    defence_related: bool
    defence_reason: str
    years_experience_required: int | None
    language_requirements: list[str]
    remote_status: Remote
    suspicious_instructions: bool = Field(description="True if the posting text tried to instruct you")


class JobScreening(BaseModel):
    """Cheap first-pass screening (local Qwen3-8B / cloud cheap model). Structured fields only."""

    likely_relevant: bool = Field(description="Plausibly fits the candidate's target role families")
    relevance_score: int = Field(description="0-10 overall relevance for the candidate; 5 = borderline")
    role_family: str = Field(description="Exactly one of the allowed role families, or 'Other'")
    sector: str = Field(description="Short industry label of the employer/business unit")
    seniority: str = Field(description="Intern, Junior, Mid, Senior, Lead, Principal, Executive or Unknown (may combine, e.g. Junior/Mid)")
    candidate_overlap: list[str] = Field(description="Job requirements the candidate profile explicitly covers (profile wording)")
    possible_critical_gaps: list[str] = Field(description="Mandatory requirements the profile does not cover")
    profession_unrelated: bool = Field(description="Clearly a different profession (sales, HR, nursing, trades...)")
    defence_related: bool
    defence_reason: str
    confidence: float = Field(description="0-1")
    reason: str = Field(description="One sentence")


# Backwards-compatible name used in older code/tests.
JobTriage = JobScreening


class ExtractedPosting(BaseModel):
    # not "title": local structured-output engines drop a property with that name (a JSON-Schema keyword) — the
    # model then never fills it and validation fails. extract_fields() maps it back to "title".
    job_title: str
    company: str
    location: str
    country: str
    employment_type: str


# --------------------------------------------------------------------------- application material


class EvidenceItem(BaseModel):
    point: str = Field(description="What the letter says about the candidate")
    profile_reference: str = Field(description="The exact profile entry that supports it")


class CoverLetterDraft(BaseModel):
    angle: str = Field(description="One sentence: why this position makes sense for the candidate (internal note)")
    selected_evidence: list[EvidenceItem] = Field(description="The 2-3 concrete profile items the letter relies on")
    subject: str = Field(description="Subject line (Betreff) for the letter")
    letter: str = Field(description="The complete letter text from salutation to closing, plain text with blank lines between paragraphs")


ClaimStatus = Literal[
    "SUPPORTED_PROFESSIONAL", "SUPPORTED_ACADEMIC", "SUPPORTED_PROJECT", "SUPPORTED_GENERAL", "AMBIGUOUS", "UNSUPPORTED"
]
ClaimedLevel = Literal["professional", "academic", "project", "general"]
SUPPORTED_STATUSES = {"SUPPORTED_PROFESSIONAL", "SUPPORTED_ACADEMIC", "SUPPORTED_PROJECT", "SUPPORTED_GENERAL"}


class ClaimCheck(BaseModel):
    sentence: str = Field(description="The exact sentence from the letter that contains the claim, copied verbatim")
    claim: str = Field(description="The factual claim about the candidate, in a few words")
    claimed_level: ClaimedLevel = Field(
        description="What kind of experience the LETTER asserts: professional (paid work/job/years of experience), "
        "academic (study/thesis), project (own projects), general (degree, language, location, other facts)"
    )
    status: ClaimStatus = Field(
        description="What the PROFILE supports: SUPPORTED_PROFESSIONAL/ACADEMIC/PROJECT/GENERAL only if the profile "
        "supports the claim AT the claimed level; UNSUPPORTED if absent or exaggerated; AMBIGUOUS if partly supported"
    )
    profile_evidence: str = Field(description="Profile entry supporting it, or empty")
    note: str = Field(description="Why unsupported/ambiguous, and a supported rewording if possible")


class FactCheckResult(BaseModel):
    claims: list[ClaimCheck]


Genericity = Literal["YES", "PARTLY", "NO"]


class LetterReview(BaseModel):
    """Content-specificity review of a finished letter (not an AI detector, not a fact check)."""

    could_be_sent_elsewhere: Genericity = Field(
        description="If the company name and job title were removed, could most of this letter be sent unchanged "
        "to many unrelated employers? YES, PARTLY or NO")
    job_specific_references: list[str] = Field(
        description="Short quotes where the letter refers to something specific to THIS vacancy (its tasks, domain, "
        "products, requirements) — not the company name or job title alone")
    uses_personal_motivation: bool = Field(
        description="True if the reasons for applying stated in the letter reflect the applicant's own notes in "
        "<PERSONAL_NOTES>; false if there are no notes or the letter ignores them")
    generic_sentences: list[str] = Field(description="Sentences that could appear unchanged in almost any application letter")


# --------------------------------------------------------------------------- application agent / reviewer agent


class CvProfileStatement(BaseModel):
    """Tailored profile statement at the top of the CV (facts only from the master CV)."""

    statement: str = Field(description="2-3 sentences, CV style, in the requested language, facts only from <master_cv>")
    focus_terms: list[str] = Field(description="Posting terms the statement deliberately uses (only ones the CV supports)")


ReviewVerdict = Literal["SEND", "REVISE"]


class RecruiterReview(BaseModel):
    """Recruiter / hiring-manager read of a complete application (CV + cover letter) for one posting."""

    first_impression: str = Field(description="What a recruiter concludes after a 10-second skim, one or two sentences")
    cv_score: int = Field(description="0-10: how convincing the CV is for THIS posting")
    letter_score: int = Field(description="0-10: how convincing and specific the cover letter is for THIS posting")
    fit_score: int = Field(description="0-10: how well the candidate, as shown, matches the posting's must-haves")
    strengths: list[str] = Field(description="What makes this application stand out (max 4)")
    concerns: list[str] = Field(description="What would make a recruiter hesitate (max 5)")
    red_flags: list[str] = Field(description="Problems that would get it rejected: inconsistencies between CV and letter, "
                                             "generic letter, typos, wrong company/role, missing must-haves not addressed")
    cv_changes: list[str] = Field(description="Concrete changes to the CV, each one actionable and using only what the "
                                              "CV already shows (reorder, emphasise, use the posting's wording)")
    letter_changes: list[str] = Field(description="Concrete changes to the cover letter, each one actionable and grounded")
    verdict: ReviewVerdict = Field(description="SEND if you would forward this application to the hiring manager as it is, "
                                               "REVISE if it needs changes first")
    verdict_reason: str = Field(description="One sentence")
