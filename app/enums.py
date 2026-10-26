"""Domain enums. Stored as plain strings so the schema stays portable (SQLite/PostgreSQL)."""

from enum import StrEnum


class JobStatus(StrEnum):
    NEW = "NEW"
    REVIEW = "REVIEW"
    PRIORITY = "PRIORITY"
    SKIPPED = "SKIPPED"
    APPLYING = "APPLYING"
    APPLIED = "APPLIED"
    HR_SCREEN = "HR_SCREEN"
    INTERVIEW = "INTERVIEW"
    TECHNICAL = "TECHNICAL"
    FINAL = "FINAL"
    OFFER = "OFFER"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# Ordered application funnel — used by analytics ("furthest stage reached").
FUNNEL = [
    JobStatus.APPLIED,
    JobStatus.HR_SCREEN,
    JobStatus.INTERVIEW,
    JobStatus.TECHNICAL,
    JobStatus.FINAL,
    JobStatus.OFFER,
]

# Dashboard quick actions -> status.
ACTIONS: dict[str, JobStatus] = {
    "priority": JobStatus.PRIORITY,
    "save": JobStatus.REVIEW,
    "applying": JobStatus.APPLYING,
    "applied": JobStatus.APPLIED,
    "skip": JobStatus.SKIPPED,
    "reject": JobStatus.REJECTED,
}


class FilterStatus(StrEnum):
    PENDING = "PENDING"
    PASSED = "PASSED"
    FLAGGED = "FLAGGED"  # passed, but something needs the LLM / a human to look (e.g. mixed defence company)
    REJECTED = "REJECTED"


class AnalysisStatus(StrEnum):
    PENDING = "PENDING"
    DONE = "DONE"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"  # hard-filtered, never sent to the LLM
    SCREENED = "SCREENED"  # cheap screening said poor match → low priority, no deep analysis


class Recommendation(StrEnum):
    PRIORITY = "PRIORITY"
    REVIEW = "REVIEW"
    LOW_PRIORITY = "LOW_PRIORITY"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


class SourceType(StrEnum):
    EMPLOYER = "employer"
    JOB_BOARD = "job_board"
    MANUAL = "manual"
    RECRUITER = "recruiter"
    AGGREGATOR = "aggregator"


# Lower = preferred as canonical source.
SOURCE_RANK: dict[str, int] = {
    SourceType.EMPLOYER: 1,
    SourceType.JOB_BOARD: 2,
    SourceType.MANUAL: 3,
    SourceType.RECRUITER: 4,
    SourceType.AGGREGATOR: 5,
}


class RemoteStatus(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class RelevanceLabel(StrEnum):
    """Human judgement: how good is this recommendation for me? Independent of application status."""

    UNRATED = "UNRATED"
    GREAT = "GREAT"
    MAYBE = "MAYBE"
    BAD = "BAD"


RELEVANCE_REASONS = [
    "wrong seniority",
    "wrong role",
    "wrong sector",
    "wrong location",
    "missing critical skill",
    "poor career direction",
    "too generic",
    "defence",
    "company not interesting",
    "already seen",
    "other",
]


class CoverLetterStrategy(StrEnum):
    """How the letter for an application came about — stored for later comparison with response rates."""

    NONE = "NONE"
    AI_WITHOUT_PERSONAL_INPUT = "AI_WITHOUT_PERSONAL_INPUT"
    PERSONAL_INPUT_AI_EDITED = "PERSONAL_INPUT_AI_EDITED"
    MANUAL = "MANUAL"
