"""Personal Application Notes: the applicant's own reasons for one application.

The principle: the AI provides the German and the structure; the applicant provides the motivation, the
priorities and the personal reasoning. Notes are informal, any language (English, German, Arabic, mixed).

They steer MOTIVATION and EMPHASIS only. They are never evidence: a note like "I know Kubernetes pretty well"
does not let a letter claim Kubernetes — skills and employment history come only from profile.yaml, and the
factual validation never sees the notes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.enums import CoverLetterStrategy
from app.llm.prompts import neutralise
from app.models import ApplicationNotes

NOTE_FIELDS = ("why_interested", "why_fit", "emphasize", "avoid")
MAX_NOTE_CHARS = 2000

# Labels shown to the user (UI) and to the models (prompt block).
FIELD_LABELS = {
    "why_interested": "Why does this job genuinely interest you?",
    "why_fit": "Why do you think you fit?",
    "emphasize": "What should the letter emphasize?",
    "avoid": "Anything the letter should avoid or be careful about?",
}


@dataclass
class PersonalNotes:
    why_interested: str = ""
    why_fit: str = ""
    emphasize: str = ""
    avoid: str = ""

    @property
    def is_empty(self) -> bool:
        return not any(getattr(self, f) for f in NOTE_FIELDS)

    @property
    def has_motivation(self) -> bool:
        """The two fields the UI asks for before generating."""
        return bool(self.why_interested or self.why_fit)

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def clean_note(text: str | None) -> str:
    """Keep the applicant's words (any script), drop control characters and runaway length."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:MAX_NOTE_CHARS]


def get_row(session: Session, job_id: int) -> ApplicationNotes | None:
    return session.scalars(select(ApplicationNotes).where(ApplicationNotes.job_id == job_id)).first()


def get_notes(session: Session, job_id: int) -> PersonalNotes:
    row = get_row(session, job_id)
    if row is None:
        return PersonalNotes()
    return PersonalNotes(**{f: getattr(row, f) or "" for f in NOTE_FIELDS})


def _row(session: Session, job_id: int) -> ApplicationNotes:
    row = get_row(session, job_id)
    if row is None:
        row = ApplicationNotes(job_id=job_id, cover_letter_strategy=CoverLetterStrategy.NONE)
        session.add(row)
    return row


def save_notes(session: Session, job_id: int, **fields: str | None) -> PersonalNotes:
    """Upsert the notes; the stored strategy is left alone (it describes letters, not notes)."""
    row = _row(session, job_id)
    for name in NOTE_FIELDS:
        if name in fields:
            setattr(row, name, clean_note(fields[name]) or None)
    session.flush()
    return get_notes(session, job_id)


def get_strategy(session: Session, job_id: int) -> CoverLetterStrategy:
    row = get_row(session, job_id)
    return CoverLetterStrategy(row.cover_letter_strategy) if row else CoverLetterStrategy.NONE


def set_strategy(session: Session, job_id: int, strategy: CoverLetterStrategy | str) -> None:
    _row(session, job_id).cover_letter_strategy = CoverLetterStrategy(strategy)
    session.flush()


def strategy_for(notes: PersonalNotes) -> CoverLetterStrategy:
    return CoverLetterStrategy.AI_WITHOUT_PERSONAL_INPUT if notes.is_empty else CoverLetterStrategy.PERSONAL_INPUT_AI_EDITED


def record_generation(session: Session, job_id: int, notes: PersonalNotes) -> CoverLetterStrategy:
    """After a generation run: remember how this application's letter came about (a manual choice wins)."""
    row = _row(session, job_id)
    if row.cover_letter_strategy != CoverLetterStrategy.MANUAL:
        row.cover_letter_strategy = strategy_for(notes)
    session.flush()
    return CoverLetterStrategy(row.cover_letter_strategy)


def notes_block(notes: PersonalNotes | None) -> str:
    """Delimited prompt block. The applicant's words, not instructions — and never evidence."""
    if notes is None or notes.is_empty:
        return ""
    lines = [f"{name.upper()}: {neutralise(getattr(notes, name)) or '(none)'}" for name in NOTE_FIELDS]
    return "<PERSONAL_NOTES>\n" + "\n".join(lines) + "\n</PERSONAL_NOTES>"
