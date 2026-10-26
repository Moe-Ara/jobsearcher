"""What the application agent learns from your feedback.

When you tell the agent what to change in an application ("too long", "don't mention the Unity project", "lead with
the robotics work"), one model call turns it into:
  - changes for THIS application: letter changes, CV statement changes, CV items to leave out or to emphasise
  - notes for THIS job (saved to the job's personal notes: emphasise / avoid)
  - general rules for EVERY future application ("Keep letters under 300 words", "Never list Unity on my CV") —
    saved in config/agent_memory.yaml and given to the writer, the CV agent and the reviewer from then on.

Rules are preferences about style, emphasis and selection. They never add facts: facts still come only from
profile.yaml / cv.yaml, and the fact check still applies. You can edit or delete every rule (More → Agent memory).
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

FEEDBACK_VERSION = "feedback-plan-v2"
Scope = Literal["letter", "cv", "all"]


class LearnedRule(BaseModel):
    id: int
    text: str
    applies_to: Scope = "all"
    cv_exclude: list[str] = Field(default_factory=list)  # terms whose CV items are always left out
    learned_from: str = ""
    added: str = ""


class GeneralRule(BaseModel):
    rule: str = Field(description="A short, general instruction that applies to EVERY future application")
    applies_to: Scope = "all"
    cv_exclude: list[str] = Field(default_factory=list, description="CV terms to always leave out (only if the rule says so)")


class FeedbackPlan(BaseModel):
    letter_changes: list[str] = Field(default_factory=list, description="Concrete changes to this cover letter")
    cv_changes: list[str] = Field(default_factory=list, description="Changes to this CV's profile statement / wording")
    cv_exclude: list[str] = Field(default_factory=list, description="Words, as they appear on the CV, of items to leave out (e.g. 'Unity')")
    cv_emphasize: list[str] = Field(default_factory=list, description="Skills / topics, as written on the CV, to put first (e.g. 'Robotik')")
    job_emphasize: str = Field("", description="What to emphasise for THIS job (kept in the job's notes)")
    job_avoid: str = Field("", description="What to avoid for THIS job (kept in the job's notes)")
    general_rules: list[GeneralRule] = Field(default_factory=list,
                                             description="Only preferences meant for ALL applications, else empty")


FEEDBACK_SYSTEM = """You turn an applicant's feedback on one job application (cover letter + CV) into instructions.

Split it into:
- letter_changes / cv_changes: what to change in THIS application, as clear instructions in English.
- cv_exclude: words identifying CV items (bullets, projects, skills) the applicant wants left OUT of this CV.
- cv_emphasize: skills or topics the applicant wants this CV to put first.
- job_emphasize / job_avoid: short notes about THIS job only.
- general_rules: ONLY what is clearly a lasting preference for all applications (style, length, tone, things to
  always / never mention). A remark about this particular company or posting is NOT a general rule. Do not repeat a
  rule the applicant already has (listed below). When in doubt, leave general_rules empty.

Never turn feedback into new facts about the applicant. The feedback may be in any language; write English."""


def feedback_user(feedback: str, job, existing: list[LearnedRule]) -> str:
    rules = "\n".join(f"- {r.text}" for r in existing) or "(none)"
    return (f"Job: {job.title} at {job.company}\n\nThe applicant's existing general rules:\n{rules}\n\n"
            f"The applicant's feedback on this application:\n<<<\n{feedback[:4000]}\n>>>")


# --------------------------------------------------------------------------- storage


def memory_path() -> Path:
    from app.settings import get_settings

    return Path(get_settings().config_dir) / "agent_memory.yaml"


_HEADER = ("# What the application agent learned from your feedback — applied to every new application\n"
           "# (letters, CVs and the reviewer's check). Edit or delete freely; the app adds rules when you give feedback\n"
           "# with \"learn from this\" ticked. applies_to: letter | cv | all. cv_exclude: CV items with these words are\n"
           "# always left out.\n\n")


def load_rules() -> list[LearnedRule]:
    try:
        data = yaml.safe_load(memory_path().read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return []
    except yaml.YAMLError as exc:
        log.warning("agent_memory.yaml can't be read: %s", exc)
        return []
    return [LearnedRule.model_validate(r) for r in data.get("rules") or [] if isinstance(r, dict) and r.get("text")]


def save_rules(rules: list[LearnedRule]) -> None:
    text = _HEADER + yaml.safe_dump({"rules": [r.model_dump() for r in rules]}, sort_keys=False, allow_unicode=True,
                                    width=110)
    path = memory_path()
    if path.exists():  # previous version kept, like every config change made in the app
        backups = path.parent / ".backups"
        backups.mkdir(exist_ok=True)
        from datetime import datetime

        (backups / f"agent_memory-{datetime.now():%Y%m%d-%H%M%S}.yaml").write_bytes(path.read_bytes())
    path.write_text(text, encoding="utf-8")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", " ", text.lower()).strip()


def add_rules(new: list[GeneralRule | dict[str, Any]], learned_from: str = "") -> list[LearnedRule]:
    """Append rules (duplicates by wording are skipped). Returns the rules actually added."""
    rules = load_rules()
    seen = {_norm(r.text) for r in rules}
    next_id = max((r.id for r in rules), default=0) + 1
    added = []
    for item in new:
        g = item if isinstance(item, GeneralRule) else GeneralRule.model_validate(item)
        text = g.rule.strip()
        if not text or _norm(text) in seen:
            continue
        rule = LearnedRule(id=next_id, text=text[:400], applies_to=g.applies_to,
                           cv_exclude=[t.strip() for t in g.cv_exclude if t.strip()][:10],
                           learned_from=learned_from[:200], added=date.today().isoformat())
        rules.append(rule)
        added.append(rule)
        seen.add(_norm(text))
        next_id += 1
    if added:
        save_rules(rules)
    return added


def delete_rule(rule_id: int) -> bool:
    rules = load_rules()
    kept = [r for r in rules if r.id != rule_id]
    if len(kept) != len(rules):
        save_rules(kept)
    return len(kept) != len(rules)


def update_rule(rule_id: int, text: str, applies_to: str) -> bool:
    rules = load_rules()
    for r in rules:
        if r.id == rule_id:
            r.text = text.strip()[:400] or r.text
            r.applies_to = applies_to if applies_to in ("letter", "cv", "all") else r.applies_to
            save_rules(rules)
            return True
    return False


# --------------------------------------------------------------------------- use in prompts


def rules_for(target: str) -> list[LearnedRule]:
    """target: letter | cv | review (the reviewer checks every rule)."""
    return [r for r in load_rules() if target == "review" or r.applies_to in (target, "all")]


def instructions_block(target: str) -> str:
    rules = rules_for(target)
    if not rules:
        return ""
    lines = "\n".join(f"- {r.text}" for r in rules)
    if target == "review":
        return ("\n\nTHE APPLICANT'S STANDING INSTRUCTIONS (learned from their earlier feedback). An application that "
                f"ignores one of them needs revising — name the rule in your reasons:\n{lines}\n")
    return ("\n\nTHE APPLICANT'S STANDING INSTRUCTIONS (learned from their earlier feedback) — follow them in this "
            f"document. They are preferences about style, emphasis and selection, never facts:\n{lines}\n")


def cv_excludes() -> list[str]:
    return [t for r in rules_for("cv") for t in r.cv_exclude]


# --------------------------------------------------------------------------- feedback → plan


def plan_feedback(session, job, feedback: str, setup) -> FeedbackPlan:
    """One model call; without a model the feedback goes to letter and CV as it is (nothing is learned)."""
    from app.llm.roles import Role
    from app.services.llm_calls import llm_call

    target = setup.target(Role.VALIDATOR) or setup.target(Role.WRITER)
    if target is None:
        return FeedbackPlan(letter_changes=[feedback], cv_changes=[feedback])
    try:
        res = llm_call(session, target, kind="feedback_plan", job_id=job.id, prompt_version=FEEDBACK_VERSION,
                       profile_hash="-", system=FEEDBACK_SYSTEM, user=feedback_user(feedback, job, load_rules()),
                       schema=FeedbackPlan, use_cache=False)
        plan = FeedbackPlan.model_validate(res.data)
    except Exception as exc:  # noqa: BLE001 - the applicant's words still reach both documents
        log.warning("Feedback could not be structured (%s) — applying it as written", exc)
        return FeedbackPlan(letter_changes=[feedback], cv_changes=[feedback])
    if not (plan.letter_changes or plan.cv_changes or plan.cv_exclude or plan.cv_emphasize):
        plan.letter_changes = [feedback]  # never drop the applicant's request
    return plan
