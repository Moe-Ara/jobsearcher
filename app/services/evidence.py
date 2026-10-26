"""Evidence-level guard for generated letters (deterministic, independent of any LLM).

A skill listed only under `academic` / `project` in profile.yaml must not be presented as paid professional
experience. The LLM fact check classifies claims too; this guard catches the most damaging exaggeration even
when a model misses it, and reconciles the LLM's verdict with the level the letter actually claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config_files import CandidateProfile

# Sentence asserts PAID professional experience (German + English).
_PRO_MARKERS = re.compile(
    r"beruflich\w*|berufserfahrung|berufspraxis|mehrjährig\w*|langjährig\w*|jahrelang\w*|\d+\s*jahre\w*\s+(\w+\s+){0,3}erfahrung"
    r"|in meiner (tätigkeit|position|rolle|arbeit|anstellung)\b|professionell\w*"
    r"|als [\w+#./-]*(entwickler|ingenieur|engineer|developer|programmierer)\w*\s+(\w+\s+){0,3}(tätig|gearbeitet|beschäftigt|angestellt)"
    r"|\bprofessional(ly)?\b|years of (professional |industry |hands-on )?(experience|work)|work experience|industry experience"
    r"|in my (role|job|position) (at|as)|worked as an? |commercial(ly)? experience",
    re.IGNORECASE,
)
# Sentence clearly frames the experience as study / own projects → not a professional claim.
_NON_PRO_FRAMING = re.compile(
    r"studium|masterarbeit|bachelorarbeit|abschlussarbeit|hochschule|universität|eigenen? projekt\w*|privat\w*|hobby"
    r"|\bstud(y|ies|ying)\b|\bthesis\b|universit|personal projects?|side projects?|own projects?",
    re.IGNORECASE,
)
# Generic AI wording maps to the profile's AI-related skills.
_AI_TERMS = re.compile(r"\bKI\b|künstliche[nr]? intelligenz|\bAI\b|artificial intelligence|machine learning|maschinelle[sn]? lernen|\bML\b|\bLLMs?\b|deep learning", re.IGNORECASE)
_AI_SKILL = re.compile(r"machine learning|\bllm|\bai\b|intelligen|fine-tuning|classification|supervised|computer vision|perception|agentic|prompt|autonom", re.IGNORECASE)


@dataclass
class LevelViolation:
    sentence: str
    skill: str
    levels: list[str]

    def as_claim(self) -> dict[str, str]:
        levels = " and ".join(_LEVEL_NAMES.get(level, level) for level in self.levels) or "no"
        return {
            "sentence": self.sentence, "claim": f"professional experience with {self.skill}",
            "claimed_level": "professional", "status": "UNSUPPORTED", "profile_evidence": "",
            "note": f"Evidence-level exaggeration: {self.skill} is only backed by {levels} experience in profile.yaml. "
                    f"Rephrase as {levels} experience.",
            "source": "evidence-guard",
        }


_LEVEL_NAMES = {"direct_professional": "professional", "academic": "academic", "project": "project"}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _skill_regex(skill: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w+#]){re.escape(skill)}(?![\w+#])", re.IGNORECASE)


def find_level_violations(letter: str, profile: CandidateProfile) -> list[LevelViolation]:
    levels = profile.skills.levels()
    original = {}
    for name, _ in profile.skills.all():
        original.setdefault(name.strip().lower(), name.strip())
    professional_text = profile.professional_corpus()
    employers = [e.employer for e in profile.experience if e.employer]
    ai_levels = set().union(*(lv for s, lv in levels.items() if _AI_SKILL.search(s))) if levels else set()

    def is_professional(skill: str, lv: set[str]) -> bool:
        return "direct_professional" in lv or _skill_regex(skill).search(professional_text) is not None

    out: list[LevelViolation] = []
    for sentence in split_sentences(letter):
        employer_context = any(_skill_regex(e).search(sentence) for e in employers)
        if not (_PRO_MARKERS.search(sentence) or employer_context) or (_NON_PRO_FRAMING.search(sentence) and not employer_context):
            continue
        flagged = set()
        for skill, lv in levels.items():
            if len(skill) < 2 or skill in flagged:  # single letters ("C") are too ambiguous to match safely
                continue
            if _skill_regex(skill).search(sentence) and not is_professional(skill, lv):
                out.append(LevelViolation(sentence, original.get(skill, skill), sorted(lv)))
                flagged.add(skill)
        if not flagged and _AI_TERMS.search(sentence) and ai_levels and "direct_professional" not in ai_levels \
                and not _AI_SKILL.search(professional_text):
            out.append(LevelViolation(sentence, "AI / machine learning", sorted(ai_levels)))
    return out


# Named technologies a letter might borrow from the vacancy. Any of these mentioned as the applicant's own
# experience must exist somewhere in profile.yaml (terms the profile contains are skipped automatically).
TECH_TERMS = [
    "JUnit", "Mockito", "TestNG", "Selenium", "Cypress", "Playwright", "Cucumber", "Kubernetes", "K8s", "OpenShift",
    "Helm", "Terraform", "Ansible", "Puppet", "Chef", "Azure", "AWS", "GCP", "Google Cloud", "Kafka", "RabbitMQ",
    "MQTT", "GraphQL", "gRPC", "MongoDB", "Cassandra", "MySQL", "MariaDB", "Oracle", "NoSQL", "Elasticsearch",
    "OpenSearch", "InfluxDB", "TimescaleDB", "Grafana", "Prometheus", "Kibana", "Datadog", "Podman", "Spark",
    "Hadoop", "Airflow", "Kubeflow", "MLflow", "PyTorch", "TensorFlow", "Keras", "scikit-learn", "NumPy", "React",
    "Angular", "Vue", "Node.js", "Go", "Golang", "Rust", "Kotlin", "Scala", "Ruby", "PHP", "Swift", "Spring Boot",
    "Quarkus", "Hibernate", "Maven", "Gradle", "GitLab", "GitHub Actions", "Bitbucket", "Jira", "SAP", "ABAP",
    "OpenCV", "CUDA", "Unreal", "Omniverse", "Isaac Sim", "MATLAB", "Simulink", "LabVIEW", "CANoe", "AUTOSAR",
    "Django", "FastAPI", "Splunk", "QRadar", "LangChain", "Hugging Face", "Snowflake", "Databricks", "dbt",
    "Power BI", "Tableau", "Salesforce", "ServiceNow", "Microservices", "Microservice", "Kubernetes-Operator", "FaaS",
    "Serverless",
    # generic areas models like to borrow from vacancies
    "Cloud", "Cloud-native", "IoT", "Big Data", "Edge Computing", "Data Warehouse", "Data Lake", "Event Streaming",
]
_LEARNING = re.compile(
    r"einarbeit\w*|lern\w*|vertief\w*|kennenlernen|interessier\w*|würde|möchte|gern\w*\s+\w*\s*(mit|in)|reizt|spannend"
    r"|\blearn\w*|interested|would like|eager|keen to|look forward", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(ich|mein\w*|mir|mich|I|my|me)\b", re.IGNORECASE)
# A transferable-skill bridge that explicitly disclaims the target technology ("Mit X habe ich noch nicht gearbeitet,
# aber …", "I haven't used X yet, but …") is honest — it must not be removed as an unknown-technology claim.
_DISCLAIMED = re.compile(
    r"\b(noch nicht|bisher nicht|zwar nicht|nicht direkt|nicht selbst|keine (direkte|praktische|eigene) erfahrung|"
    r"not yet|haven't|have not|hasn't|no (direct|hands-on|practical) experience|not directly|without direct)\b",
    re.IGNORECASE)


def find_unknown_technologies(letter: str, profile: CandidateProfile) -> list[dict[str, str]]:
    """Named technologies claimed as own experience that profile.yaml does not contain at all."""
    corpus = profile.evidence_corpus()
    unknown = [t for t in TECH_TERMS if t.lower() not in corpus]
    out = []
    for sentence in split_sentences(letter):
        if not _FIRST_PERSON.search(sentence) or _LEARNING.search(sentence) or _DISCLAIMED.search(sentence):
            continue
        hits = [t for t in unknown if _skill_regex(t).search(sentence)]
        if hits:
            out.append({
                "sentence": sentence, "claim": f"experience with {', '.join(hits)}", "claimed_level": "professional",
                "status": "UNSUPPORTED", "profile_evidence": "",
                "note": f"{', '.join(hits)} not in profile.yaml (likely copied from the vacancy). Remove it, or phrase it "
                        "as willingness to learn.",
                "source": "evidence-guard",
            })
    return out


# Personal circumstances that sample letters love (notice period, start date, salary, why I'm leaving). They are facts
# about the applicant: only allowed when profile.yaml states something on the topic.
_CIRCUMSTANCES = {
    "notice period": (r"kündigungsfrist|notice period|weeks'? notice|months'? notice",
                      r"kündigungsfrist|notice"),
    "start date / availability": (r"eintrittstermin|frühestmöglich|(ab sofort|sofort|unmittelbar|jederzeit) (zur verfügung|verfügbar)"
                                  r"|verfügbar ab|start date|available (from|immediately|to start)"
                                  r"|can start (on|from|immediately)",
                                  r"eintritt|verfügbar|available|availability|start date"),
    "salary": (r"gehalt|vergütungsvorstellung|jahresbrutto|salary|compensation expectation",
               r"gehalt|salary|compensation"),
    "reason for leaving / current job situation": (
        r"keine (aufstiegs|entwicklungs|weiterentwicklungs)möglichkeit|keine möglichkeit(en)? (der|zur) (beruflichen )?weiterentwicklung"
        r"|in ungekündigter stellung|ungekündigt|no (room|opportunit\w*) for (growth|advancement)|looking for a new employer",
        r"ungekündigt|reason for leaving|wechselgrund"),
}


def find_personal_circumstances(letter: str, profile: CandidateProfile) -> list[dict[str, str]]:
    """Notice period, start date, salary or reasons for leaving that profile.yaml does not mention (template residue)."""
    profile_text = profile.as_prompt_text().lower()
    out = []
    for sentence in split_sentences(letter):
        for topic, (pattern, profile_keys) in _CIRCUMSTANCES.items():
            if re.search(pattern, sentence, re.IGNORECASE) and not re.search(profile_keys, profile_text):
                out.append({
                    "sentence": sentence, "claim": topic, "claimed_level": "general", "status": "UNSUPPORTED",
                    "profile_evidence": "",
                    "note": f"{topic} is not in profile.yaml (typical sample-letter content). Add it to the profile if true.",
                    "source": "evidence-guard",
                })
                break
    return out


def reconcile_claim(claim: dict) -> dict:
    """Make the LLM verdict consistent with the level the letter claims (professional claims need professional support)."""
    claimed, status = claim.get("claimed_level"), claim.get("status")
    if claimed == "professional" and status in ("SUPPORTED_ACADEMIC", "SUPPORTED_PROJECT", "SUPPORTED_GENERAL"):
        level = status.removeprefix("SUPPORTED_").lower()
        return {**claim, "status": "UNSUPPORTED",
                "note": (claim.get("note") or "") + f" [Evidence-level exaggeration: claimed professional, profile supports {level} only.]"}
    if claimed in ("academic", "project") and status == "SUPPORTED_GENERAL":
        return {**claim, "status": "AMBIGUOUS"}
    return claim
