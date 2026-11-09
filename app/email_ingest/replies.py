"""Replies from employers: rejections, interview invitations and receipt confirmations in your mailbox.

Every message that looks like it is about one of your applications is read once, classified by wording (German and
English) and matched to the job you applied to: the e-mail you sent (In-Reply-To / References), the company's mail
domain, or the company and job title it names. A matched rejection sets the job to Rejected, an invitation to
Interview, a receipt confirmation turns "Applying" into "Applied". Unmatched replies are listed on the Replies page,
where one click assigns them.
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup

Kind = Literal["rejection", "interview", "next_step", "received", "other"]

_W = r"[^.!?\n]{0,160}?"  # within one sentence

REJECTION = re.compile("|".join([
    # German
    r"\bleider\b" + _W + r"(nicht (weiter |mehr |näher |länger )?(berücksichtig|in (die|unsere) (engere|nähere)|anbieten|"
    r"weiterverfolg|zum zug|fortsetzen|eingeladen|für (sie|dich) entschieden|weiter(gehen|führen)|in betracht|"
    r"positiv|erfolgreich|weiter im (auswahl|bewerbungs))|absagen|absage|keine (positive|bessere|erfreulichere|andere)|"
    r"(eine[mn]?|andere[nr]?|weitere[nr]?) (anderen? )?(bewerber|kandidat)|(gegen|nicht für) (sie|dich) entschieden|"
    r"nicht (mehr )?(verfügbar|vakant|zu besetzen)|(bereits|inzwischen|anderweitig) (besetzt|vergeben)|"
    r"keine (berücksichtigung|einladung|möglichkeit))",
    r"(zugunsten|für) (eine[sn]? )?(anderen?|weiteren?) (bewerber|kandidat)\w*" + _W + r"entschieden",
    r"(uns|wir)\b" + _W + r"für (jemand(en)?|eine andere person|einen anderen|eine andere kandidat)\w*" + _W + r"entschieden",
    r"mit anderen (bewerber|kandidat)\w*" + _W + r"(fortzusetzen|fortsetzen|fortzuführen|fortführen|weiter(zu)?gehen|zu gehen)",
    r"kein(e|en)? (positive[sn]?|erfreuliche[sn]?|bessere[sn]?) (feedback|nachricht|rückmeldung|antwort|bescheid)",
    r"(für (ihren|ihre|deinen|deine|den|die) (weiteren?|zukünftigen?) (beruflichen? )?(berufsweg|weg|werdegang|lebensweg|"
    r"laufbahn|zukunft|jobsuche|stellensuche|karriere)|(ihre|deine|die) (weitere|zukünftige) berufliche zukunft)" + _W +
    r"(viel erfolg|alles gute|wünschen|das beste)",
    r"(best of luck|wish you (all the best|every success|the best|success|luck)) (in|with|for) (your|the) "
    r"(future|job search|search|career|endeavou?rs|next)",
    r"(uns|wir haben uns) (gegen (sie|dich)|für (eine[n]? )?(andere|anderen|weitere))" + _W + r"entschieden",
    r"nicht (weiter|mehr) (berücksichtigen|berücksichtigt)",
    r"nicht in die (engere|nähere) auswahl",
    r"\b(diese[rn]?|unsere[rn]?|trotz dieser|als) absage\b(?! erhalten)",
    r"\babsage (auf|zu|für|bezüglich) (ihre|deine|eure|die) bewerbung",
    r"\b(eine|die) absage (fällt|ist) uns",
    r"(bedauern|bedauerlicherweise|es tut (mir|uns) leid)" + _W + r"(keine positive|nicht (weiter |mehr )?berücksichtig|"
    r"absagen|anderen? (bewerber|kandidat)|nicht in die (engere|nähere))",
    r"(stelle|position|vakanz) (ist |wurde )?(bereits |inzwischen |zwischenzeitlich |anderweitig )+(besetzt|vergeben)",
    # English
    r"\bunfortunately\b" + _W + r"(not (be )?(moving|move|proceed|progress|able to (offer|move|proceed|consider)|selected|"
    r"successful|considered|shortlisted|taking|going)|decided|other candidates?|another candidate|won't be|will not be|"
    r"been filled|no longer|isn't a match|is not a match|cannot offer|can't offer|unable to (offer|move|proceed|progress))",
    r"(decided|chosen|chose|opted) (not )?to (move|proceed|go|pursue|continue|progress) (forward )?with (other|another|a different|"
    r"candidates|someone)",
    r"(will not|won't|not be|are not|aren't|not) (be )?(moving|move|proceeding|proceed|progressing) (forward )?with your "
    r"(application|candidacy|profile)",
    r"(regret|sorry) to (inform|let you know|tell you)",
    r"(position|role|vacancy) has (now )?been filled",
    r"not (been )?selected (for|to)",
    r"(decided|decision) (not )?to (offer|pursue) (the (role|position) )?to (another|other|a different)",
]), re.I)

INTERVIEW = re.compile("|".join([
    r"(einladen|einladung|laden (sie|dich|ihnen|dir)\b" + _W + r"\bein)" + _W +
    r"(gespräch|interview|kennenlernen|videocall|video-call|call|termin|telefonat|vor ort)",
    r"(vorstellungsgespräch|bewerbungsgespräch|kennenlerngespräch|kennenlern-call|erstgespräch|telefoninterview|"
    r"video-?interview|persönliches gespräch|ersten? (austausch|gespräch|telefonat|termin))" + _W +
    r"(termin|einladen|vereinbaren|vorschlag|vorschlagen|zeitfenster|verfügbar|zeit (für|haben)|buchen|kalender)",
    r"(gerne|würden|möchten)" + _W + r"(kennenlernen|persönlich kennen|näher kennen)",
    r"(invite|inviting|invitation|like to (schedule|set up|arrange|invite|book)|love to (chat|talk|speak|schedule)|"
    r"(schedule|book|arrange) (a|an|your) (call|interview|chat|time|meeting))" + _W +
    r"(interview|call|chat|conversation|meeting|discussion|video)",
    r"next (step|round|stage)" + _W + r"(interview|call|conversation|assessment|challenge|case study)",
    r"(calendly\.com|(?<![\w.-])cal\.com/|outlook\.office365\.com/(book|owa/calendar)|meetings\.hubspot|book a (time|slot))",
    r"(coding|technical|online) (challenge|assessment|test)" + _W + r"(invite|link|complete|start|bearbeiten)",
]), re.I)

# A calendar invitation / reminder for an interview: the subject says it.
SUBJECT_INTERVIEW = re.compile(r"^((updated|aktualisiert|erinnerung|reminder|invitation|einladung|new event|neuer termin)\s*:\s*)*"
                               r"(interview|vorstellungsgespräch|bewerbungsgespräch|kennenlerngespräch|video-?interview|"
                               r"einladung zu(m|r) (vorstellungs|bewerbungs|kennenlern|video|telefon|online|persönlichen))", re.I)

# A job board confirming the application it forwarded ("Your application was sent: …", "Deine Bewerbung ist raus").
SUBJECT_SENT = re.compile(r"application (was |has been )?(sent|submitted)|bewerbung (wurde |ist )?(verschickt|versendet|versandt|"
                          r"raus|übermittelt|gesendet)|bewerbung über indeed|application to .{3,120} at ", re.I)

# A person is working on your application and needs something: documents, an answer, a test.
NEXT_STEP = re.compile("|".join([
    r"(bewerbungsprozess|bewerbungsverfahren|auswahlprozess|auswahlverfahren|prozess) (mit (ihnen|dir) )?(gerne )?"
    r"(fortführen|fortsetzen|weiterführen|fortzuführen|fortzusetzen)",
    r"(benötigen|brauchen) (wir )?(noch )?(von (ihnen|dir) )?(folgende|die (nach)?folgenden|weitere|ein paar|zwei|drei|einige|"
    r"noch)" + _W + r"(unterlagen|angaben|informationen|dinge|dokumente|nachweise)",
    r"\b(kurze|eine|folgende) rückfrage",
    r"im nächsten schritt" + _W + r"(kennenlernen|gespräch|telefonat|meeting|test|austausch)",
    r"(bitten|bitte)" + _W + r"(erneut|nochmals|noch einmal) (zu )?(schicken|senden|einreichen|bewerben|hochladen)",
    r"(persönlichkeitstest|online-?test|testaufgabe|probeaufgabe|case study|fallstudie|coding challenge|assessment|"
    r"fragebogen|questionnaire)" + _W + r"(bitte|durchführen|ausfüllen|bearbeiten|complete|fill|link|start)",
    r"(would like|like|happy|pleased) to (move|proceed|continue|go) (forward |ahead )?with your (application|candidacy)",
    r"(we )?need (some )?(additional|further|more|a few) (information|details|documents)",
    r"please (send|provide|upload|share)" + _W + r"(documents?|certificates?|transcripts?|references?|cv|résumé|resume|"
    r"salary expectations?|availability)",
]), re.I)

RECEIVED = re.compile("|".join([
    r"eingangsbestätigung",
    r"\beingang (ihrer|deiner|der|ihre[rn]?) (online-?)?bewerbung",
    r"(vielen )?dank für (die|ihre|deine) (online-?)?bewerbung",
    r"bewerbung (wurde )?(erfolgreich )?(verschickt|versendet|versandt|gesendet|weitergeleitet)",
    r"(unterlagen|bewerbungsunterlagen) (haben wir|sind|wurden)" + _W + r"(erhalten|eingegangen|angekommen)",
    r"(haben|wir haben) (ihre|deine) (unterlagen|bewerbungsunterlagen)" + _W + r"(erhalten|bekommen)",
    r"(ihre|deine|eure) bewerbung" + _W + r"(erhalten|eingegangen|angekommen|übermittelt|ist raus|gut bei uns)",
    r"(erhalten|eingegangen|angekommen) " + _W + r"(ihre|deine) bewerbung",
    r"bewerbung (ist )?(erfolgreich )?(eingegangen|angekommen|übermittelt|raus)",
    r"(vielen )?dank für (ihre|deine|eure) bewerbung",
    r"(we('ve| have)?|we) (successfully )?received your application",
    r"thank(s| you) for (applying|your application|your interest|submitting)",
    r"application (was |has been )?(sent|submitted|received)",
    r"your application (to|for|at)\b",
    r"bewerbung über indeed",
]), re.I)

# Job-board marketing that mentions interviews or applications but is about no reply from an employer.
MARKETING = re.compile(r"(gute|deine|erhöhe deine) chancen|vergiss nicht deine bewerbung|bewerbung abschließen|"
                       r"complete your application|finish your application|passwort|password|verify|bestätige deine e-mail|"
                       r"confirm your e-mail|new device|neues gerät|job-angebot wartet|jobs? (für|for) (dich|you)|"
                       r"stellenangebote von|neue jobs|new jobs|job alert|jobagent|newsletter", re.I)


def plain_text(html: str, text: str) -> str:
    """The message as readable text: the plain part, else the HTML's text; entities decoded, blank runs collapsed."""
    use_text = bool(text and text.strip()) and (len(text.strip()) > 40 or not (html or "").strip())
    body = text if use_text else BeautifulSoup(html or "", "lxml").get_text("\n")
    body = html_lib.unescape(body).replace("­", "").replace("͏", "").replace("​", "")
    body = re.sub(r"[ \t ]+", " ", body)
    return re.sub(r"\n\s*\n+", "\n", body).strip()


def _quoted_cut(body: str) -> str:
    """Drop the quoted original below a reply (your own application text must not decide the class)."""
    m = re.search(r"^(\s*>|\s*-{2,}\s*(original|ursprüngliche)|.{0,160}\b(schrieb|wrote)\b[^\n]{0,80}:\s*$|"
                  r"\s*(von|from):\s.*\n\s*(gesendet|sent|an|to|datum|date):)", body, re.I | re.M)
    return body[:m.start()] if m and m.start() > 40 else body


# A sentence that only describes what MIGHT happen ("if we would like to invite you …", "should you not hear from us,
# we have chosen other candidates") is no decision.
_IF = r"\b(if|in case|unless|falls|sofern|sollten (wir|sie|du)|wenn (wir|sie|du|ihr)|solange|sobald|once)\b"
_HYPOTHETICAL_REJECTION = re.compile(_IF, re.I)
_HYPOTHETICAL_INTERVIEW = re.compile(_IF + r"|\b(hope|hoffen|could|might|may|will be to|whether|ob|eventuell|"
                                     r"gegebenenfalls|ggf|learn more|interview[- ]?(process|guide|prozess|tipps|tips))\b", re.I)


def _sentences(subject: str, body: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", f"{subject}.\n{body}")
    return re.split(r"(?<=[.!?])\s+", flat)


def _decides(sentences: list[str], rx: re.Pattern[str], hypothetical: re.Pattern[str]) -> bool:
    return any(rx.search(s) and not hypothetical.search(s) for s in sentences)


def classify(subject: str, body: str) -> Kind:
    """rejection > interview > next step > received > other, from the subject and the reply's own text (quotes
    removed)."""
    sentences = _sentences(subject, _quoted_cut(body)[:6000])
    if _decides(sentences, REJECTION, _HYPOTHETICAL_REJECTION):
        return "rejection"
    if MARKETING.search(subject):
        return "other"
    if SUBJECT_SENT.search(subject):
        return "received"
    if SUBJECT_INTERVIEW.search(subject.strip()) or _decides(sentences, INTERVIEW, _HYPOTHETICAL_INTERVIEW):
        return "interview"
    if _decides(sentences, NEXT_STEP, _HYPOTHETICAL_INTERVIEW):
        return "next_step"
    if any(RECEIVED.search(s) for s in sentences):
        return "received"
    return "other"


# --------------------------------------------------------------------------- matching a reply to your application

# Senders shared by many employers (applicant tracking systems, job boards, mail providers): their domain says nothing
# about the employer — the company is read from the sender's name, the address's local part or the text instead.
SHARED_DOMAINS = {
    "personio.de", "personio.com", "join.com", "softgarden.io", "softgarden.de", "myworkday.com", "workday.com",
    "successfactors.eu", "successfactors.com", "greenhouse-mail.io", "greenhouse.io", "ashbyhq.com", "recruitee.com",
    "recruitee-mailbox.com", "recruitee-email.com", "smartrecruiters.com", "onlyfy.jobs", "workablemail.com",
    "workable.com", "teamtailor-mail.com", "teamtailor.com", "dvinci-hr.com", "dvinci.de", "concludis.de", "hrworks.de",
    "icims.eu", "icims.com", "lever.co", "jobs2web.com", "rexx-systems.com", "umantis.com", "prescreen.io",
    "jetzt-bewerben.co", "recruware.com", "homerun.co", "gem.com", "bamboohr.com", "taleo.net", "oraclecloud.com",
    "eightfold.ai", "avature.net", "jobvite.com", "mssa.com", "inbound.workablemail.com", "communication.hrworks.de",
    "indeed.com", "indeedemail.com", "linkedin.com", "stepstone.de", "xing.com", "heyjobs.co", "meinestadt.de",
    "germantechjobs.de", "jobgether.com", "bayt.com", "naukrigulf.com", "gulftalent.com", "arbeitsagentur.de",
    "web.de", "gmx.de", "gmx.net", "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "yahoo.com", "t-online.de",
    "icloud.com", "posteo.de", "mailbox.org", "freenet.de",
}
_SECOND_LEVEL = {"co", "com", "org", "net", "ac", "gov", "or", "ne"}
# Words that never identify an employer on their own.
_GENERIC = set("""
gmbh ag se kg co mbh ug ev e v inc ltd llc plc sa nv bv the der die das und and of for fur von zu group gruppe holding
holdings international global germany deutschland europe european solutions solution software systems system technologies
technology tech engineering consulting consultants services service digital data smart it ai ki media partner partners
team career careers jobs job recruiting hr personal personalservice management innovation innovations industrie industries
universitat university hochschule institut institute stadt landratsamt bundesamt zentrum center centre gesellschaft
verwaltung network networks energy automotive robotics automation research labs lab studio studios company companies
""".split())

# Words that appear in ordinary e-mail text and first names: never an employer's telling word on their own.
_NOT_A_NAME = set("""
information inform informationen decision employer confidential anonymous brand anchor john michael thomas peter daniel
alexander andreas stefan martin markus christian maria anna julia sarah laura lisa alfred gustav hans karl fritz heinrich
wilhelm franz otto max open best first next better your ihre deine unser unsere our main meine mein neue new reply
noreply live update news
""".split())

_SENT = ("APPLYING", "APPLIED", "HR_SCREEN", "INTERVIEW", "TECHNICAL", "FINAL", "OFFER")
_OPEN = ("NEW", "REVIEW", "PRIORITY")
_RANK = {s: i for i, s in enumerate(_SENT)}
TARGET = {"rejection": "REJECTED", "interview": "INTERVIEW", "next_step": "HR_SCREEN", "received": "APPLIED"}
KIND_LABEL = {"rejection": "Rejection", "interview": "Interview invitation", "next_step": "Next step",
              "received": "Application received", "other": "Other"}


def norm(text: str) -> str:
    from app.core.text import strip_accents

    return " " + re.sub(r"[^a-z0-9]+", " ", strip_accents((text or "").lower())).strip() + " "


def registrable(domain: str) -> str:
    """mail.jobs.example.co.uk → example.co.uk; karriere.hnu.de → hnu.de."""
    labels = (domain or "").lower().strip(".").split(".")
    if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def is_shared(domain: str) -> bool:
    d = (domain or "").lower()
    return any(d == s or d.endswith("." + s) for s in SHARED_DOMAINS) or registrable(d) in SHARED_DOMAINS


def sender_words(from_addr: str, from_name: str) -> str:
    """The sender's name, the address's local part and the sub-domains in front of a shared system
    ("acme-robotics-jobs@m.personio.de", "anna@beispiel-health.msg.join.com") — where an ATS names the employer."""
    local, _, domain = (from_addr or "").lower().partition("@")
    labels = domain.split(".")
    reg = registrable(domain)
    subs = labels[: max(0, len(labels) - len(reg.split(".")))]
    own_reg = reg.split(".")[0] if not is_shared(domain) else ""
    words = norm(" ".join([from_name or "", local.replace(".", " "), " ".join(subs), own_reg])).split()
    return " " + " ".join(w for w in words if w not in _SENDER_NOISE) + " "


# Mailbox words in sender addresses ("no-reply@", "jobs@", "notifications@") — never part of an employer's name.
_SENDER_NOISE = set("""
no reply noreply donotreply do not nicht antworten notifications notification notify jobs job careers career karriere
recruiting recruitment recruiter bewerbung bewerbungen hr info mail email system team talent people candidates candidate
hiring support service contact kontakt personal personalabteilung office jobportal workday mailer bounce news via
""".split())


def company_keys(company: str) -> tuple[str, str]:
    """(full name without legal form, its most telling word): "NORDLICHT – Connecting People …" → ("nordlicht
    connecting people", "nordlicht")."""
    from app.core.normalize import normalize_company

    phrase = normalize_company(company)
    words = [w for w in phrase.split() if w not in _GENERIC and w not in _NOT_A_NAME and not w.isdigit()]
    acronyms = {a.lower() for a in re.findall(r"\b[A-Z][A-Z0-9&]{1,4}\b", company or "")}
    key = next((w for w in words if w in acronyms and len(w) >= 2), "")  # acronym employers: ABC, XYZ, e2x …
    if not key:
        key = max((w for w in words if len(w) >= 4), key=len, default="")  # the most telling word: "nordlicht"
    return phrase, key


def title_words(title: str) -> set[str]:
    from app.core.normalize import normalize_title

    return {w for w in normalize_title(title).split() if len(w) >= 2 and w not in _TITLE_STOP}


_TITLE_STOP = set("""
and und der die das the for fur with mit bei all genders gn von in im am an zum zur als oder or to of at stelle position
role job vacancy bewerbung application ref nr id de en
""".split())


_POSITION = re.compile(
    r"(?:bewerbung|application|bewerben|applied)\s+(?:um die stelle |auf die stelle |für die stelle |auf die position |"
    r"für die position |on the position |for the (?:position|role) of |to the (?:position|role) of )?"
    r"(?:als|as|for|auf|to)\s*:?\s*[„\"“'(]?(?P<t>[^\n„\"“”|]{4,110})"
    r"|(?:stelle|position|role|vacancy)\s+(?:als|as|of)\s+[„\"“']?(?P<u>[^\n„\"“”.|]{4,110})"
    r"|(?:bewerbung|application)\s*[-–:|]\s*(?P<v>[^\n„\"“”|]{4,110})"
    r"|(?:für die|an der|auf die|for the|in the) (?:stelle|position|role)\s+[„\"“']?(?P<w>[A-Z][^\n„\"“”|]{3,110})", re.I)


def position_words(subject: str, body: str) -> set[str]:
    """Title words of the position the e-mail names ("Ihre Bewerbung als Data Engineer (m/w/d)", "for the position of
    …"); empty when it names none. Company names after "bei"/"at" are cut off."""
    for hay in (subject, body):
        for m in _POSITION.finditer(hay or ""):
            t = re.split(r"\s+(?:bei|at|@|in unserem|im|vom|from|ist|wurde|has|was|is|erhalten|eingegangen)\b|"
                         r"\s+[-–|]\s+(?=[A-ZÄÖÜ][\w.&]+(?:\s|$))",
                         m.group("t") or m.group("u") or m.group("v") or m.group("w"))[0]
            words = title_words(t)
            if len(words) >= 2 or (len(words) == 1 and len(t) >= 8):
                return words
    return set()


@dataclass
class JobRef:
    id: int
    status: str
    company: str
    title: str
    phrase: str
    key: str
    words: set[str]
    domains: set[str]
    changed_at: object


@dataclass
class Match:
    job_id: int | None
    score: int
    why: str
    candidates: list[int]


class ReplyMatcher:
    """Jobs you applied to (and, for strong evidence only, ones you may have applied to outside the app), indexed by
    their employer's telling word for fast lookup."""

    def __init__(self, session) -> None:
        from sqlalchemy import select

        from app.models import ApplicationPackage, Job

        self.sent_ids: dict[str, int] = {}
        to_domains: dict[int, set[str]] = {}
        for pkg in session.scalars(select(ApplicationPackage).where(ApplicationPackage.meta.is_not(None))):
            sub = (pkg.meta or {}).get("submission") or {}
            if sub.get("message_id"):
                self.sent_ids[sub["message_id"].strip()] = pkg.job_id
            if "@" in (sub.get("to") or ""):
                dom = sub["to"].split("@", 1)[1].lower()
                if not is_shared(dom):
                    to_domains.setdefault(pkg.job_id, set()).add(registrable(dom))
        self.jobs: dict[int, JobRef] = {}
        self.by_key: dict[str, list[int]] = {}
        self.by_domain: dict[str, list[int]] = {}
        rows = session.execute(select(Job.id, Job.status, Job.company, Job.title, Job.canonical_url, Job.source_url,
                                      Job.status_changed_at).where(Job.status.in_(_SENT + _OPEN + ("REJECTED",))))
        for jid, status, company, title, *urls, changed in rows:
            phrase, key = company_keys(company)
            if not key and not phrase:
                continue
            domains = set(to_domains.get(jid, set()))
            for u in urls:
                host = re.sub(r"^https?://", "", u or "").split("/")[0].split(":")[0]
                if host and not is_shared(host) and "." in host:
                    domains.add(registrable(host))
            ref = JobRef(jid, status, company, title, phrase, key, title_words(title), domains, changed)
            self.jobs[jid] = ref
            if key:
                self.by_key.setdefault(key, []).append(jid)
            for d in domains:
                self.by_domain.setdefault(d, []).append(jid)
        self.domains = set(self.by_domain)

    def is_candidate(self, from_addr: str, subject: str, in_reply_to: str, references: str) -> bool:
        if set(re.findall(r"<[^>]+>", f"{in_reply_to} {references}")) & set(self.sent_ids):
            return True
        dom = (from_addr or "").rpartition("@")[2].lower()
        return bool(APPLICATION_SUBJECT.search(subject or "")) or (not is_shared(dom) and registrable(dom) in self.domains)

    def match(self, kind: str, from_addr: str, from_name: str, subject: str, body: str, in_reply_to: str = "",
              references: str = "") -> Match:
        for mid in re.findall(r"<[^>]+>", f"{in_reply_to} {references}"):
            if mid in self.sent_ids and self.sent_ids[mid] in self.jobs:
                return Match(self.sent_ids[mid], 100, "reply to the application e-mail you sent", [])
        sender = sender_words(from_addr, from_name)
        subj, text = norm(subject), norm(_quoted_cut(body)[:3000])
        words = set((sender + subj + text).split())
        dom = registrable((from_addr or "").rpartition("@")[2])
        pool = {j for w in words for j in self.by_key.get(w, [])}
        if not is_shared(dom):
            pool |= set(self.by_domain.get(dom, []))
        named = position_words(subject, _quoted_cut(body)[:1500])  # the position the e-mail is about, if it says
        here = set((subj + text).split())
        scored = []
        for jid in pool:
            ref = self.jobs[jid]
            company, why = 0, ""
            if not is_shared(dom) and dom in ref.domains:
                company, why = 70, f"sender domain {dom}"
            for where, hay, full, part in (("sender", sender, 60, 50), ("subject", subj, 50, 40), ("text", text, 35, 0)):
                long_phrase = len(ref.phrase) >= 4 and f" {ref.phrase} " in hay
                key_hit = ref.key and f" {ref.key} " in hay and (len(ref.key) >= 4 or where == "sender")
                pts = full if long_phrase else part if key_hit else 0  # in the body only the full name counts
                if pts > company:
                    company, why = pts, f"company “{ref.company}” in the {where}"
            if not company:
                continue
            if named and ref.words:  # the e-mail names a position: compare it with this job's title
                sim = len(named & ref.words) / len(named | ref.words)
                cover = len(named & ref.words) / len(ref.words)
                contradicts = sim < 0.25  # it names another position: offered to you, never linked by itself
            else:
                sim = cover = len(ref.words & here) / len(ref.words) if ref.words else 0
                contradicts = False
            score = company + (round(40 * sim) if sim >= 0.25 else 0) - (30 if contradicts else 0)
            if ref.status in _OPEN and (company < 50 or cover < 0.6):
                continue  # a job you did not apply to via the app needs the company AND the title
            scored.append((score, jid, why + (f" + title ({round(sim * 100)} %)" if sim >= 0.25 else ""), contradicts))
        scored.sort(key=lambda t: t[0], reverse=True)
        if not scored or scored[0][0] < 45:
            return Match(None, scored[0][0] if scored else 0,
                         "the position named differs from your applications" if scored and scored[0][3] else "no application matches",
                         [t[1] for t in scored[:5] if t[0] >= 15])
        best = scored[0]
        rivals = [t for t in scored[1:] if best[0] - t[0] < 10 and not self._twins(best[1], t[1])]
        if rivals:
            return Match(None, best[0], "several applications match", [best[1], *[t[1] for t in rivals[:4]]])
        return Match(best[1], best[0], best[2], [])

    def _twins(self, a: int, b: int) -> bool:
        ja, jb = self.jobs[a], self.jobs[b]
        return ja.phrase == jb.phrase and ja.words == jb.words


# A subject that may be about an application (cheap header check before a message body is fetched).
APPLICATION_SUBJECT = re.compile(
    r"bewerbung|application|applying|candidacy|candidate|kandidatur|bewerber|absage|interview|vorstellungsgespräch|"
    r"gespräch|einladung|rückmeldung|feedback|ihre unterlagen|deine unterlagen|position|\bstelle\b|recruiting|"
    r"^(aw|re|wg|fw|antw|sv)\s*:", re.I)


def own_addresses() -> set[str]:
    """Your own addresses (your sent mail and copies are never "replies")."""
    from app.settings import get_settings

    st = get_settings()
    out = set()
    mail_domain = re.sub(r"^(imap4?|mail)\.", "", (st.imap_host or "").lower())
    for user in (st.imap_user, st.smtp_user):
        user = (user or "").strip().lower()
        if "@" in user:
            out.add(user)
        elif user and mail_domain:
            out.add(f"{user}@{mail_domain}")
    try:
        from app.services.cv import get_master_cv

        mail = getattr(get_master_cv().contact, "email", None)
        if mail:
            out.add(str(mail).lower())
    except Exception:  # noqa: BLE001 - no CV yet
        pass
    return out


def process_reply(msg, matcher: ReplyMatcher, *, dry_run: bool = False) -> dict:
    """Classify one message, find its application, store it (once) and move the job along."""
    from email.utils import parseaddr, parsedate_to_datetime

    from sqlalchemy import select

    from app.db import session_scope
    from app.email_ingest.service import _bodies
    from app.models import EmailMessage, Job

    message_id = str(msg.get("Message-ID") or "").strip() or f"nomid:{hash(msg.as_bytes())}"
    from_name, from_addr = parseaddr(str(msg.get("From", "")))
    subject = re.sub(r"\s+", " ", str(msg.get("Subject", ""))).strip()[:1000]
    try:
        received = parsedate_to_datetime(str(msg.get("Date"))).astimezone().replace(tzinfo=None) if msg.get("Date") else None
    except (TypeError, ValueError):
        received = None
    html, text = _bodies(msg)
    body = plain_text(html, text)
    kind = classify(subject, body)
    irt, refs = str(msg.get("In-Reply-To", "") or ""), str(msg.get("References", "") or "")
    m = matcher.match(kind, from_addr, from_name, subject, body, irt, refs)
    if kind == "other" and m.score < 100:
        m = Match(None, 0, "", [])  # an unrelated message is never tied to a job by its wording alone
    out = {"kind": kind, "job_id": m.job_id, "why": m.why, "candidates": m.candidates, "subject": subject,
           "from": from_addr, "received": received, "effect": None}
    sender = from_name or from_addr
    note = f"E-mail ({KIND_LABEL[kind].lower()}): “{subject[:140]}” from {sender}" + (f", {received:%d.%m.%Y}" if received else "")
    with session_scope() as s:
        job = s.get(Job, m.job_id) if m.job_id else None
        if kind in TARGET and job is not None:
            out["effect"] = apply_effect(s, job, kind, received_at=received, note=note, dry_run=dry_run)
            if out["effect"] and not dry_run and job.id in matcher.jobs:
                matcher.jobs[job.id].status = job.status
        if dry_run:
            s.rollback()
            return out
        row = s.scalars(select(EmailMessage).where(EmailMessage.message_id == message_id)).first()
        if row is None:
            row = EmailMessage(message_id=message_id)
            s.add(row)
        row.from_addr, row.subject, row.received_at = from_addr[:300], subject, received
        row.reply_kind, row.job_id, row.effect = kind, m.job_id, out["effect"]
        row.snippet = _quoted_cut(body)[:900]
        row.match_note = (m.why if m.job_id else (f"possible jobs: {', '.join(map(str, m.candidates))}" if m.candidates
                                                  else m.why))[:300] or None
    return out


def reapply_linked(dry_run: bool = True) -> list[dict]:
    """Linked replies whose effect was blocked, applied again in date order (after a rule fix). Dry run by default."""
    from sqlalchemy import select

    from app.db import session_scope
    from app.models import EmailMessage, Job

    out = []
    with session_scope() as s:
        rows = s.scalars(select(EmailMessage).where(EmailMessage.job_id.is_not(None), EmailMessage.effect.is_(None),
                                                    EmailMessage.reply_kind.in_(tuple(TARGET)))
                         .order_by(EmailMessage.received_at)).all()
        for row in rows:
            job = s.get(Job, row.job_id)
            note = (f"E-mail ({KIND_LABEL[row.reply_kind].lower()}): “{(row.subject or '')[:140]}” from {row.from_addr}"
                    + (f", {row.received_at:%d.%m.%Y}" if row.received_at else ""))
            effect = apply_effect(s, job, row.reply_kind, received_at=row.received_at, note=note, dry_run=dry_run)
            if effect:
                out.append({"reply": row.id, "job": job.id, "company": job.company, "effect": effect, "subject": row.subject})
                if not dry_run:
                    row.effect = effect
        if dry_run:
            s.rollback()
    return out


def apply_effect(session, job, kind: str, *, received_at=None, note: str, forced: bool = False,
                 dry_run: bool = False) -> str | None:
    """Move the job (and copies of the same vacancy) along: never backwards, never over a newer status of yours."""
    from sqlalchemy import select

    from app.enums import JobStatus
    from app.models import StatusEvent
    from app.services.jobs import set_status
    from app.services.submit import vacancy_twins

    target = TARGET.get(kind)
    if target is None or job is None:
        return None
    before = job.status

    def changed_by_you_after(j) -> bool:
        """Your own status change (not one made from an e-mail) after this e-mail arrived — yours wins."""
        if not received_at or not j.status_changed_at or j.status_changed_at <= received_at or j.status in ("APPLYING", "APPLIED"):
            return False
        last = session.scalars(select(StatusEvent).where(StatusEvent.job_id == j.id)
                               .order_by(StatusEvent.changed_at.desc(), StatusEvent.id.desc()).limit(1)).first()
        return not (last and (last.note or "").startswith("E-mail ("))

    def allowed(j) -> bool:
        if forced:
            return j.status != target
        if changed_by_you_after(j):
            return False
        if kind == "rejection":  # an open job only reaches here on company AND title: you applied outside the app
            return j.status in _SENT or j.status in _OPEN
        if kind == "interview":
            return j.status in _OPEN or _RANK.get(j.status, 99) < _RANK["INTERVIEW"]
        if kind == "next_step":
            return j.status in _OPEN or j.status in ("APPLYING", "APPLIED")
        return j.status in _OPEN or j.status == "APPLYING"  # received

    if not allowed(job):
        return None
    if dry_run:
        return f"{before} → {target}"
    set_status(session, job, JobStatus(target), note=note)
    for twin in vacancy_twins(session, job):
        if allowed(twin):
            set_status(session, twin, JobStatus(target), note=f"{note} (same vacancy as job {job.id})")
    return f"{before} → {target}"
