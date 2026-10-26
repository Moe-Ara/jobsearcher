"""Render a tailored CV (dict from app.services.cv.build_cv) as PDF and as LaTeX.

PDF: single column, real text (no icon fonts, no columns), ASCII date ranges ("02/2024 - 05/2026") — the layout
ATS parsers read reliably. The reviewer agent's ATS check extracts this PDF's text layer and verifies it.
This is the built-in fallback renderer; the primary one is LaTeX (app.services.latex, via app.services.toolbox).
"""

from __future__ import annotations

import io
from typing import Any

NAVY = "#1F3A68"
GREY = "#5A6070"
RULE = "#B9C3D3"


def _latin1_safe(text: str) -> str:
    """The built-in PDF fonts cover cp1252 (German umlauts, ß, en dash). Map the rest to plain equivalents."""
    replacements = {"→": "->", "≥": ">=", "≤": "<=", "•": "-", "✓": "", " ": " ", "‑": "-", "−": "-"}
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def _esc(text: str) -> str:
    from xml.sax.saxutils import escape

    return escape(_latin1_safe(text or ""))


def page_count(cv: dict[str, Any]) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(render_pdf(cv))).pages)


def render_pdf(cv: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.1 * cm,
                            bottomMargin=1.0 * cm, title=f"{cv['name']} - CV", author=cv["name"])
    navy, grey = colors.HexColor(NAVY), colors.HexColor(GREY)
    st = {
        "name": ParagraphStyle("name", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=navy),
        "headline": ParagraphStyle("headline", fontName="Helvetica", fontSize=11.5, leading=15),
        "contact": ParagraphStyle("contact", fontName="Helvetica", fontSize=9, leading=12),
        "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=navy, spaceBefore=5),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.4, leading=12, alignment=TA_LEFT),
        "bold": ParagraphStyle("bold", fontName="Helvetica-Bold", fontSize=10, leading=13),
        "right": ParagraphStyle("right", fontName="Helvetica", fontSize=9, leading=13, textColor=grey, alignment=2),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=9, leading=12, textColor=grey),
        "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.4, leading=12, leftIndent=11,
                                 bulletIndent=2),
    }
    width = doc.width - 12  # the frame pads 6 pt on each side: tables must use the same width as paragraphs
    s = cv["sections"]
    c = cv["contact"]
    story: list[Any] = [Paragraph(_esc(cv["name"].upper()), st["name"]), Paragraph(_esc(cv["headline"]), st["headline"]),
                        Spacer(1, 3)]
    address = ", ".join(x for x in (c.get("street"), " ".join(y for y in (c.get("postal_code"), c.get("location")) if y)) if x)
    contact = " | ".join(_esc(x) for x in (c.get("email"), c.get("phone"), address) if x)
    links = " | ".join(f'<link href="{_esc(link["url"])}">{_esc(link["label"])}</link>' for link in c.get("links", []))
    story.append(Paragraph(contact, st["contact"]))
    if links:
        story.append(Paragraph(links, st["contact"]))
    if cv.get("personal"):
        story.append(Paragraph(" | ".join(_esc(x) for x in cv["personal"]), st["contact"]))
    story += [Spacer(1, 3), HRFlowable(width="100%", thickness=1.2, color=navy, spaceBefore=1, spaceAfter=2)]

    def heading(text: str, first: Any) -> Any:
        """Section heading kept on the same page as the section's first block (no orphaned headings)."""
        return KeepTogether([Paragraph(_esc(text.upper()), st["h"]),
                             HRFlowable(width="100%", thickness=0.6, color=colors.HexColor(RULE), spaceBefore=1, spaceAfter=3),
                             *(first if isinstance(first, list) else [first])])

    def row(left: str, right: str, lstyle: str = "bold") -> Table:
        t = Table([[Paragraph(left, st[lstyle]), Paragraph(_esc(right), st["right"])]], colWidths=[width * 0.72, width * 0.28],
                  hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return t

    if cv.get("profile_statement"):
        story.append(heading(s.get("profile", "Profile"), Paragraph(_esc(cv["profile_statement"]), st["body"])))
    for i, r in enumerate(cv["experience"]):
        block = [row(_esc(r["title"]), r["dates"]),
                 row(_esc(r["org"] + (f", {r['place']}" if r.get("place") else "")), r.get("detail") or "", "small")]
        block += [Paragraph(_esc(b), st["bullet"], bulletText="-") for b in r["bullets"]]
        story += [heading(s.get("experience", "Experience"), block) if i == 0 else KeepTogether(block), Spacer(1, 3)]
    for i, e in enumerate(cv["education"]):
        block = [row(_esc(e["title"]), e["dates"]), row(_esc(e["org"]), e.get("grade") or "", "small")]
        if e.get("focus"):
            block.append(Paragraph(_esc(e["focus"]), st["body"]))
        if e.get("thesis"):
            t = e["thesis"]
            text = f"<b>{_esc(t['label'])}:</b> <i>{_esc(t['title'])}</i>" + (f" - {_esc(t['summary'])}" if t.get("summary") else "")
            block.append(Paragraph(text, st["body"]))
        if e.get("project"):
            block.append(Paragraph(f"<b>{_esc(e['project']['label'])}:</b> {_esc(e['project']['title'])}", st["body"]))
        if e.get("modules"):
            block.append(Paragraph(f"<b>{_esc(e['modules']['label'])}:</b> {_esc(e['modules']['text'])}", st["body"]))
        story += [heading(s.get("education", "Education"), block) if i == 0 else KeepTogether(block), Spacer(1, 3)]
    if cv.get("projects"):
        items = [Paragraph(_esc(p), st["bullet"], bulletText="-") for p in cv["projects"]]
        story += [heading(s.get("projects", "Projects"), items[:1]), *items[1:]]
    rows = [[Paragraph(f"<b>{_esc(r['label'])}</b>", st["body"]), Paragraph(_esc(", ".join(r["items"])), st["body"])]
            for r in cv["skills"] if r["items"]]
    if rows:
        t = Table(rows, colWidths=[width * 0.3, width * 0.7], hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5), ("TOPPADDING", (0, 0), (-1, -1), 0)]))
        story.append(heading(s.get("skills", "Skills"), t))
    story.append(heading(s.get("languages", "Languages"), Paragraph(_esc("   |   ".join(cv["languages"])), st["body"])))
    doc.build(story)
    return buf.getvalue()


def sender_lines(cv: dict[str, Any]) -> list[str]:
    """Name, street, postcode + town, phone, e-mail — whatever config/cv.yaml has (never invented)."""
    c = cv["contact"]
    town = " ".join(x for x in (c.get("postal_code"), c.get("location")) if x)
    return [x for x in (cv["name"], c.get("street"), town, c.get("phone"), c.get("email")) if x]


def render_letter_pdf(letter: str, subject: str | None, cv: dict[str, Any], recipient: list[str], date_line: str,
                      attachments: list[str] | None = None, attachments_label: str = "Anlagen") -> bytes:
    """One-page application letter in the classic German layout (your instruction, DIN 5008 style):

        sender block (name, street, postcode + town, phone, e-mail)
        return-address line (small, one line) above the recipient
        recipient: company, department, contact person, street, postcode + town
        date (TT.MM.JJJJ), subject line in bold
        letter text (salutation → … → closing and name, from the letter pipeline)
        attachments ("Anlagen: Lebenslauf, Zeugnisse")
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.0 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm, title=subject or "Cover letter", author=cv["name"])
    grey = colors.HexColor(GREY)
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=10.5, leading=14.5, spaceAfter=8)
    line = ParagraphStyle("l", parent=body, spaceAfter=0)
    name = ParagraphStyle("n", parent=line, fontName="Helvetica-Bold", textColor=colors.HexColor(NAVY))
    returnline = ParagraphStyle("rl", fontName="Helvetica", fontSize=7.5, leading=10, textColor=grey)
    right = ParagraphStyle("r", parent=line, alignment=2)
    bold = ParagraphStyle("bo", parent=body, fontName="Helvetica-Bold")
    sender = sender_lines(cv)
    story: list[Any] = [Paragraph(_esc(sender[0]), name)] + [Paragraph(_esc(x), line) for x in sender[1:]]
    c = cv["contact"]
    town = " ".join(x for x in (c.get("postal_code"), c.get("location")) if x)
    back = ", ".join(x for x in (cv["name"], c.get("street"), town) if x)  # return address: name, street, town
    story += [Spacer(1, 30), Paragraph(f"<u>{_esc(back)}</u>", returnline), Spacer(1, 4)]
    story += [Paragraph(_esc(x), line) for x in recipient if x]
    story += [Spacer(1, 26), Paragraph(_esc(date_line), right), Spacer(1, 18)]
    if subject:
        story.append(Paragraph(_esc(subject), bold))
    story.append(Spacer(1, 6))
    for para in [p.strip() for p in (letter or "").split("\n\n") if p.strip()]:
        story.append(Paragraph(_esc(para).replace("\n", "<br/>"), body))
    if attachments:
        story += [Spacer(1, 10), Paragraph(f"<b>{_esc(attachments_label)}</b>", line)]
        story += [Paragraph(_esc(a), line) for a in attachments]
    doc.build(story)
    return buf.getvalue()
