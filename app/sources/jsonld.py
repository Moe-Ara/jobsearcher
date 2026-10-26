"""schema.org JobPosting (JSON-LD) extraction + plain page fallback.

Most career sites and many ATSs (Workday, SuccessFactors, Personio, StepStone, ...) embed JobPosting
JSON-LD for Google for Jobs, which makes it the most robust generic extraction strategy.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from app.core.text import clean_line, clean_text, html_to_text

log = logging.getLogger(__name__)


def _iter_nodes(data: Any):
    if isinstance(data, list):
        for item in data:
            yield from _iter_nodes(item)
    elif isinstance(data, dict):
        yield data
        for key in ("@graph", "itemListElement", "mainEntity", "item"):
            if key in data:
                yield from _iter_nodes(data[key])


def _is_job_posting(node: dict[str, Any]) -> bool:
    t = node.get("@type")
    types = t if isinstance(t, list) else [t]
    return any(isinstance(x, str) and x.lower() == "jobposting" for x in types)


def extract_jsonld_postings(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    out: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        text = (script.string or script.get_text() or "").strip()
        if not text:
            continue
        text = re.sub(r"^\s*<!\[CDATA\[|\]\]>\s*$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:  # some sites put raw newlines inside strings
                data = json.loads(re.sub(r"[\r\n\t]+", " ", text))
            except json.JSONDecodeError:
                continue
        out.extend(n for n in _iter_nodes(data) if _is_job_posting(n))
    return out


def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return _text(value.get("name") or value.get("value") or value.get("description"))
    if isinstance(value, list):
        return ", ".join(t for t in (_text(v) for v in value) if t)
    return str(value)


def _parse_date(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", value)
        return datetime.fromisoformat(m.group(1)) if m else None


def jsonld_to_fields(node: dict[str, Any]) -> dict[str, Any]:
    """Map a JobPosting node to JobPosting-compatible keyword fields."""
    org = _first(node.get("hiringOrganization")) or {}
    locations = node.get("jobLocation") or []
    if isinstance(locations, dict):
        locations = [locations]
    loc_parts, country = [], None
    for loc in locations:
        addr = loc.get("address", {}) if isinstance(loc, dict) else {}
        if isinstance(addr, str):
            loc_parts.append(addr)
            continue
        city = _text(addr.get("addressLocality"))
        region = _text(addr.get("addressRegion"))
        country = country or _text(addr.get("addressCountry")) or None
        part = ", ".join(x for x in (city, region) if x)
        if part:
            loc_parts.append(part)
    if country and loc_parts:
        # the town wins over a contradicting addressCountry (Michael Page tags Dubai postings "SG": a template default)
        from app.core.normalize import canonical_country, infer_country

        named = infer_country("; ".join(loc_parts))
        if named and named != (canonical_country(country) or country):
            country = named
    remote = None
    if "TELECOMMUTE" in json.dumps(node.get("jobLocationType", "")).upper():
        remote = "remote"

    salary = node.get("baseSalary") or {}
    sal_min = sal_max = None
    currency = None
    if isinstance(salary, dict):
        currency = salary.get("currency")
        val = salary.get("value")
        if isinstance(val, dict):
            sal_min = _to_float(val.get("minValue") or val.get("value"))
            sal_max = _to_float(val.get("maxValue") or val.get("value"))
        else:
            sal_min = sal_max = _to_float(val)

    ident = node.get("identifier")
    ext_id = _text(ident.get("value")) if isinstance(ident, dict) else (_text(ident) or None)

    return {
        "title": clean_line(_text(node.get("title")) or _text(node.get("name"))),
        "company": clean_line(_text(org.get("name") if isinstance(org, dict) else org)),
        "location": clean_line("; ".join(dict.fromkeys(loc_parts))),
        "country": clean_line(country) or None,
        "remote_status": remote,
        "description": html_to_text(_text(node.get("description"))),
        "requirements": html_to_text(
            "\n".join(x for x in (_text(node.get("qualifications")), _text(node.get("experienceRequirements")), _text(node.get("skills"))) if x)
        ),
        "responsibilities": html_to_text(_text(node.get("responsibilities"))),
        "employment_type": clean_line(_text(node.get("employmentType"))) or None,
        "date_posted": _parse_date(node.get("datePosted")),
        "salary_min": sal_min,
        "salary_max": sal_max,
        "currency": clean_line(currency) or None if currency else None,
        "external_job_id": clean_line(ext_id) or None if ext_id else None,
        "url": node.get("url") if isinstance(node.get("url"), str) else None,
    }


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def extract_page_fallback(html: str) -> dict[str, str]:
    """Title + main text when a page has no JobPosting JSON-LD."""
    soup = BeautifulSoup(html, "lxml")
    title = ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = og["content"]
    elif soup.find("h1"):
        title = soup.find("h1").get_text(" ")
    elif soup.title:
        title = soup.title.get_text(" ")
    site = soup.find("meta", attrs={"property": "og:site_name"})
    for tag in soup(["nav", "header", "footer", "aside", "script", "style", "noscript", "form", "svg"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"}) or soup.body or soup
    return {
        "title": clean_line(title),
        "site_name": clean_line(site.get("content")) if site and site.get("content") else "",
        "text": clean_text(html_to_text(str(main))),
    }
