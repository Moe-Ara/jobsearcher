"""Field normalisation and light inference (country, remote, seniority, experience, languages)."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.core.text import strip_accents

# --------------------------------------------------------------------------- titles

_GENDER_RE = re.compile(
    r"\(\s*(?:[mwfdxi]\s*[/|,]\s*)+[mwfdxi]\s*\)"  # (m/w/d), (f/m/x), (m|w|d)
    r"|\(\s*(?:all genders?|alle geschlechter|gn|mwd|wmd|mfd|fmd|fmx|mfx|div|d)\s*\*?\s*\)"
    r"|\b[mwfd]\s*/\s*[mwfd](?:\s*/\s*[mwfdx])?\b"
    r"|\ball genders?\b|\(?\*?gn\*?\)?$",
    re.IGNORECASE,
)

_TITLE_SYNONYMS = {
    "sr": "senior",
    "snr": "senior",
    "jr": "junior",
    "ai": "artificial intelligence",
    "ki": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "nlp": "natural language processing",
    "llm": "large language model",
    "llms": "large language model",
    "genai": "generative artificial intelligence",
    "swe": "software engineer",
    "sde": "software engineer",
    "dev": "developer",
    "devs": "developer",
    "eng": "engineer",
    "engr": "engineer",
    "entwickler": "developer",
    "entwicklerin": "developer",
    "softwareentwickler": "software developer",
    "softwareentwicklerin": "software developer",
    "ingenieur": "engineer",
    "ingenieurin": "engineer",
    "informatiker": "computer scientist",
    "informatikerin": "computer scientist",
    "mgr": "manager",
    "mlops": "mlops",
    "sre": "site reliability engineer",
    "fullstack": "full stack",
    "backend": "back end",
    "frontend": "front end",
}

_TITLE_NOISE_RE = re.compile(r"\b(?:remote|hybrid|on[- ]?site|vollzeit|teilzeit|full[- ]?time|part[- ]?time)\b")


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    t = _GENDER_RE.sub(" ", title)
    t = strip_accents(t.lower())
    t = re.sub(r"\bc\+\+", " cplusplus ", t)
    t = re.sub(r"\bc#", " csharp ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = _TITLE_NOISE_RE.sub(" ", t)
    words = [_TITLE_SYNONYMS.get(w, w) for w in t.split()]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


# --------------------------------------------------------------------------- companies

_LEGAL_RE = re.compile(
    r"\b(?:gmbh|ag|se|kg|kgaa|mbh|ohg|gbr|ug|co|inc|incorporated|ltd|limited|llc|l l c|plc|corp|corporation"
    r"|company|e v|ev|haftungsbeschrankt|holding|holdings|group|gruppe|international|fze|fzco|fz llc|fz"
    r"|s a|sa|b v|bv|n v|nv|ab|as|oy|spa|s p a|pjsc|psc|jsc|wll|w l l)\b"
)


def normalize_company(name: str | None) -> str:
    if not name:
        return ""
    n = strip_accents(name.lower()).replace("&", " and ")
    n = re.sub(r"[^a-z0-9]+", " ", n)
    prev = None
    while prev != n:  # strip stacked legal suffixes ("gmbh & co kg")
        prev = n
        n = _LEGAL_RE.sub(" ", n)
        n = re.sub(r"\s+", " ", n).strip()
    n = re.sub(r"\band\s*$", "", n).strip()
    return n or strip_accents(name.lower()).strip()


# --------------------------------------------------------------------------- URLs

_TRACKING_PARAMS = {
    "gh_src", "gh_jid", "lever-source", "lever-origin", "source", "src", "ref", "refid", "referer",
    "trk", "trkinfo", "trackingid", "tracking_id", "lipi", "midtoken", "midsig", "eid", "otptoken",
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid", "mc_cid",
    "mc_eid", "from", "tk", "alid", "cmp", "campaign", "rx_source", "rx_medium", "rx_campaign", "sid",
    "iid", "feed", "jobagent", "searchid", "origin", "_ga",
}  # fmt: skip


def canonicalize_url(url: str | None) -> str | None:
    """Stable URL for dedup: https, lowercase host, no tracking params/fragment/trailing slash.

    Known job boards are reduced to their stable job-id form.
    """
    if not url:
        return None
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return None
    parts = urlsplit(url)
    host = parts.netloc.lower().split("@")[-1]
    if host.endswith(":443") or host.endswith(":80"):
        host = host.rsplit(":", 1)[0]
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/") or "/"

    m = re.search(r"/(?:comm/)?jobs/view/(?:[^/]*-)?(\d{6,})", path)
    if "linkedin." in host and m:
        return f"https://www.linkedin.com/jobs/view/{m.group(1)}"
    query = dict(parse_qsl(parts.query, keep_blank_values=False))
    if "indeed." in host:
        jk = query.get("jk") or query.get("vjk")
        if jk:
            return f"https://{host}/viewjob?jk={jk}"
    if "arbeitsagentur.de" in host:
        m = re.search(r"/jobdetail/([^/?#]+)", path)
        if m:
            return f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{m.group(1)}"

    kept = sorted((k, v) for k, v in query.items() if k.lower() not in _TRACKING_PARAMS)
    return urlunsplit(("https", host, path, urlencode(kept), ""))


JOB_BOARD_HOSTS = {
    "linkedin.": "linkedin",
    "indeed.": "indeed",
    "stepstone.": "stepstone",
    "arbeitsagentur.de": "arbeitsagentur",
    "xing.com": "xing",
    "glassdoor.": "glassdoor",
    "bayt.com": "bayt",
    "naukrigulf.com": "naukrigulf",
    "gulftalent.com": "gulftalent",
    "monster.": "monster",
    "jobware.de": "jobware",
    "stellenanzeigen.de": "stellenanzeigen",
    "get-in-it.de": "get-in-it",
    "wellfound.com": "wellfound",
}
AGGREGATOR_HOSTS = {"jooble.": "jooble", "careerjet.": "careerjet", "adzuna.": "adzuna", "kimeta.": "kimeta", "jobrapido.": "jobrapido"}


def classify_url_host(url: str | None) -> tuple[str, str] | None:
    """Return (source_name, source_type) for known job boards/aggregators, else None."""
    if not url:
        return None
    host = urlsplit(url).netloc.lower()
    for needle, name in JOB_BOARD_HOSTS.items():
        if needle in host:
            return name, "job_board"
    for needle, name in AGGREGATOR_HOSTS.items():
        if needle in host:
            return name, "aggregator"
    return None


# --------------------------------------------------------------------------- locations

COUNTRY_ALIASES: dict[str, list[str]] = {
    "Germany": ["germany", "deutschland", "de", "deu", "german"],
    "Saudi Arabia": ["saudi arabia", "ksa", "saudi", "kingdom of saudi arabia", "sa", "sau", "المملكة العربية السعودية"],
    "UAE": ["uae", "united arab emirates", "u a e", "emirates", "ae", "are", "الإمارات"],
    "Austria": ["austria", "osterreich", "at", "aut"],
    "Switzerland": ["switzerland", "schweiz", "suisse", "ch", "che"],
    "Netherlands": ["netherlands", "nederland", "the netherlands", "holland", "nl", "nld"],
    "United Kingdom": ["united kingdom", "uk", "england", "great britain", "britain", "gb", "gbr", "scotland", "wales",
                       "northern ireland"],
    # US state names: "Cambridge, Massachusetts" / "Oxford, Mississippi" are not the English university towns
    "United States": ["united states", "usa", "us", "u s", "united states of america", "california", "massachusetts",
                      "texas", "new york", "illinois", "virginia", "colorado", "florida", "oregon", "pennsylvania",
                      "new jersey", "north carolina", "michigan", "ohio", "arizona", "utah", "maryland", "minnesota",
                      "connecticut", "mississippi", "tennessee", "missouri", "wisconsin", "indiana", "nevada",
                      "washington dc", "washington d c"],
    "France": ["france", "fr", "fra"],
    "Spain": ["spain", "espana", "es", "esp"],
    "Italy": ["italy", "italia", "it", "ita"],
    "Poland": ["poland", "polska", "pl", "pol"],
    "Portugal": ["portugal", "pt", "prt"],
    "Ireland": ["ireland", "ie", "irl"],
    "Sweden": ["sweden", "se", "swe"],
    "Denmark": ["denmark", "dk", "dnk"],
    "Belgium": ["belgium", "be", "bel"],
    "Czech Republic": ["czech republic", "czechia", "cz", "cze"],
    "Romania": ["romania", "ro", "rou"],
    "India": ["india", "in", "ind"],
    "Canada": ["canada", "ca", "can"],
    "Qatar": ["qatar", "qa", "qat"],
    "Bahrain": ["bahrain", "bh", "bhr"],
    "Kuwait": ["kuwait", "kw", "kwt"],
    "Oman": ["oman", "om", "omn"],
    "Egypt": ["egypt", "eg", "egy"],
    "Jordan": ["jordan", "jo", "jor"],
    "Singapore": ["singapore", "sg", "sgp"],
    "Israel": ["israel", "il", "isr"],
    "Turkey": ["turkey", "turkiye", "tr", "tur"],
    "Finland": ["finland", "fi", "fin"],
    "Norway": ["norway", "no", "nor"],
    "Hungary": ["hungary", "hu", "hun"],
    "Greece": ["greece", "gr", "grc"],
    "Bulgaria": ["bulgaria", "bg", "bgr"],
    "Serbia": ["serbia", "rs", "srb"],
    "Ukraine": ["ukraine", "ua", "ukr"],
    "Pakistan": ["pakistan", "pk", "pak"],
    "China": ["china", "cn", "chn"],
    "Japan": ["japan", "jp", "jpn"],
    "Australia": ["australia", "au", "aus"],
    "Brazil": ["brazil", "br", "bra"],
    "Mexico": ["mexico", "mx", "mex"],
    "Malaysia": ["malaysia", "my", "mys"],
    "Philippines": ["philippines", "ph", "phl"],
    "Vietnam": ["vietnam", "viet nam", "vn", "vnm"],
    "Thailand": ["thailand", "th", "tha"],
    "Indonesia": ["indonesia", "id", "idn"],
    "South Korea": ["south korea", "korea", "kr", "kor"],
    "Taiwan": ["taiwan", "tw", "twn"],
    "Hong Kong": ["hong kong", "hk", "hkg"],
    "New Zealand": ["new zealand", "nz", "nzl"],
    "South Africa": ["south africa", "za", "zaf"],
    "Nigeria": ["nigeria", "ng", "nga"],
    "Kenya": ["kenya", "ke", "ken"],
    "Morocco": ["morocco", "ma", "mar"],
    "Tunisia": ["tunisia", "tn", "tun"],
    "Lebanon": ["lebanon", "lb", "lbn"],
    "Argentina": ["argentina", "ar", "arg"],
    "Chile": ["chile", "cl", "chl"],
    "Colombia": ["colombia", "co", "col"],
    "Peru": ["peru", "pe", "per"],
    "Croatia": ["croatia", "hr", "hrv"],
    "Slovakia": ["slovakia", "sk", "svk"],
    "Slovenia": ["slovenia", "si", "svn"],
    "Lithuania": ["lithuania", "lt", "ltu"],
    "Latvia": ["latvia", "lv", "lva"],
    "Estonia": ["estonia", "ee", "est"],
    "Luxembourg": ["luxembourg", "luxemburg", "lu", "lux"],
    "Cyprus": ["cyprus", "cy", "cyp"],
    "Malta": ["malta", "mt", "mlt"],
}

CITY_COUNTRY: dict[str, str] = {
    **{c: "Germany" for c in [
        "berlin", "munich", "munchen", "muenchen", "hamburg", "frankfurt", "cologne", "koln", "koeln",
        "dusseldorf", "duesseldorf", "stuttgart", "dresden", "leipzig", "hannover", "hanover", "nuremberg",
        "nurnberg", "nuernberg", "karlsruhe", "darmstadt", "aachen", "bremen", "essen", "dortmund", "mannheim",
        "ulm", "ingolstadt", "wolfsburg", "bonn", "heidelberg", "regensburg", "augsburg", "jena", "potsdam",
        "kiel", "freiburg", "erlangen", "sindelfingen", "boblingen", "boeblingen", "walldorf", "garching",
        "braunschweig", "bochum", "wiesbaden", "mainz", "kassel", "munster", "muenster", "saarbrucken",
        "magdeburg", "rostock", "chemnitz", "paderborn", "gottingen", "goettingen", "wurzburg", "wuerzburg",
        "heilbronn", "ludwigsburg", "esslingen", "friedrichshafen", "tubingen", "tuebingen", "konstanz",
        "oberkochen", "weinheim", "ludwigshafen", "leverkusen", "duisburg", "gelsenkirchen", "bielefeld",
        "osnabruck", "oldenburg", "lubeck", "luebeck", "erfurt", "halle", "schweinfurt", "bamberg", "bayreuth",
        "passau", "landshut", "rosenheim", "unterschleissheim", "ottobrunn", "taufkirchen", "eschborn",
        "offenbach", "neckarsulm", "zuffenhausen", "weissach", "herzogenaurach", "gerlingen", "renningen",
        "abstatt", "leonberg", "ditzingen", "baden-baden", "pforzheim", "reutlingen", "kaiserslautern",
    ]},
    **{c: "Saudi Arabia" for c in [
        "riyadh", "jeddah", "jiddah", "dammam", "khobar", "al khobar", "dhahran", "neom", "mecca", "makkah",
        "medina", "madinah", "jubail", "yanbu", "tabuk", "abha", "thuwal", "kaust", "king abdullah economic city",
        "al riyadh", "ar riyad", "kafd", "diriyah", "al ula", "alula", "red sea", "taif", "qassim", "buraidah",
    ]},
    **{c: "UAE" for c in [
        "dubai", "abu dhabi", "sharjah", "ajman", "al ain", "ras al khaimah", "fujairah", "umm al quwain",
        "masdar", "jebel ali", "dubai internet city", "dubai media city", "difc", "adgm", "yas island", "saadiyat",
        "dubai silicon oasis", "jlt", "business bay",
    ]},
    **{c: "Luxembourg" for c in [
        "esch-sur-alzette", "esch sur alzette", "belval", "kirchberg", "bertrange", "strassen", "leudelange",
        "differdange", "dudelange", "windhof", "contern", "capellen", "munsbach", "senningerberg", "howald",
        "hesperange", "mamer", "ettelbruck", "diekirch", "mersch", "grevenmacher", "remich", "wiltz", "betzdorf",
        "bettembourg", "foetz", "mondercange", "sanem", "schifflange", "hautcharage", "colmar-berg", "roost",
    ]},
    "london": "United Kingdom", "manchester": "United Kingdom", "paris": "France", "amsterdam": "Netherlands",
    "eindhoven": "Netherlands", "vienna": "Austria", "wien": "Austria", "graz": "Austria", "linz": "Austria",
    "zurich": "Switzerland", "zuerich": "Switzerland", "basel": "Switzerland", "geneva": "Switzerland",
    "bern": "Switzerland", "lausanne": "Switzerland", "madrid": "Spain", "barcelona": "Spain", "lisbon": "Portugal",
    "warsaw": "Poland", "krakow": "Poland", "wroclaw": "Poland", "prague": "Czech Republic", "brno": "Czech Republic",
    "dublin": "Ireland", "stockholm": "Sweden", "copenhagen": "Denmark", "brussels": "Belgium", "milan": "Italy",
    "bucharest": "Romania", "budapest": "Hungary", "doha": "Qatar", "manama": "Bahrain", "muscat": "Oman",
    "cairo": "Egypt", "amman": "Jordan", "bangalore": "India", "bengaluru": "India", "hyderabad": "India",
    "pune": "India", "new york": "United States", "san francisco": "United States", "seattle": "United States",
    "boston": "United States", "austin": "United States", "toronto": "Canada", "singapore": "Singapore",
    "tel aviv": "Israel", "istanbul": "Turkey", "helsinki": "Finland", "oslo": "Norway", "athens": "Greece",
    # German states / regions / more cities (incl. Bodensee region) and nationwide wording
    **{c: "Germany" for c in [
        "bundesweit", "deutschlandweit", "nrw", "nordrhein westfalen", "baden wurttemberg", "baden wuerttemberg", "bayern",
        "bavaria", "hessen", "hesse", "sachsen", "saxony", "niedersachsen", "lower saxony", "thuringen", "thueringen",
        "brandenburg", "rheinland pfalz", "schleswig holstein", "mecklenburg vorpommern", "sachsen anhalt", "saarland",
        "ravensburg", "weingarten", "markdorf", "immenstaad", "meersburg", "uberlingen", "ueberlingen", "tettnang",
        "lindau", "wangen", "kempten", "memmingen", "biberach", "neu ulm", "aschaffenburg", "zweibrucken",
        "zweibruecken", "sankt ingbert", "st ingbert", "homburg", "kaiserslautern", "trier", "koblenz", "siegen",
        "hagen", "wuppertal", "krefeld", "monchengladbach", "moenchengladbach", "aachen", "gutersloh", "guetersloh",
        "hamm", "herne", "oberhausen", "mulheim", "muelheim", "solingen", "remscheid", "neuss", "ratingen", "hildesheim",
        "salzgitter", "wolfenbuttel", "celle", "luneburg", "lueneburg", "flensburg", "neumunster", "cottbus", "gera",
        "zwickau", "plauen", "gorlitz", "goerlitz", "dessau", "wismar", "stralsund", "greifswald", "schwerin",
        "fulda", "giessen", "marburg", "hanau", "darmstadt", "russelsheim", "ruesselsheim", "frankenthal", "speyer",
        "landau", "worms", "bad homburg", "oberursel", "offenburg", "villingen", "schwenningen", "tuttlingen",
        "singen", "radolfzell", "lorrach", "loerrach", "aalen", "heidenheim", "schwabisch hall", "schwaebisch hall",
        "schwabisch gmund", "schwaebisch gmuend", "goppingen", "goeppingen", "boblingen", "nurtingen", "waiblingen",
        "kirchheim", "ansbach", "amberg", "weiden", "straubing", "deggendorf", "kaufbeuren", "neuburg", "freising",
        "dachau", "erding", "starnberg", "garmisch", "traunstein", "coburg", "hof", "aschheim", "unterfohring",
        "unterfoehring", "ismaning", "martinsried", "planegg", "puchheim", "germering", "furth", "fuerth",
    ]},
    "monterrey": "Mexico", "saltillo": "Mexico", "guadalajara": "Mexico", "coimbra": "Portugal", "braga": "Portugal",
    "lima": "Peru", "bogota": "Colombia", "buenos aires": "Argentina", "washington": "United States",
    # cities seen in real feeds that were missing
    **{c: "India" for c in ["gurugram", "gurgaon", "delhi", "new delhi", "noida", "chennai", "mumbai", "kolkata",
                            "ahmedabad", "kochi", "trivandrum", "jaipur", "coimbatore"]},
    **{c: "United States" for c in [
        "dallas", "houston", "atlanta", "denver", "chicago", "los angeles", "san jose", "san diego", "redwood city",
        "palo alto", "mountain view", "sunnyvale", "santa clara", "menlo park", "raleigh", "durham", "miami", "phoenix",
        "detroit", "philadelphia", "pittsburgh", "portland", "salt lake city", "minneapolis", "nashville", "charlotte",
        "washington dc", "washington d c", "remotely in the usa",
        # US states
        "california", "texas", "illinois", "north carolina", "massachusetts", "florida", "colorado", "virginia",
        "arizona", "oregon", "ohio", "michigan", "pennsylvania", "new jersey", "utah", "minnesota", "tennessee",
        "wisconsin", "maryland", "missouri", "indiana", "connecticut", "kentucky", "south carolina", "alabama",
    ]},
    **{c: "France" for c in ["nantes", "lille", "lyon", "toulouse", "annecy", "bordeaux", "marseille", "nice",
                             "grenoble", "strasbourg", "montpellier", "rennes", "sophia antipolis"]},
    "sofia": "Bulgaria", "plovdiv": "Bulgaria", "belgrade": "Serbia", "novi sad": "Serbia", "santiago": "Chile",
    "kuala lumpur": "Malaysia", "penang": "Malaysia", "cyberjaya": "Malaysia", "petaling jaya": "Malaysia",
    "manila": "Philippines", "ho chi minh": "Vietnam", "hanoi": "Vietnam", "bangkok": "Thailand", "jakarta": "Indonesia",
    "sydney": "Australia", "melbourne": "Australia", "brisbane": "Australia", "sao paulo": "Brazil",
    "mexico city": "Mexico", "karachi": "Pakistan", "lahore": "Pakistan", "islamabad": "Pakistan", "seoul": "South Korea",
    "tokyo": "Japan", "shanghai": "China", "beijing": "China", "shenzhen": "China", "taipei": "Taiwan",
    "edinburgh": "United Kingdom", "birmingham": "United Kingdom", "bristol": "United Kingdom", "rotterdam": "Netherlands",
    "the hague": "Netherlands", "utrecht": "Netherlands", "antwerp": "Belgium", "ghent": "Belgium", "porto": "Portugal",
    "valencia": "Spain", "seville": "Spain", "gdansk": "Poland", "poznan": "Poland", "lodz": "Poland", "cluj": "Romania",
    "iasi": "Romania", "timisoara": "Romania", "zagreb": "Croatia", "ljubljana": "Slovenia", "bratislava": "Slovakia",
    "vilnius": "Lithuania", "riga": "Latvia", "tallinn": "Estonia", "salzburg": "Austria", "innsbruck": "Austria",
    # United Kingdom (no "york": it is inside "new york"; US states are matched before towns)
    "glasgow": "United Kingdom", "leeds": "United Kingdom", "liverpool": "United Kingdom", "sheffield": "United Kingdom",
    "newcastle upon tyne": "United Kingdom", "nottingham": "United Kingdom", "leicester": "United Kingdom",
    "cambridge": "United Kingdom", "oxford": "United Kingdom", "reading": "United Kingdom", "southampton": "United Kingdom",
    "brighton": "United Kingdom", "cardiff": "United Kingdom", "belfast": "United Kingdom", "aberdeen": "United Kingdom",
    "dundee": "United Kingdom", "coventry": "United Kingdom", "milton keynes": "United Kingdom", "guildford": "United Kingdom",
    "bath": "United Kingdom", "exeter": "United Kingdom", "plymouth": "United Kingdom", "swansea": "United Kingdom",
    "warrington": "United Kingdom", "stevenage": "United Kingdom", "slough": "United Kingdom", "basingstoke": "United Kingdom",
    "cheltenham": "United Kingdom", "norwich": "United Kingdom", "portsmouth": "United Kingdom", "bournemouth": "United Kingdom",
    "ipswich": "United Kingdom", "harwell": "United Kingdom", "didcot": "United Kingdom", "farnborough": "United Kingdom",
    "crawley": "United Kingdom", "watford": "United Kingdom", "luton": "United Kingdom", "sunderland": "United Kingdom",
    "middlesbrough": "United Kingdom", "stockport": "United Kingdom", "wolverhampton": "United Kingdom",
    "preston": "United Kingdom", "chester": "United Kingdom", "canterbury": "United Kingdom", "colchester": "United Kingdom",
    "warwick": "United Kingdom", "loughborough": "United Kingdom", "swindon": "United Kingdom", "gloucester": "United Kingdom",
    "northampton": "United Kingdom", "peterborough": "United Kingdom", "inverness": "United Kingdom", "stirling": "United Kingdom",
    "derby": "United Kingdom", "bracknell": "United Kingdom", "woking": "United Kingdom", "maidenhead": "United Kingdom",
    "london area": "United Kingdom", "greater london": "United Kingdom", "greater manchester": "United Kingdom",
}  # fmt: skip


def _alias_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for country, aliases in COUNTRY_ALIASES.items():
        idx[country.lower()] = country
        for a in aliases:
            idx[a] = country
    return idx


_ALIAS_INDEX = _alias_index()
_SHORT_ALIASES = {a for a in _ALIAS_INDEX if len(a) <= 3}


def canonical_country(value: str | None) -> str | None:
    """Map a free-text country (name or ISO code) to a canonical name."""
    if not value:
        return None
    v = re.sub(r"[^a-z؀-ۿ ]+", " ", strip_accents(value.lower())).strip()
    v = re.sub(r"\s+", " ", v)
    return _ALIAS_INDEX.get(v)


_US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND "
                 "OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split())


def infer_country(location: str | None, explicit: str | None = None) -> str | None:
    """Best-effort country from an explicit country field or a free-text location."""
    c = canonical_country(explicit)
    if c:
        return c
    if not location:
        return None
    loc = strip_accents(location.lower())
    loc_clean = re.sub(r"[^a-z؀-ۿ]+", " ", loc)
    padded = f" {loc_clean} "
    # 1) full country names / long aliases anywhere in the string
    for alias, country in sorted(_ALIAS_INDEX.items(), key=lambda kv: -len(kv[0])):
        if alias in _SHORT_ALIASES:
            continue
        if f" {alias} " in padded:
            return country
    # 2) known cities — but a US state code as the last segment wins over an English town name or an unknown town:
    #    "Cambridge, MA" / "Southfield, MI" are in the US; "Bangalore, IN" stays India. DE stays Germany.
    town = next((country for city, country in CITY_COUNTRY.items() if f" {city.replace('-', ' ')} " in padded), None)
    last = re.split(r"[,;|]", location)[-1].strip()
    if re.fullmatch(r"[A-Z]{2}", last) and last in _US_STATES and last != "DE" and town in (None, "United Kingdom"):
        return "United States"
    if town:
        return town
    # 3) ISO/short codes as their own segment, written in capitals: "Redwood City, US, California", "Riyadh, SA"
    for seg in re.split(r"[,/|;()\-–]", location):
        seg = seg.strip()
        if seg.isupper() and seg.lower() in _SHORT_ALIASES:
            return _ALIAS_INDEX[seg.lower()]
    for code in re.findall(r"\b(USA|UK|UAE|KSA)\b", location):
        return _ALIAS_INDEX[code.lower()]
    # 3b) US state code as the last segment: "Southfield, MI", "King of Prussia, PA" (codes that are also country
    #     codes, e.g. DE, were matched as countries above)
    last = re.split(r"[,;|]", location)[-1].strip()
    if re.fullmatch(r"[A-Z]{2}", last) and last in _US_STATES:
        return "United States"
    # 4) 5-digit postcode + town ("66386 Sankt Ingbert"): German-format address in German-market sources
    if re.search(r"(^|[,;]\s*)\d{5}\s+[A-ZÄÖÜ][a-zäöüß]", location):  # "66386 Sankt Ingbert", "Str. 4, 66482 Zweibrücken"
        return "Germany"
    return None


def infer_country_from_title(title: str | None) -> str | None:
    """Last resort for postings without a location, e.g. 'Data Scientist - Remote Malaysia (m/f/d)'."""
    if not title:
        return None
    padded = " " + re.sub(r"[^a-z؀-ۿ]+", " ", strip_accents(title.lower())) + " "
    for alias, country in sorted(_ALIAS_INDEX.items(), key=lambda kv: -len(kv[0])):
        if len(alias) > 3 and alias != "german" and f" {alias} " in padded:
            return country
    for code in re.findall(r"\b(USA|UK|UAE|KSA)\b", title):
        return _ALIAS_INDEX[code.lower()]
    return None


_REMOTE_RE = re.compile(r"\b(fully remote|100% remote|remote[- ]first|remote|work from home|wfh|home[- ]?office|telecommute|anywhere)\b", re.I)
_HYBRID_RE = re.compile(r"\b(hybrid|hybride?s? arbeiten|mobiles arbeiten|teilweise remote|partially remote|flexible work(ing)? location)\b", re.I)
_ONSITE_RE = re.compile(r"\b(on[- ]?site|onsite|in[- ]office|vor ort|präsenz)\b", re.I)


_BODY_REMOTE_RE = re.compile(r"\b(fully remote|100\s?% remote|remote[- ]first|work from anywhere|full remote)\b", re.I)


def infer_remote_status(title: str | None, location: str | None, body: str | None = None) -> str | None:
    """Title/location are a strong signal; the body only counts for explicit phrases."""
    head = " ".join(t for t in (title, location) if t)
    if head:
        if _HYBRID_RE.search(head):
            return "hybrid"
        if _REMOTE_RE.search(head):
            return "remote"
        if _ONSITE_RE.search(head):
            return "onsite"
    if body:
        text = body[:8000]
        if _BODY_REMOTE_RE.search(text):
            return "remote"
        if _HYBRID_RE.search(text) or re.search(r"\b(home[- ]?office|remote work|work from home)\b", text, re.I):
            return "hybrid"
    return None


# --------------------------------------------------------------------------- seniority / experience

_SENIORITY_PATTERNS = [
    ("Intern", r"\b(intern|internship|praktikum|praktikant|werkstudent|working student|trainee|thesis|abschlussarbeit)\b"),
    ("Executive", r"\b(head of|director|vice president|vp|chief|cto|cio|geschäftsführer|bereichsleiter)\b"),
    ("Principal", r"\b(principal|distinguished|fellow)\b"),
    ("Lead", r"\b(lead|staff|team ?lead|teamleiter|tech lead|architect)\b"),
    ("Senior", r"\b(senior|sr\.?|snr)\b"),
    ("Junior", r"\b(junior|jr\.?|graduate|entry[- ]level|berufseinsteiger|absolvent)\b"),
    ("Mid", r"\b(mid[- ]?level|intermediate|professional)\b"),
]


def infer_seniority(title: str | None) -> str | None:
    if not title:
        return None
    t = title.lower()
    for level, pattern in _SENIORITY_PATTERNS:
        if re.search(pattern, t):
            return level
    return None


_SOURCE_SENIORITY = [
    ("Intern", r"intern|student|werkstudent|praktik|trainee|apprentice"),
    ("Executive", r"executive|director|vp|vice president|c-level|head"),
    ("Principal", r"principal|distinguished"),
    ("Lead", r"lead|staff|manager"),
    ("Senior", r"senior|mid-senior|expert"),
    ("Junior", r"junior|entry|associate|graduate|berufseinsteiger|young professional"),
    ("Mid", r"mid|experienced|professional|berufserfahren|intermediate"),
]


def canonical_seniority(title: str | None, source_label: str | None = None) -> str | None:
    """Explicit title words win; otherwise map the ATS/board label to Intern/Junior/Mid/Senior/Lead/Principal/Executive."""
    from_title = infer_seniority(title)
    if from_title:
        return from_title
    if not source_label:
        return None
    label = source_label.lower()
    for level, pattern in _SOURCE_SENIORITY:
        if re.search(pattern, label):
            return level
    return None


_YEARS_RE = re.compile(
    r"(?<![\d.])(\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|–|to|bis)\s*(\d{1,2})\s*)?\+?\s*"
    r"(?:years?|yrs?|jahre[n]?)(?:\s+of)?(?:\s+(?:professional|relevant|hands-on|industry|work|practical))?"
    r"\s*(?:experience|erfahrung|berufserfahrung|in\b|of\b|working)",
    re.IGNORECASE,
)


def extract_years_experience(*texts: str | None) -> float | None:
    """Smallest 'N years of experience' requirement mentioned (min of ranges)."""
    found: list[int] = []
    for text in texts:
        if not text:
            continue
        for m in _YEARS_RE.finditer(text):
            n = int(m.group(1))
            if 0 < n <= 25:
                found.append(n)
    return float(min(found)) if found else None


_LANG_NAMES = {
    "German": r"german|deutsch(?:kenntnisse)?",
    "English": r"english|englisch(?:kenntnisse)?",
    "Arabic": r"arabic|arabisch",
    "French": r"french|französisch",
}
_LEVEL_RE = r"(native|mother tongue|muttersprach\w*|fluent|fluency|fließend\w*|verhandlungssicher\w*|business[- ]fluent|excellent|very good|sehr gut\w*|gut\w*|good|basic|grundkenntnisse|c2|c1|b2|b1|a2|proficien\w*)"


def extract_language_requirements(*texts: str | None) -> list[str]:
    """e.g. ['German (C1)', 'English (fluent)'] — level words within ~40 chars of the language name."""
    text = " ".join(t for t in texts if t)
    out: dict[str, str] = {}
    for lang, pattern in _LANG_NAMES.items():
        for m in re.finditer(rf"\b({pattern})\b", text, re.IGNORECASE):
            # "German (C1)" / "German: fluent" first, then "fluent German"
            after = re.match(rf"[\s:(\-–]*{_LEVEL_RE}", text[m.end() : m.end() + 30], re.IGNORECASE)
            before = re.search(rf"{_LEVEL_RE}[\s\w-]{{0,12}}$", text[max(0, m.start() - 30) : m.start()], re.IGNORECASE)
            lvl = after or before
            if lvl:
                out[lang] = lvl.group(1).lower()
                break
            out.setdefault(lang, "")
    return [f"{k} ({v})" if v else k for k, v in out.items()]
