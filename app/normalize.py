"""Normalization of messy free-text values coming from the registry.

Real ClinicalTrials.gov data is inconsistent: the same drug appears as
"Pembrolizumab", "pembrolizumab 200 mg IV Q3W", "MK-3475" and "Keytruda";
the same sponsor as "Merck Sharp & Dohme LLC" and "Merck Sharp & Dohme Corp.".
Each function here returns a *grouping key*; display labels are chosen later
as the most frequent raw spelling within a key (see aggregate.LabelPicker).
"""

from __future__ import annotations

import re
from functools import lru_cache

from app.vocab import PHASE_LABELS

# ---------------------------------------------------------------- phases ---

PHASE_ORDER = [
    "Early Phase 1",
    "Phase 1",
    "Phase 1/Phase 2",
    "Phase 2",
    "Phase 2/Phase 3",
    "Phase 3",
    "Phase 4",
    "Not Applicable",
    "Unknown",
]


def phase_label(phases: list[str] | None) -> str:
    """Collapse the API's phases array into one ordinal bucket.

    ["PHASE1","PHASE2"] -> "Phase 1/Phase 2"; [] or missing -> "Unknown".
    """
    if not phases:
        return "Unknown"
    labels = [PHASE_LABELS.get(p, p.title()) for p in phases]
    return "/".join(labels)


# ----------------------------------------------------------- interventions ---

_DOSE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|ml|iu|units?|%)"
    r"(?:\s*/\s*\d*(?:[.,]\d+)?\s*(?:mg|ml|kg|m2|m²|l|day|d|hr|h|dose|week|wk))?(?!\w)",
    re.IGNORECASE,
)
_FORM_WORDS = re.compile(
    r"\b(?:injections?|injectable|tablets?|capsules?|oral|intravenous(?:ly)?|iv|i\.v\.|sc|subcutaneous(?:ly)?|"
    r"infusion|solution|for injection|suspension|cream|gel|q\dw|q\d+w|every \d+ weeks?|"
    r"regimen|therapy|treatment|drug|arm|dose|dosing|high|low)\b",
    re.IGNORECASE,
)
_PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_NON_WORD = re.compile(r"[^\w\s\-+/]")
_WS = re.compile(r"\s+")

PLACEBO_KEY = "placebo"


def clean_intervention_name(name: str) -> str:
    """Lowercased, dose/form-stripped key for an intervention name."""
    s = name.lower()
    if "placebo" in s:
        return PLACEBO_KEY
    s = _PAREN.sub(" ", s)
    s = _DOSE.sub(" ", s)
    s = _FORM_WORDS.sub(" ", s)
    s = _NON_WORD.sub(" ", s)
    s = _WS.sub(" ", s).strip(" -/+")
    return s or name.lower().strip()


def intervention_key(
    name: str,
    other_names: list[str] | None,
    mesh_terms: list[str] | None,
    aliases: dict[str, str] | None = None,
) -> str:
    """Canonical key for one intervention entry of one trial.

    Resolution order (most to least reliable):
      1. caller-supplied aliases (the searched drug term matched by name/otherNames)
      2. a MeSH intervention term of the same trial that appears inside the name
      3. regex-cleaned name
    """
    low = name.lower()
    if "placebo" in low:
        return PLACEBO_KEY
    if aliases:
        candidates = [low] + [o.lower() for o in (other_names or [])]
        for needle, canonical in aliases.items():
            if any(_word_in(needle, c) for c in candidates):
                return canonical
    if mesh_terms:
        hits = [m.lower() for m in mesh_terms if len(m) > 3 and _word_in(m.lower(), low)]
        if hits:
            return max(hits, key=len)
    return clean_intervention_name(name)


@lru_cache(maxsize=4096)
def _word_regex(needle: str) -> re.Pattern[str]:
    # hyphens count as word characters: "paclitaxel" must not match "nab-paclitaxel"
    return re.compile(r"(?<![\w-])" + re.escape(needle) + r"(?![\w-])")


def _word_in(needle: str, haystack: str) -> bool:
    return bool(_word_regex(needle).search(haystack))


# ---------------------------------------------------------------- sponsors ---

_CORP_SUFFIX = re.compile(
    r"[,.]?\s*\b(?:inc|incorporated|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|gmbh|ag|sa|s\.a|"
    r"plc|bv|b\.v|nv|kk|spa|s\.p\.a|pty|lp|ab|as|oy|srl)\.?\s*$",
    re.IGNORECASE,
)


def sponsor_key(name: str) -> str:
    s = name.lower().replace("&", " and ").strip()
    prev = None
    while prev != s:  # strip stacked suffixes: "Foo Co., Ltd."
        prev = s
        s = _CORP_SUFFIX.sub("", s).rstrip(" ,.")
    s = _NON_WORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


# ------------------------------------------------------------------ generic ---


def text_key(value: str) -> str:
    return _WS.sub(" ", value.strip().lower())


def year_of(date_str: str | None) -> int | None:
    """API dates are 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD'."""
    if not date_str or len(date_str) < 4 or not date_str[:4].isdigit():
        return None
    return int(date_str[:4])


def months_between(start: str | None, end: str | None) -> float | None:
    def parts(d: str) -> tuple[int, int, int] | None:
        bits = d.split("-")
        try:
            y = int(bits[0])
            m = int(bits[1]) if len(bits) > 1 else 6
            day = int(bits[2]) if len(bits) > 2 else 15
            return y, m, day
        except (ValueError, IndexError):
            return None

    if not start or not end:
        return None
    a, b = parts(start), parts(end)
    if not a or not b:
        return None
    months = (b[0] - a[0]) * 12 + (b[1] - a[1]) + (b[2] - a[2]) / 30.0
    return round(months, 1) if months >= 0 else None
