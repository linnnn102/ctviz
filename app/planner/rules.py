"""Deterministic, keyword-based planner.

Used when no LLM is configured, when the caller sets use_llm=false, or when
the LLM call fails / returns something unusable. It is intentionally
conservative: it only extracts entities it can locate in the text, and
leaves anything ambiguous to the structured request fields.
"""

from __future__ import annotations

import re

from app.plan import Cohort, NetworkSpec, QueryPlan, SearchFilters
from app.planner import lexicon as lx
from app.schemas import VisualizeRequest

_YEAR = r"((?:19|20)\d{2})"


def _find_dims(text: str, patterns: list[tuple[str, str]]) -> list[str]:
    """Dimensions mentioned in `text`, in order of first appearance, de-duplicated."""
    hits: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for pat, dim in patterns:
        for m in re.finditer(pat, text):
            if any(a <= m.start() < b for a, b in taken):
                continue  # already claimed by a more specific pattern
            hits.append((m.start(), dim))
            taken.append((m.start(), m.end()))
    seen, ordered = set(), []
    for _, dim in sorted(hits):
        if dim not in seen:
            seen.add(dim)
            ordered.append(dim)
    return ordered


def classify_entity(phrase: str) -> str:
    """Guess which search area a free-text entity belongs to."""
    if lx.CONDITION_HINT.search(phrase):
        return "condition"
    if lx.DRUG_HINT.search(phrase):
        return "intervention"
    return "term"


def _clean_phrase(phrase: str) -> str | None:
    words = [w for w in re.split(r"\s+", phrase.strip(" ?.,;:!\"'")) if w]
    # trim stopwords at both ends; keep inner ones ("non small cell lung cancer")
    while words and words[0].lower() in lx.STOPWORDS:
        words.pop(0)
    while words and words[-1].lower() in lx.STOPWORDS:
        words.pop()
    out = " ".join(words).strip(" ?.,;:!\"'")
    if not out or re.fullmatch(_YEAR, out) or len(out) < 3:
        return None
    return out


def _extract_years(text: str) -> tuple[int | None, int | None]:
    if m := re.search(rf"between\s+{_YEAR}\s+and\s+{_YEAR}|{_YEAR}\s*(?:-|–|to)\s*{_YEAR}", text):
        a, b = [int(g) for g in m.groups() if g]
        return min(a, b), max(a, b)
    lo = hi = None
    if m := re.search(rf"(?:since|from|after|starting|beginning)\s+(?:in\s+)?{_YEAR}", text):
        lo = int(m.group(1)) + (1 if "after" in m.group(0) else 0)
    if m := re.search(rf"(?:before|until|through|up to|prior to)\s+{_YEAR}", text):
        hi = int(m.group(1)) - (1 if re.match(r"before|prior", m.group(0)) else 0)
    if lo is None and hi is None and (m := re.search(rf"\bin\s+{_YEAR}\b", text)):
        lo = hi = int(m.group(1))
    if m := re.search(r"(?:last|past)\s+(\d{1,2})\s+years", text):
        from datetime import date

        lo = date.today().year - int(m.group(1))
    return lo, hi


def _extract_country(text: str) -> str | None:
    for c in lx.COUNTRIES:
        if re.search(r"\b" + re.escape(c.lower()) + r"\b", text):
            return c
    for alias, c in lx.COUNTRY_ALIASES.items():
        if re.search(r"(?<![\w.])" + re.escape(alias) + r"(?![\w])", text):
            return c
    return None


def _extract_listed(text: str, patterns: list[tuple[str, list[str]]]) -> list[str] | None:
    out: list[str] = []
    for pat, vals in patterns:
        if re.search(pat, text):
            out.extend(v for v in vals if v not in out)
            if pat.startswith(r"phase\s*(?:1|i)\s*/") or pat.startswith(r"phase\s*(?:2|ii)\s*/"):
                break
    return out or None


def _comparison_pair(original: str) -> list[str] | None:
    """'... Drug A vs Drug B ...' / 'between A and B' -> ['Drug A', 'Drug B']."""
    m = re.search(r"(.+?)\s+(?:vs\.?|versus|compared\s+(?:to|with)|against)\s+(.+)", original, re.IGNORECASE)
    if not m:
        m = re.search(r"between\s+(.+?)\s+and\s+(.+)", original, re.IGNORECASE)
        if not m or re.match(_YEAR, m.group(1)):
            return None
    left = re.split(r"\b(?:involving|for|of|between|with|compare|comparing|trials?|in)\b", m.group(1),
                    flags=re.IGNORECASE)[-1]
    right = re.split(r"\b(?:trials?|studies|in|for|by|across|over|since|per|from|during)\b|[?,.;]", m.group(2),
                     flags=re.IGNORECASE)[0]
    a, b = _clean_phrase(left), _clean_phrase(right)
    return [a, b] if a and b else None


def _main_entity(original: str, dims_words: re.Pattern[str]) -> str | None:
    """Best-effort subject entity: 'for X trials', 'involving X', 'X trials'."""
    pats = [
        r"\b(?:involving|testing|evaluating|using|of|for|on|with|about)\s+(?:the\s+)?(?:drug|condition|disease)?\s*"
        r"([A-Za-z0-9][\w\-' ]{2,60}?)\s+(?:trials?|studies)\b",
        r"\b(?:involving|testing|evaluating|using|for|with|about)\s+(?:the\s+)?(?:drug\s+|condition\s+|disease\s+)?"
        r"([A-Za-z0-9][\w\-']*(?:\s+[A-Za-z0-9][\w\-']*){0,4}?)(?=\s+(?:since|from|between|in|by|per|over|across|"
        r"changed|trials?|studies|vs|versus)\b|[?.,]|$)",
        r"\b(?:are|do|have)\s+([A-Za-z0-9][\w\-' ]{2,60}?)\s+(?:trials?|studies)\b",
        r"^([A-Za-z0-9][\w\-' ]{2,60}?)\s+(?:trials?|studies)\b",
    ]
    for pat in pats:
        for m in re.finditer(pat, original, re.IGNORECASE):
            # "enrollment sizes for phase 3 diabetes" -> "phase 3 diabetes": keep the last prepositional segment
            seg = re.split(r"\b(?:for|of|in|with|on|involving|about)\b", m.group(1), flags=re.IGNORECASE)[-1]
            cand = _clean_phrase(seg)
            if not cand:
                continue
            low = cand.lower()
            if dims_words.fullmatch(low) or _extract_country(low) or re.fullmatch(r"(?:this|that|the)\s+\w+", low):
                continue
            # drop leading dimension words: "recruiting melanoma" -> "melanoma"
            cand = re.sub(r"^(?:recruiting|completed|active|terminated|ongoing|new|phase\s*\w+)\s+", "", cand,
                          flags=re.IGNORECASE)
            if _clean_phrase(cand) and not dims_words.fullmatch(cand.lower()):
                return cand
    return None


_DIM_WORDS = re.compile(
    r"(?:the\s+)?(?:phases?|status(?:es)?|sponsors?|drugs?|conditions?|countr(?:y|ies)|interventions?|"
    r"intervention types?|sites?|investigators?|enrol+ment|recruiting|completed|all|different|various|two|"
    r"sponsor categories|types?)"
)


def plan_with_rules(req: VisualizeRequest) -> QueryPlan:
    original = req.query.strip()
    text = original.lower()
    notes: list[str] = []

    numeric = _find_dims(text, lx.NUMERIC_PATTERNS)
    categorical = _find_dims(text, lx.DIMENSION_PATTERNS)

    # ---- intent -----------------------------------------------------------
    is_network = bool(re.search(r"network|co-?occur|↔|<->|<=>|relationship|connected|linked|graph|"
                                r"collaborat|work together|partner", text))
    if is_network:
        intent = "network"
    elif len(numeric) >= 2 or (numeric and re.search(r"scatter|correlat|relationship|against|\bvs\b|versus", text)
                               and len(numeric) >= 2):
        intent = "scatter"
    elif numeric and re.search(r"histogram|distribution|distributed|spread|how (?:large|big|long)|range|typical|"
                               r"sizes?", text):
        intent = "histogram"
    elif re.search(r"over time|per year|each year|by year|yearly|annual|trend|timeline|since (?:19|20)\d{2}|"
                   r"changed|growth|evolv|history|started each|over the (?:years|decade)", text):
        intent = "time_trend"
    elif re.search(r"^(?:list|show me the trials|which trials|what trials|find trials)", text):
        intent = "list"
    elif re.match(r"how many", text) and not categorical and not re.search(r"\b(?:each|per|by)\b", text):
        intent = "count"
    else:
        intent = "distribution"

    # ---- filters ----------------------------------------------------------
    filters = SearchFilters()
    lo, hi = _extract_years(text)
    filters.start_year_min, filters.start_year_max = lo, hi
    filters.country = _extract_country(text)
    filters.statuses = _extract_listed(text, lx.STATUS_PATTERNS)
    # specific phases ("phase 3") are filters; the generic word "phases" is a grouping (see lexicon)
    filters.phases = _extract_listed(text, lx.PHASE_PATTERNS)
    if re.search(r"\bobservational\b", text) and "study_type" not in categorical:
        filters.study_type = "OBSERVATIONAL"

    cohorts: list[Cohort] = []
    pair = None
    if intent not in {"network", "scatter"} and re.search(r"\bvs\.?\b|versus|compar|against|between", text):
        pair = _comparison_pair(original)
    if pair:
        for name in pair:
            area = classify_entity(name)
            cohorts.append(Cohort(label=name, filters=SearchFilters(**{area: name})))
    else:
        if entity := _main_entity(original, _DIM_WORDS):
            setattr(filters, classify_entity(entity), entity)

    # ---- dimensions -------------------------------------------------------
    plan = QueryPlan(intent=intent, filters=filters, cohorts=cohorts)
    if intent == "network":
        ents = [d for d in categorical if d in {"sponsor", "drug", "intervention", "condition", "country",
                                                 "site", "investigator"}]
        if re.search(r"\b(\w{3,}?)s?\s*(?:↔|<->|<=>|-|–|and|to|with)\s*\1s?\b", text) and ents:
            ents = [ents[0], ents[0]]  # "drug ↔ drug", "sponsors with sponsors"
        elif re.search(r"co-?occur|combination|together|paired", text) and len(set(ents)) <= 1:
            ents = [ents[0], ents[0]] if ents else ["drug", "drug"]
        src = ents[0] if ents else "drug"
        tgt = ents[1] if len(ents) > 1 else src
        plan.network = NetworkSpec(source=src, target=tgt)
    elif intent == "scatter":
        plan.y, plan.x = numeric[0], numeric[1]  # "Y vs X" convention
    elif intent == "histogram":
        plan.group_by = numeric[0]
    elif intent == "time_trend":
        temporal = _find_dims(text, lx.TEMPORAL_PATTERNS)
        plan.group_by = temporal[0] if temporal else "start_year"
        split = [d for d in categorical if d not in {"drug", "intervention"} or "by drug" in text]
        if split and not cohorts and re.search(r"\bby\s+(?:" + "|".join(["phase", "status", "sponsor", "countr",
                                                                         "intervention", "condition", "type"]) + ")",
                                               text):
            plan.series_by = split[0]
    elif intent == "distribution":
        cands = [d for d in categorical if not (d == "drug" and re.search(r"this drug|the drug", text))]
        plan.group_by = cands[0] if cands else "phase"
        if plan.group_by == "sponsor" and re.search(r"categor|class|type", text):
            plan.group_by = "sponsor_class"

    if re.search(r"compar", text) and not cohorts and intent != "network":
        notes.append("comparison requested but the compared entities could not be identified from text; "
                     "pass them in the `compare` field")

    plan.rationale = (
        f"Rule-based interpretation: intent={intent}"
        + (f", group_by={plan.group_by}" if plan.group_by else "")
        + (f", series_by={plan.series_by}" if plan.series_by else "")
        + (f", network={plan.network.source}<->{plan.network.target}" if plan.network else "")
        + (f", cohorts={[c.label for c in cohorts]}" if cohorts else "")
        + ("; " + "; ".join(notes) if notes else "")
    )
    return plan
