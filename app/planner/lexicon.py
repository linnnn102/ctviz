"""Small lexicons used by the rule-based planner and the grounding validator."""

from __future__ import annotations

import re

# Ordered: more specific phrases first (e.g. "intervention type" before "intervention").
DIMENSION_PATTERNS: list[tuple[str, str]] = [
    (r"intervention\s+types?|types? of interventions?|modalit", "intervention_type"),
    (r"sponsor\s+(?:categor|class|type)|types? of sponsors?|industry\s+vs|academic", "sponsor_class"),
    (r"study\s+types?|observational\s+vs|interventional\s+vs", "study_type"),
    (r"primary\s+purpose|purpose", "primary_purpose"),
    (r"allocation|randomi[sz]ed\s+vs", "allocation"),
    (r"results\s+posted|have\s+results|with\s+results", "has_results"),
    (r"phases?\b(?!\s*(?:[1-4]|i{1,3}v?)\b)", "phase"),
    (r"statu(?:s|ses)\b", "status"),
    (r"sponsors?|compan(?:y|ies)|pharma", "sponsor"),
    (r"investigators?|principal|officials?", "investigator"),
    (r"\bsites?\b|hospitals?|centers?|centres?|facilit", "site"),
    (r"countr(?:y|ies)|geograph|\bnations?\b|\bwhere\b", "country"),
    (r"conditions?|diseases?|indications?", "condition"),
    (r"drugs?|medications?|compounds?|therap(?:y|ies)\b", "drug"),
    (r"interventions?", "intervention"),
]

NUMERIC_PATTERNS: list[tuple[str, str]] = [
    (r"enrol+ment|sample\s+size|participants|patients\s+enrolled|trial\s+size", "enrollment"),
    (r"duration|how\s+long|length\s+of", "duration_months"),
    (r"number\s+of\s+sites|site\s+count|how\s+many\s+sites", "site_count"),
    (r"number\s+of\s+arms|arm\s+count|arms\b", "arm_count"),
]

TEMPORAL_PATTERNS: list[tuple[str, str]] = [
    (r"complet", "completion_year"),
    (r"posted|registered|registration", "first_posted_year"),
]

STATUS_PATTERNS: list[tuple[str, list[str]]] = [
    (r"not\s+yet\s+recruiting", ["NOT_YET_RECRUITING"]),
    (r"(?<!not yet )\brecruiting\b|currently\s+enrolling|open\s+trials", ["RECRUITING"]),
    (r"\bcompleted\b", ["COMPLETED"]),
    (r"\bterminated\b", ["TERMINATED"]),
    (r"\bwithdrawn\b", ["WITHDRAWN"]),
    (r"\bsuspended\b", ["SUSPENDED"]),
    (r"\bactive\b(?!\s+ingredient)", ["RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]),
]

PHASE_PATTERNS: list[tuple[str, list[str]]] = [
    (r"early\s+phase\s*(?:1|i)\b", ["EARLY_PHASE1"]),
    (r"phase\s*(?:1|i)\s*/\s*(?:2|ii)\b", ["PHASE1", "PHASE2"]),
    (r"phase\s*(?:2|ii)\s*/\s*(?:3|iii)\b", ["PHASE2", "PHASE3"]),
    (r"phase\s*(?:4|iv)\b", ["PHASE4"]),
    (r"phase\s*(?:3|iii)\b", ["PHASE3"]),
    (r"phase\s*(?:2|ii)\b", ["PHASE2"]),
    (r"phase\s*(?:1|i)\b", ["PHASE1"]),
]

COUNTRIES = [
    "United States", "Canada", "Mexico", "Brazil", "Argentina", "Chile", "Colombia", "Peru",
    "United Kingdom", "Ireland", "France", "Germany", "Italy", "Spain", "Portugal", "Netherlands",
    "Belgium", "Switzerland", "Austria", "Sweden", "Norway", "Denmark", "Finland", "Poland",
    "Czechia", "Hungary", "Romania", "Greece", "Turkey", "Russian Federation", "Ukraine", "Israel",
    "Egypt", "South Africa", "Nigeria", "Kenya", "Saudi Arabia", "Iran", "India", "Pakistan",
    "Bangladesh", "China", "Hong Kong", "Taiwan", "Japan", "Korea, Republic of", "Singapore",
    "Thailand", "Vietnam", "Malaysia", "Indonesia", "Philippines", "Australia", "New Zealand",
]

COUNTRY_ALIASES = {
    "us": "United States", "u.s.": "United States", "usa": "United States", "america": "United States",
    "uk": "United Kingdom", "britain": "United Kingdom", "england": "United Kingdom",
    "south korea": "Korea, Republic of", "korea": "Korea, Republic of", "russia": "Russian Federation",
    "czech republic": "Czechia",
}

DRUG_HINT = re.compile(
    r"(?:mab|nib|tinib|ciclib|parib|stat|pril|sartan|olol|vir|cillin|mycin|cycline|platin|taxel|"
    r"rubicin|tide|gliptin|flozin|azole|prazole|lukast|semide|dronate|lisib|lutamide|vec|leucel|"
    r"mide|zole|pine|done|pam|caine|sone|olone)\b|\bvaccine\b|\binsulin\b|\baspirin\b",
    re.IGNORECASE,
)

CONDITION_HINT = re.compile(
    r"cancer|carcinoma|oma\b|itis\b|disease|syndrome|disorder|diabetes|infection|failure|leuka?emia|"
    r"asthma|copd|hiv|covid|depress|alzheimer|parkinson|obesity|hypertension|stroke|sclerosis|"
    r"an(?:a)?emia|pain|injury|tumou?r|neoplasm|malaria|tuberculosis|hepatitis|influenza|autism|"
    r"schizophrenia|epilepsy|migraine|psoriasis|eczema|lupus|sepsis",
    re.IGNORECASE,
)

STOPWORDS = {
    "trial", "trials", "study", "studies", "clinical", "the", "a", "an", "of", "for", "in", "on", "with",
    "and", "or", "by", "to", "from", "since", "between", "across", "over", "per", "each", "how", "what",
    "which", "who", "where", "when", "are", "is", "has", "have", "been", "do", "does", "most", "common",
    "number", "many", "show", "me", "compare", "distribution", "distributed", "changed", "time",
    "year", "years", "network", "involving", "this", "that", "these", "those", "top", "frequently",
    "co-occur", "cooccur", "occur", "combination", "combinations", "vs", "versus", "trend",
}
