"""Dimension registry: how to pull each analyzable attribute out of a study.

Every extractor returns `Value`s that carry the *exact* API field path and
verbatim text that produced them. Aggregation never invents data; citations
are just these excerpts carried through to the datum they contributed to.

To add a dimension: add its name to vocab.Dimension and register a
`DimensionDef` here. Nothing else needs to change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from app import normalize as nz
from app.vocab import STATUS_LABELS

Kind = Literal["categorical", "temporal", "numeric"]

P = "protocolSection"
ID = f"{P}.identificationModule"
STATUS = f"{P}.statusModule"
DESIGN = f"{P}.designModule"
SPONSOR = f"{P}.sponsorCollaboratorsModule"
ARMS = f"{P}.armsInterventionsModule"
LOCS = f"{P}.contactsLocationsModule"
COND = f"{P}.conditionsModule"


@dataclass(frozen=True, slots=True)
class Value:
    key: Any                              # grouping key (str / int year / float)
    label: str                            # display spelling (raw)
    excerpts: tuple[tuple[str, str], ...]  # ((field_path, verbatim_text), ...)


@dataclass
class ExtractContext:
    # needle (lowercase) -> canonical key; used to fold synonyms of the searched drug
    intervention_aliases: dict[str, str] = field(default_factory=dict)


Extractor = Callable[[dict, ExtractContext], list[Value]]


@dataclass(frozen=True)
class DimensionDef:
    name: str
    title: str
    kind: Kind
    multi: bool                 # can one trial contribute several values?
    api_fields: tuple[str, ...]  # `fields=` pieces needed for extraction
    extract: Extractor
    entity: bool = False        # usable as a network node type
    unit: str | None = None
    order: list[str] | None = None  # fixed ordinal order of labels, if any


# ------------------------------------------------------------- helpers ---


def dig(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _scalar(path: str, label_map: dict[str, str] | None = None) -> Extractor:
    def ex(study: dict, _: ExtractContext) -> list[Value]:
        raw = dig(study, path)
        if raw is None or raw == "":
            return [Value("Unknown", "Unknown", ())]
        text = str(raw)
        label = label_map.get(text, text) if label_map else text
        return [Value(label, label, ((path, text),))]

    return ex


def _year(path: str) -> Extractor:
    def ex(study: dict, _: ExtractContext) -> list[Value]:
        raw = dig(study, path)
        y = nz.year_of(raw)
        return [Value(y, str(y), ((path, raw),))] if y else []

    return ex


def _number(path: str) -> Extractor:
    def ex(study: dict, _: ExtractContext) -> list[Value]:
        raw = dig(study, path)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return [Value(float(raw), str(raw), ((path, str(raw)),))]
        return []

    return ex


# ----------------------------------------------------------- extractors ---


def _phase(study: dict, _: ExtractContext) -> list[Value]:
    path = f"{DESIGN}.phases"
    phases = dig(study, path)
    label = nz.phase_label(phases)
    excerpt = ((path, ", ".join(phases)),) if phases else ()
    return [Value(label, label, excerpt)]


def _interventions(drugs_only: bool) -> Extractor:
    allowed = {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"} if drugs_only else None

    def ex(study: dict, ctx: ExtractContext) -> list[Value]:
        items = dig(study, f"{ARMS}.interventions") or []
        meshes = [m.get("term", "") for m in (dig(study, "derivedSection.interventionBrowseModule.meshes") or [])]
        out: dict[str, Value] = {}
        for i, it in enumerate(items):
            name = it.get("name")
            if not name or (allowed and it.get("type") not in allowed):
                continue
            key = nz.intervention_key(name, it.get("otherNames"), meshes, ctx.intervention_aliases)
            if drugs_only and key == nz.PLACEBO_KEY:
                continue  # placebo is a comparator, not a drug of interest
            if key not in out:  # one value per trial per key; keep first excerpt
                out[key] = Value(key, name, ((f"{ARMS}.interventions[{i}].name", name),))
        return list(out.values())

    return ex


def _intervention_type(study: dict, _: ExtractContext) -> list[Value]:
    items = dig(study, f"{ARMS}.interventions") or []
    out: dict[str, Value] = {}
    for i, it in enumerate(items):
        t = it.get("type")
        if t and t not in out:
            label = t.replace("_", " ").title()
            out[t] = Value(label, label, ((f"{ARMS}.interventions[{i}].type", t),))
    return list(out.values()) or [Value("Unknown", "Unknown", ())]


def _conditions(study: dict, _: ExtractContext) -> list[Value]:
    mpath = "derivedSection.conditionBrowseModule.meshes"
    meshes = dig(study, mpath) or []
    if meshes:
        out = {}
        for i, m in enumerate(meshes):
            term = m.get("term")
            if term and nz.text_key(term) not in out:
                out[nz.text_key(term)] = Value(nz.text_key(term), term, ((f"{mpath}[{i}].term", term),))
        return list(out.values())
    raw = dig(study, f"{COND}.conditions") or []
    out = {}
    for i, c in enumerate(raw):
        k = nz.text_key(c)
        if k and k not in out:
            out[k] = Value(k, c, ((f"{COND}.conditions[{i}]", c),))
    return list(out.values())


def _countries(study: dict, _: ExtractContext) -> list[Value]:
    locs = dig(study, f"{LOCS}.locations") or []
    out: dict[str, Value] = {}
    for i, loc in enumerate(locs):
        c = loc.get("country")
        if c and c not in out:
            out[c] = Value(c, c, ((f"{LOCS}.locations[{i}].country", c),))
    return list(out.values())


def _sites(study: dict, _: ExtractContext) -> list[Value]:
    locs = dig(study, f"{LOCS}.locations") or []
    out: dict[str, Value] = {}
    for i, loc in enumerate(locs):
        f = loc.get("facility")
        if f and nz.text_key(f) not in out:
            out[nz.text_key(f)] = Value(nz.text_key(f), f, ((f"{LOCS}.locations[{i}].facility", f),))
    return list(out.values())


def _sponsor(study: dict, _: ExtractContext) -> list[Value]:
    path = f"{SPONSOR}.leadSponsor.name"
    name = dig(study, path)
    if not name:
        return [Value("unknown", "Unknown", ())]
    return [Value(nz.sponsor_key(name), name, ((path, name),))]


def _investigators(study: dict, _: ExtractContext) -> list[Value]:
    offs = dig(study, f"{LOCS}.overallOfficials") or []
    out: dict[str, Value] = {}
    for i, o in enumerate(offs):
        n = o.get("name")
        if n and nz.text_key(n) not in out:
            out[nz.text_key(n)] = Value(nz.text_key(n), n, ((f"{LOCS}.overallOfficials[{i}].name", n),))
    return list(out.values())


def _has_results(study: dict, _: ExtractContext) -> list[Value]:
    v = study.get("hasResults")
    label = "Has results" if v else "No results posted"
    return [Value(label, label, (("hasResults", str(bool(v)).lower()),))]


def _duration(study: dict, _: ExtractContext) -> list[Value]:
    sp, cp = f"{STATUS}.startDateStruct.date", f"{STATUS}.completionDateStruct.date"
    s, c = dig(study, sp), dig(study, cp)
    m = nz.months_between(s, c)
    if m is None:
        return []
    return [Value(m, f"{m} months", ((sp, s), (cp, c)))]


def _count_of(path: str, title: str) -> Extractor:
    def ex(study: dict, _: ExtractContext) -> list[Value]:
        items = dig(study, path)
        if items is None:
            return []
        n = float(len(items))
        return [Value(n, str(int(n)), ((path, f"{int(n)} {title}"),))]

    return ex


# -------------------------------------------------------------- registry ---

INTERVENTION_FIELDS = ("InterventionName", "InterventionType", "InterventionOtherName", "InterventionMeshTerm")

_DEFS = [
    DimensionDef("phase", "Phase", "categorical", False, ("Phase",), _phase, order=nz.PHASE_ORDER),
    DimensionDef("status", "Overall status", "categorical", False, ("OverallStatus",),
                 _scalar(f"{STATUS}.overallStatus", STATUS_LABELS)),
    DimensionDef("study_type", "Study type", "categorical", False, ("StudyType",),
                 _scalar(f"{DESIGN}.studyType", {"INTERVENTIONAL": "Interventional", "OBSERVATIONAL": "Observational",
                                                 "EXPANDED_ACCESS": "Expanded Access"})),
    DimensionDef("sponsor", "Lead sponsor", "categorical", False, ("LeadSponsorName",), _sponsor, entity=True),
    DimensionDef("sponsor_class", "Sponsor class", "categorical", False, ("LeadSponsorClass",),
                 _scalar(f"{SPONSOR}.leadSponsor.class",
                         {"INDUSTRY": "Industry", "NIH": "NIH", "FED": "Federal (US)", "OTHER_GOV": "Other government",
                          "INDIV": "Individual", "NETWORK": "Network", "OTHER": "Other (academic/other)",
                          "AMBIG": "Ambiguous", "UNKNOWN": "Unknown"})),
    DimensionDef("intervention_type", "Intervention type", "categorical", True, ("InterventionType",),
                 _intervention_type),
    DimensionDef("drug", "Drug", "categorical", True, INTERVENTION_FIELDS, _interventions(True), entity=True),
    DimensionDef("intervention", "Intervention", "categorical", True, INTERVENTION_FIELDS, _interventions(False),
                 entity=True),
    DimensionDef("condition", "Condition", "categorical", True, ("ConditionMeshTerm", "Condition"), _conditions,
                 entity=True),
    DimensionDef("country", "Country", "categorical", True, ("LocationCountry",), _countries, entity=True),
    DimensionDef("site", "Site", "categorical", True, ("LocationFacility",), _sites, entity=True),
    DimensionDef("investigator", "Overall official", "categorical", True, ("OverallOfficialName",), _investigators,
                 entity=True),
    DimensionDef("primary_purpose", "Primary purpose", "categorical", False, ("DesignPrimaryPurpose",),
                 _scalar(f"{DESIGN}.designInfo.primaryPurpose")),
    DimensionDef("allocation", "Allocation", "categorical", False, ("DesignAllocation",),
                 _scalar(f"{DESIGN}.designInfo.allocation")),
    DimensionDef("has_results", "Results posted", "categorical", False, ("HasResults",), _has_results),
    DimensionDef("start_year", "Start year", "temporal", False, ("StartDate",), _year(f"{STATUS}.startDateStruct.date")),
    DimensionDef("completion_year", "Completion year", "temporal", False, ("CompletionDate",),
                 _year(f"{STATUS}.completionDateStruct.date")),
    DimensionDef("first_posted_year", "First posted year", "temporal", False, ("StudyFirstPostDate",),
                 _year(f"{STATUS}.studyFirstPostDateStruct.date")),
    DimensionDef("enrollment", "Enrollment", "numeric", False, ("EnrollmentCount",),
                 _number(f"{DESIGN}.enrollmentInfo.count"), unit="participants"),
    DimensionDef("duration_months", "Planned duration", "numeric", False, ("StartDate", "CompletionDate"), _duration,
                 unit="months"),
    DimensionDef("site_count", "Number of sites", "numeric", False, ("LocationCountry",),
                 _count_of(f"{LOCS}.locations", "locations"), unit="sites"),
    DimensionDef("arm_count", "Number of arms", "numeric", False, ("ArmGroupLabel",),
                 _count_of(f"{ARMS}.armGroups", "arm groups"), unit="arms"),
]

REGISTRY: dict[str, DimensionDef] = {d.name: d for d in _DEFS}

BASE_FIELDS = ("NCTId", "BriefTitle")


def get(name: str) -> DimensionDef:
    return REGISTRY[name]


def by_kind(kind: Kind) -> list[str]:
    return [d.name for d in _DEFS if d.kind == kind]


def entity_dims() -> list[str]:
    return [d.name for d in _DEFS if d.entity]


def catalog_for_prompt() -> str:
    lines = []
    for d in _DEFS:
        tag = d.kind + (", multi-valued" if d.multi else "") + (", network-capable" if d.entity else "")
        lines.append(f"- {d.name}: {d.title} ({tag})")
    return "\n".join(lines)
