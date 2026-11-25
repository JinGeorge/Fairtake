"""Count stage (spec §4.7): exclusions, counts, majority rules, group mismatch, robustness.

Pure functions, no LLM. Shares are compared as exact fractions, so a count that
sits exactly on a threshold (24 of 40 = 0.60) is never misjudged by float error.

Group totals come from each respondent's own reconciled group (v2.3), not from
the stance-to-group mapping. A label without a per-response group (the
baselines) falls back to its stance's group.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod

from pipeline.config import PipelineSettings
from schema.report import (
    GROUP_TYPES,
    KIND_ORDER,
    NO_GROUP,
    NOT_ADDRESSED,
    OTHER,
    RESERVED_STANCES,
    Excluded,
    FinalExclusion,
    FinalLabel,
    GroupResult,
    Kind,
    Stance,
    StanceResult,
    ThemeKind,
)

HALF = Fraction(1, 2)
LEADING_KINDS = frozenset({"majority", "narrow_majority", "plurality"})
TRIAGE_REASONS = ("off_topic", "low_content")


@dataclass(frozen=True)
class SubIssueCount:
    id: str
    stance_counts: dict[str, int]  # every codebook stance, including other / not_addressed
    stance_members: dict[str, list[str]]
    stance_contested: dict[str, int]
    contested_ids: list[str]
    group_totals: dict[str, int]  # for / against / conditional / alternative, from per-response groups
    not_addressed: int
    addressed: int
    other_count: int
    contested_count: int | None  # None without double coding
    stance_result: StanceResult
    group_result: GroupResult | None
    theme_kinds: dict[str, ThemeKind]  # real stances only
    group_mismatch: dict[str, int]  # real stances only: members whose own group differs
    group_flags: list[str]  # stances with at least GROUP_MISMATCH_MIN mismatched members
    response_groups: dict[str, str]  # respondent id -> its group (none when it has none)


@dataclass(frozen=True)
class CountResult:
    status: str  # "ok" | "too_few_responses"
    n_received: int
    n_respondents: int
    excluded: Excluded
    ambiguous_count: int
    respondent_ids: list[str]
    sub_issues: list[SubIssueCount]
    excluded_ids: dict[str, list[str]]  # reason -> ids
    contested_exclusions: int | None = None  # None without double coding


# ---------------------------------------------------------------------------
# Majority rule
# ---------------------------------------------------------------------------


def majority_kind(
    counts: Mapping[str, int], n: int, clear: Fraction, margin: Fraction
) -> tuple[Kind, str | None]:
    """Apply the majority rule to one list of counts.

    Returns the kind and the leading id, or ``None`` when there is no clear leader.
    Ties for the top place never produce a leader.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_id, top = ranked[0] if ranked else (None, 0)
    second = ranked[1][1] if len(ranked) > 1 else 0
    top_share, second_share = Fraction(top, n), Fraction(second, n)

    if top > second:
        if top_share >= clear:
            return "majority", top_id
        if top_share > HALF:
            return "narrow_majority", top_id
        if top_share - second_share >= margin:
            return "plurality", top_id
    return "no_clear_majority", None


def _real_counts(stance_counts: Mapping[str, int]) -> dict[str, int]:
    return {s: c for s, c in stance_counts.items() if s not in RESERVED_STANCES}


def _sorted_kinds(kinds: set[str]) -> list[Kind]:
    return [k for k in KIND_ORDER if k in kinds]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Versions of a response (spec §4.7 robustness check, step 1)
# ---------------------------------------------------------------------------

# A version is None (excluded) or {sub-issue: (stance, group)}.
Version = dict[str, tuple[str, str]] | None


def _stance_group(groups: Mapping[str, str | None], stance: str) -> str:
    return groups.get(stance) or NO_GROUP


def _own_group(label: FinalLabel, groups: Mapping[str, str | None]) -> str:
    return label.group if label.group is not None else _stance_group(groups, label.stance)


def _primary(labels: Mapping[str, FinalLabel], groups: Mapping[str, Mapping[str, str | None]]) -> Version:
    return {sub: (lab.stance, _own_group(lab, groups[sub])) for sub, lab in labels.items()}


def _alternative(labels: Mapping[str, FinalLabel], groups: Mapping[str, Mapping[str, str | None]]) -> Version:
    """Pass B's version: ``alt`` and ``alt_group`` applied where present."""
    out = {}
    for sub, lab in labels.items():
        stance = lab.alt or lab.stance
        if lab.group is None:
            group = _stance_group(groups[sub], stance)
        else:
            group = lab.alt_group or lab.group
        out[sub] = (stance, group)
    return out


def _is_in(version: Version) -> bool:
    return version is not None and any(stance != NOT_ADDRESSED for stance, _ in version.values())


# ---------------------------------------------------------------------------
# Robustness check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Robustness:
    robust: bool
    alt_kinds: list[Kind]


def _enumerate(
    fixed: Mapping[str, int],
    fixed_n: int,
    units: Sequence[Sequence[str | None]],
    base: tuple[Kind, str | None],
    keys: Sequence[str],
    settings: PipelineSettings,
) -> _Robustness:
    """Recompute one result for every combination of unit outcomes.

    ``fixed`` holds the counts from responses with one possible outcome and
    ``fixed_n`` how many of them are counted. Each unit lists its possible
    outcomes: a key (stance or group) to count, ``""`` for counted under a key
    that is not compared (for example ``other``), or ``None`` for not counted.
    The denominator is recomputed for each combination.
    """
    if not units:
        return _Robustness(True, [])
    if prod(len(u) for u in units) > 2 ** settings.sens_max_contested:
        return _Robustness(False, [])
    clear, margin = settings.clear_majority_exact, settings.plurality_margin_exact
    outcomes = {base}
    too_few = False
    for combo in product(*units):
        counts = {k: fixed.get(k, 0) for k in keys}
        n = fixed_n
        for outcome in combo:
            if outcome is None:
                continue
            n += 1
            if outcome in counts:
                counts[outcome] += 1
        if n < settings.min_respondents:
            too_few = True
            continue
        outcomes.add(majority_kind(counts, n, clear, margin))
    robust = len(outcomes) == 1 and not too_few
    return _Robustness(robust, _sorted_kinds({k for k, _ in outcomes} - {base[0]}))


def _stance_outcome(version: Version, sub: str) -> str | None:
    if not _is_in(version):
        return None
    return version[sub][0]  # type: ignore[index]


def _group_outcomes(version: Version, sub: str, groups: Mapping[str, str | None]) -> set[str | None]:
    if not _is_in(version):
        return {None}
    stance, group = version[sub]  # type: ignore[index]
    out: set[str | None] = {group}
    if stance != OTHER:
        out.add(_stance_group(groups, stance))  # a mismatched response may take its stance's group
    return out


def _robustness(
    sub: str,
    groups: Mapping[str, str | None],
    versions: Mapping[str, tuple[Version, Version]],
    base_stance: tuple[Kind, str | None],
    base_group: tuple[Kind, str | None],
    real_stances: Sequence[str],
    settings: PipelineSettings,
) -> tuple[_Robustness, _Robustness]:
    fixed_s: dict[str, int] = {}
    fixed_g: dict[str, int] = {}
    n_s = n_g = 0
    units_s: list[list[str | None]] = []
    units_g: list[list[str | None]] = []
    for primary, alt in versions.values():
        s_opts = sorted({_stance_outcome(primary, sub), _stance_outcome(alt, sub)}, key=str)
        g_opts = sorted(_group_outcomes(primary, sub, groups) | _group_outcomes(alt, sub, groups), key=str)
        for opts, fixed, units, is_stance in ((s_opts, fixed_s, units_s, True), (g_opts, fixed_g, units_g, False)):
            if len(opts) > 1:
                units.append(opts)
                continue
            (only,) = opts
            if only is None:
                continue
            if is_stance:
                n_s += 1
            else:
                n_g += 1
            fixed[only] = fixed.get(only, 0) + 1
    stance_r = _enumerate(fixed_s, n_s, units_s, base_stance, real_stances, settings)
    group_r = _enumerate(fixed_g, n_g, units_g, base_group, ("for", "against"), settings)
    return stance_r, group_r


# ---------------------------------------------------------------------------
# Whole poll
# ---------------------------------------------------------------------------


def _check_inputs(
    n_received: int,
    no_content_ids: Sequence[str],
    kept_ids: Sequence[str],
    exclusions: Mapping[str, FinalExclusion],
    alt_exclusions: Mapping[str, str],
    codebook: Mapping[str, Sequence[Stance]],
    labels: Mapping[str, Mapping[str, FinalLabel]],
) -> list[str]:
    if len(set(no_content_ids) | set(kept_ids)) != n_received or set(no_content_ids) & set(kept_ids):
        raise ValueError("no_content_ids and kept_ids must split the received responses exactly")
    unknown = set(exclusions) - set(kept_ids)
    if unknown:
        raise ValueError(f"exclusions name responses that were not kept: {sorted(unknown)}")

    coded = [rid for rid in kept_ids if rid not in exclusions]
    extra = set(labels) - set(coded)
    if extra:
        raise ValueError(f"labels given for responses that are not coded: {sorted(extra)}")
    stray = set(alt_exclusions) - set(coded)
    if stray:
        raise ValueError(f"alt_exclusions name responses that are not coded: {sorted(stray)}")

    valid = {sub: {s.id for s in stances} for sub, stances in codebook.items()}
    for sub, ids in valid.items():
        if not RESERVED_STANCES <= ids:
            raise ValueError(f"sub-issue {sub!r} is missing other / not_addressed")

    def check(rid: str, per_sub: Mapping[str, FinalLabel] | None) -> None:
        if per_sub is None or set(per_sub) != set(codebook):
            raise ValueError(f"response {rid!r} needs exactly one label per sub-issue")
        for sub, label in per_sub.items():
            if label.stance not in valid[sub]:
                raise ValueError(f"{rid!r}: unknown stance {label.stance!r} in {sub!r}")
            if label.alt is not None and label.alt not in valid[sub]:
                raise ValueError(f"{rid!r}: unknown alt stance {label.alt!r} in {sub!r}")

    for rid in coded:
        check(rid, labels.get(rid))
    for rid, ex in exclusions.items():
        if ex.alt is not None:
            check(rid, ex.alt)
    return coded


def _count_sub_issue(
    sub: str,
    stances: Sequence[Stance],
    respondents: Sequence[str],
    labels: Mapping[str, Mapping[str, FinalLabel]],
    versions: Mapping[str, tuple[Version, Version]],
    n: int,
    settings: PipelineSettings,
    double_coded: bool,
) -> SubIssueCount:
    clear, margin = settings.clear_majority_exact, settings.plurality_margin_exact
    groups = {s.id: s.group for s in stances}
    counts = {s.id: 0 for s in stances}
    members: dict[str, list[str]] = {s.id: [] for s in stances}
    contested = {s.id: 0 for s in stances}
    mismatch = {s.id: 0 for s in stances}
    totals = dict.fromkeys(GROUP_TYPES, 0)
    response_groups: dict[str, str] = {}
    contested_ids: list[str] = []

    for rid in respondents:
        label = labels[rid][sub]
        group = _own_group(label, groups)
        counts[label.stance] += 1
        members[label.stance].append(rid)
        response_groups[rid] = group
        if group in totals:
            totals[group] += 1
        if label.stance != OTHER and group != _stance_group(groups, label.stance):
            mismatch[label.stance] += 1
        if label.status == "contested" or label.group_status == "contested":
            contested[label.stance] += 1
            contested_ids.append(rid)

    real = _real_counts(counts)
    stance_kind, leader = majority_kind(real, n, clear, margin)
    has_group = any(groups.get(s.id) in ("for", "against") for s in stances)
    sides = {"for": totals["for"], "against": totals["against"]}
    group_kind, group_leader = majority_kind(sides, n, clear, margin)

    if double_coded:
        stance_r, group_r = _robustness(
            sub, groups, versions, (stance_kind, leader), (group_kind, group_leader), list(real), settings
        )
        stance_robust, stance_alts = stance_r.robust, stance_r.alt_kinds
        group_robust, group_alts = group_r.robust, group_r.alt_kinds
        contested_count: int | None = sum(contested.values())
    else:
        stance_robust = stance_alts = group_robust = group_alts = None
        contested_count = None

    theme_kinds: dict[str, ThemeKind] = {
        s: (stance_kind if s == leader and stance_kind in LEADING_KINDS else "minority")  # type: ignore[misc]
        for s in real
    }
    real_mismatch = {s: mismatch[s] for s in real}
    flags = [s for s, k in real_mismatch.items() if k >= settings.group_mismatch_min]

    return SubIssueCount(
        id=sub,
        stance_counts=counts,
        stance_members=members,
        stance_contested=contested,
        contested_ids=contested_ids if double_coded else [],
        group_totals=totals,
        not_addressed=counts[NOT_ADDRESSED],
        addressed=n - counts[NOT_ADDRESSED],
        other_count=counts[OTHER],
        contested_count=contested_count,
        stance_result=StanceResult(
            kind=stance_kind, stance_id=leader, robust=stance_robust, alt_kinds=stance_alts
        ),
        group_result=(
            GroupResult(
                kind=group_kind,
                group=group_leader,
                robust=group_robust if has_group else None,
                alt_kinds=group_alts if has_group else None,
                for_=totals["for"],
                against=totals["against"],
                conditional=totals["conditional"],
                alternative=totals["alternative"],
            )
            if has_group
            else None
        ),
        theme_kinds=theme_kinds,
        group_mismatch=real_mismatch,
        group_flags=flags,
        response_groups=response_groups,
    )


def count_poll(
    *,
    n_received: int,
    no_content_ids: Sequence[str],
    kept_ids: Sequence[str],
    codebook: Mapping[str, Sequence[Stance]],
    labels: Mapping[str, Mapping[str, FinalLabel]],
    settings: PipelineSettings,
    exclusions: Mapping[str, FinalExclusion] | None = None,
    alt_exclusions: Mapping[str, str] | None = None,
    double_coded: bool = True,
) -> CountResult:
    """Count a whole poll.

    ``kept_ids`` are the responses that passed Prepare and were sent to the
    coding passes. ``exclusions`` are the reconciled exclusions (§4.5);
    ``labels`` must hold one final label per sub-issue for every other kept
    response. ``alt_exclusions`` names labelled responses that pass B excluded.
    """
    exclusions = dict(exclusions or {})
    alt_exclusions = dict(alt_exclusions or {})
    coded = _check_inputs(n_received, no_content_ids, kept_ids, exclusions, alt_exclusions, codebook, labels)
    groups = {sub: {s.id: s.group for s in stances} for sub, stances in codebook.items()}

    no_position = [rid for rid in coded if all(lab.stance == NOT_ADDRESSED for lab in labels[rid].values())]
    no_position_set = set(no_position)
    respondents = [rid for rid in coded if rid not in no_position_set]

    excluded = Excluded(
        no_content=len(no_content_ids),
        off_topic=sum(1 for e in exclusions.values() if e.reason == "off_topic"),
        low_content=sum(1 for e in exclusions.values() if e.reason == "low_content"),
        no_position=len(no_position),
    )
    excluded_ids = {"no_content": list(no_content_ids)}
    for reason in TRIAGE_REASONS:
        excluded_ids[reason] = [rid for rid in kept_ids if rid in exclusions and exclusions[rid].reason == reason]
    excluded_ids["no_position"] = no_position
    n = n_received - excluded.total
    ambiguous = sum(1 for rid in respondents for lab in labels[rid].values() if lab.status == "ambiguous")
    contested_exclusions = (
        sum(e.status == "contested" for e in exclusions.values()) + len(alt_exclusions) if double_coded else None
    )

    if n < settings.min_respondents:
        return CountResult("too_few_responses", n_received, n, excluded, ambiguous, respondents, [], excluded_ids,
                           contested_exclusions)

    versions: dict[str, tuple[Version, Version]] = {}
    if double_coded:
        for rid in kept_ids:
            if rid in exclusions:
                alt_labels = exclusions[rid].alt
                versions[rid] = (None, _primary(alt_labels, groups) if alt_labels else None)
            else:
                primary = _primary(labels[rid], groups)
                alt = None if rid in alt_exclusions else _alternative(labels[rid], groups)
                versions[rid] = (primary, alt)

    subs = [
        _count_sub_issue(sub, stances, respondents, labels, versions, n, settings, double_coded)
        for sub, stances in codebook.items()
    ]
    return CountResult("ok", n_received, n, excluded, ambiguous, respondents, subs, excluded_ids,
                       contested_exclusions)
