"""Count stage (spec §4.7): exclusions, counts, majority rules, robustness check.

Pure functions, no LLM. Shares are compared as exact fractions, so a count that
sits exactly on a threshold (24 of 40 = 0.60) is never misjudged by float error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from pipeline.config import PipelineSettings
from schema.report import (
    GROUP_TYPES,
    KIND_ORDER,
    NOT_ADDRESSED,
    OTHER,
    RESERVED_STANCES,
    Excluded,
    FinalLabel,
    GroupResult,
    Kind,
    Stance,
    StanceResult,
    ThemeKind,
)

HALF = Fraction(1, 2)
LEADING_KINDS = frozenset({"majority", "narrow_majority", "plurality"})


@dataclass(frozen=True)
class SubIssueCount:
    id: str
    stance_counts: dict[str, int]  # every codebook stance, including other / not_addressed
    stance_members: dict[str, list[str]]
    stance_contested: dict[str, int]
    contested_ids: list[str]
    group_totals: dict[str, int]  # for / against / conditional / alternative
    not_addressed: int
    addressed: int
    other_count: int
    contested_count: int | None  # None without double coding
    stance_result: StanceResult
    group_result: GroupResult | None
    theme_kinds: dict[str, ThemeKind]  # real stances only


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


def _group_totals(stance_counts: Mapping[str, int], groups: Mapping[str, str | None]) -> dict[str, int]:
    totals = dict.fromkeys(GROUP_TYPES, 0)
    for stance, count in stance_counts.items():
        group = groups.get(stance)
        if group is not None:
            totals[group] += count
    return totals


def _for_against(totals: Mapping[str, int]) -> dict[str, int]:
    return {"for": totals["for"], "against": totals["against"]}


def _sorted_kinds(kinds: set[str]) -> list[Kind]:
    return [k for k in KIND_ORDER if k in kinds]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Robustness check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Robustness:
    stance_robust: bool
    stance_alts: list[Kind]
    group_robust: bool
    group_alts: list[Kind]


def robustness(
    stance_counts: Mapping[str, int],
    groups: Mapping[str, str | None],
    swaps: Sequence[tuple[str, str]],
    n: int,
    clear: Fraction,
    margin: Fraction,
    max_contested: int,
) -> _Robustness:
    """Recompute both results for every combination of contested labels.

    ``swaps`` holds one ``(final_stance, alt_stance)`` pair per contested label.
    The denominator ``n`` is held fixed, as the spec says.
    """
    base_stance = majority_kind(_real_counts(stance_counts), n, clear, margin)
    base_group = majority_kind(_for_against(_group_totals(stance_counts, groups)), n, clear, margin)

    if len(swaps) > max_contested:
        return _Robustness(False, [], False, [])

    stance_outcomes, group_outcomes = {base_stance}, {base_group}
    for mask in product((False, True), repeat=len(swaps)):
        counts = dict(stance_counts)
        for use_alt, (final, alt) in zip(mask, swaps):
            if use_alt:
                counts[final] -= 1
                counts[alt] += 1
        stance_outcomes.add(majority_kind(_real_counts(counts), n, clear, margin))
        group_outcomes.add(majority_kind(_for_against(_group_totals(counts, groups)), n, clear, margin))

    return _Robustness(
        stance_robust=len(stance_outcomes) == 1,
        stance_alts=_sorted_kinds({k for k, _ in stance_outcomes} - {base_stance[0]}),
        group_robust=len(group_outcomes) == 1,
        group_alts=_sorted_kinds({k for k, _ in group_outcomes} - {base_group[0]}),
    )


# ---------------------------------------------------------------------------
# Whole poll
# ---------------------------------------------------------------------------


def _check_inputs(
    n_received: int,
    no_content_ids: Sequence[str],
    kept_ids: Sequence[str],
    triage: Mapping[str, str],
    codebook: Mapping[str, Sequence[Stance]],
    labels: Mapping[str, Mapping[str, FinalLabel]],
) -> list[str]:
    if len(set(no_content_ids) | set(kept_ids)) != n_received or set(no_content_ids) & set(kept_ids):
        raise ValueError("no_content_ids and kept_ids must split the received responses exactly")
    unknown = set(triage) - set(kept_ids)
    if unknown:
        raise ValueError(f"triage names responses that were not sent to Call 1: {sorted(unknown)}")

    coded = [rid for rid in kept_ids if rid not in triage]
    extra = set(labels) - set(coded)
    if extra:
        raise ValueError(f"labels given for responses that are not coded: {sorted(extra)}")

    valid = {sub: {s.id for s in stances} for sub, stances in codebook.items()}
    for sub, ids in valid.items():
        if not RESERVED_STANCES <= ids:
            raise ValueError(f"sub-issue {sub!r} is missing other / not_addressed")
    for rid in coded:
        per_sub = labels.get(rid)
        if per_sub is None or set(per_sub) != set(codebook):
            raise ValueError(f"response {rid!r} needs exactly one label per sub-issue")
        for sub, label in per_sub.items():
            if label.stance not in valid[sub]:
                raise ValueError(f"{rid!r}: unknown stance {label.stance!r} in {sub!r}")
            if label.alt is not None and label.alt not in valid[sub]:
                raise ValueError(f"{rid!r}: unknown alt stance {label.alt!r} in {sub!r}")
    return coded


def _count_sub_issue(
    sub: str,
    stances: Sequence[Stance],
    respondents: Sequence[str],
    labels: Mapping[str, Mapping[str, FinalLabel]],
    n: int,
    settings: PipelineSettings,
    double_coded: bool,
) -> SubIssueCount:
    clear, margin = settings.clear_majority_exact, settings.plurality_margin_exact
    groups = {s.id: s.group for s in stances}
    counts = {s.id: 0 for s in stances}
    members: dict[str, list[str]] = {s.id: [] for s in stances}
    contested = {s.id: 0 for s in stances}
    contested_ids: list[str] = []
    swaps: list[tuple[str, str]] = []

    for rid in respondents:
        label = labels[rid][sub]
        counts[label.stance] += 1
        members[label.stance].append(rid)
        if label.status == "contested":
            contested[label.stance] += 1
            contested_ids.append(rid)
            if label.alt is not None and label.alt != label.stance:
                swaps.append((label.stance, label.alt))

    totals = _group_totals(counts, groups)
    stance_kind, leader = majority_kind(_real_counts(counts), n, clear, margin)
    has_group = any(groups.get(s.id) in ("for", "against") for s in stances)
    group_kind, group_leader = majority_kind(_for_against(totals), n, clear, margin)

    if double_coded:
        r = robustness(counts, groups, swaps, n, clear, margin, settings.sens_max_contested)
        stance_robust, stance_alts = r.stance_robust, r.stance_alts
        group_robust, group_alts = r.group_robust, r.group_alts
        contested_count: int | None = sum(contested.values())
    else:
        stance_robust = stance_alts = group_robust = group_alts = None
        contested_count = None

    theme_kinds: dict[str, ThemeKind] = {
        s: (stance_kind if s == leader and stance_kind in LEADING_KINDS else "minority")  # type: ignore[misc]
        for s in _real_counts(counts)
    }

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
                robust=group_robust,
                alt_kinds=group_alts,
                for_=totals["for"],
                against=totals["against"],
                conditional=totals["conditional"],
                alternative=totals["alternative"],
            )
            if has_group
            else None
        ),
        theme_kinds=theme_kinds,
    )


def count_poll(
    *,
    n_received: int,
    no_content_ids: Sequence[str],
    kept_ids: Sequence[str],
    triage: Mapping[str, str],
    codebook: Mapping[str, Sequence[Stance]],
    labels: Mapping[str, Mapping[str, FinalLabel]],
    settings: PipelineSettings,
    double_coded: bool = True,
) -> CountResult:
    """Count a whole poll.

    ``kept_ids`` are the responses that passed Prepare and were sent to Call 1.
    ``labels`` must hold one final label per coded response per sub-issue.
    """
    coded = _check_inputs(n_received, no_content_ids, kept_ids, triage, codebook, labels)

    no_position = [rid for rid in coded if all(lab.stance == NOT_ADDRESSED for lab in labels[rid].values())]
    no_position_set = set(no_position)
    respondents = [rid for rid in coded if rid not in no_position_set]

    excluded = Excluded(
        no_content=len(no_content_ids),
        off_topic=sum(1 for r in triage.values() if r == "off_topic"),
        low_content=sum(1 for r in triage.values() if r == "low_content"),
        no_position=len(no_position),
    )
    excluded_ids = {
        "no_content": list(no_content_ids),
        "off_topic": [rid for rid, r in triage.items() if r == "off_topic"],
        "low_content": [rid for rid, r in triage.items() if r == "low_content"],
        "no_position": no_position,
    }
    n = n_received - excluded.total
    ambiguous = sum(1 for rid in respondents for lab in labels[rid].values() if lab.status == "ambiguous")

    if n < settings.min_respondents:
        return CountResult("too_few_responses", n_received, n, excluded, ambiguous, respondents, [], excluded_ids)

    subs = [
        _count_sub_issue(sub, stances, respondents, labels, n, settings, double_coded)
        for sub, stances in codebook.items()
    ]
    return CountResult("ok", n_received, n, excluded, ambiguous, respondents, subs, excluded_ids)
