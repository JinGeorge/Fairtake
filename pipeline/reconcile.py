"""Reconcile stage (spec §4.5): combine the two coding passes into final results.

Exclusions first, per response (new in v2.3):

| Pass A        | Pass B        | Final                                     | Status    |
|---------------|---------------|-------------------------------------------|-----------|
| excluded      | excluded      | excluded (pass A's reason)                | agreed    |
| excluded      | labelled      | excluded; pass B's labels as the `alt`    | contested |
| labelled      | excluded      | labelled, every label contested; pass B's | contested |
|               |               | exclusion kept in ``alt_exclusions``      |           |
| excluded      | nothing valid | excluded, no `alt`                        | contested |
| nothing valid | excluded      | excluded (pass B's reason), no `alt`      | contested |

Then stances, per labelled response and sub-issue:

| Pass A  | Pass B                  | Final label                        | Status     |
|---------|-------------------------|------------------------------------|------------|
| valid   | valid, same stance      | that stance                        | agreed     |
| valid   | valid, different stance | pass A's stance, pass B's as `alt` | contested  |
| valid   | dropped                 | pass A's stance, no `alt`          | contested  |
| dropped | valid                   | pass B's stance, no `alt`          | contested  |
| dropped | dropped                 | `other`                            | ambiguous  |

The per-response group is reconciled with the same table, independently of the
stance (`alt_group`, `group_status`); both dropped gives group ``none``.

Without a second pass (``double_coding=false``) every valid pass-A result is
marked ``agreed``; ``count_poll(double_coded=False)`` then ignores the status.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from schema.report import NO_GROUP, OTHER, FinalExclusion, FinalLabel, Label

LabelsById = Mapping[str, Mapping[str, Label]]


@dataclass
class Reconciled:
    labels: dict[str, dict[str, FinalLabel]]  # responses that are counted (not excluded)
    pass_agreement: float | None  # share of stance labels that are agreed; None for one pass
    ambiguous: int
    exclusions: dict[str, FinalExclusion] = field(default_factory=dict)
    alt_exclusions: dict[str, str] = field(default_factory=dict)  # labelled ids pass B excluded
    group_agreement: float | None = None
    exclusion_agreement: float | None = None


def _status(agree: bool, two_passes: bool) -> str:
    return "agreed" if agree or not two_passes else "contested"


def _final(a: Label | None, b: Label | None, two_passes: bool) -> FinalLabel:
    if a and b:
        same_group = a.group == b.group
        return FinalLabel(
            stance=a.stance,
            status="agreed" if a.stance == b.stance else "contested",
            alt=None if a.stance == b.stance else b.stance,
            group=a.group,
            group_status="agreed" if same_group else "contested",
            alt_group=None if same_group else b.group,
        )
    if a:
        status = _status(False, two_passes)
        return FinalLabel(stance=a.stance, status=status, group=a.group, group_status=status)
    if b:
        return FinalLabel(stance=b.stance, status="contested", group=b.group, group_status="contested")
    return FinalLabel(stance=OTHER, status="ambiguous", group=NO_GROUP, group_status="ambiguous")


def _contested(label: FinalLabel) -> FinalLabel:
    """Mark a label contested because pass B excluded the whole response."""
    if label.status == "ambiguous":
        return label
    return label.model_copy(update={"status": "contested", "group_status": "contested"})


def reconcile(
    ids: Sequence[str],
    sub_ids: Sequence[str],
    pass_a: LabelsById,
    pass_b: LabelsById | None,
    excluded_a: Mapping[str, str] | None = None,
    excluded_b: Mapping[str, str] | None = None,
) -> Reconciled:
    """Reconcile every kept response.

    ``excluded_a`` / ``excluded_b`` are each pass's valid exclusion votes. An id
    in neither the labels nor the exclusions of a pass has nothing valid there.
    """
    two_passes = pass_b is not None
    ex_a = dict(excluded_a or {})
    ex_b = dict(excluded_b or {}) if two_passes else {}
    labels: dict[str, dict[str, FinalLabel]] = {}
    exclusions: dict[str, FinalExclusion] = {}
    alt_exclusions: dict[str, str] = {}
    same_decision = 0

    for rid in ids:
        a_labels = pass_a.get(rid, {})
        b_labels = pass_b.get(rid, {}) if pass_b is not None else {}
        a_out, b_out = ex_a.get(rid), ex_b.get(rid)
        same_decision += bool((a_out and b_out) or (a_labels and b_labels))

        if a_out:
            if not two_passes or b_out:
                exclusions[rid] = FinalExclusion(reason=a_out, status="agreed")
            elif b_labels:
                alt = {sub: _final(None, b_labels.get(sub), True) for sub in sub_ids}
                exclusions[rid] = FinalExclusion(reason=a_out, status="contested", alt=alt)
            else:
                exclusions[rid] = FinalExclusion(reason=a_out, status="contested")
            continue
        if b_out and not a_labels:
            exclusions[rid] = FinalExclusion(reason=b_out, status="contested")
            continue

        per_sub = {sub: _final(a_labels.get(sub), b_labels.get(sub), two_passes) for sub in sub_ids}
        if b_out:
            per_sub = {sub: _contested(label) for sub, label in per_sub.items()}
            alt_exclusions[rid] = b_out
        labels[rid] = per_sub

    finals = [label for per_sub in labels.values() for label in per_sub.values()]
    total = len(finals)
    ambiguous = sum(label.status == "ambiguous" for label in finals)
    if two_passes:
        pass_agreement = sum(label.status == "agreed" for label in finals) / total if total else 1.0
        group_agreement = sum(label.group_status == "agreed" for label in finals) / total if total else 1.0
        exclusion_agreement = same_decision / len(ids) if ids else 1.0
    else:
        pass_agreement = group_agreement = exclusion_agreement = None
    return Reconciled(
        labels=labels,
        pass_agreement=pass_agreement,
        ambiguous=ambiguous,
        exclusions=exclusions,
        alt_exclusions=alt_exclusions,
        group_agreement=group_agreement,
        exclusion_agreement=exclusion_agreement,
    )


def exclusion_agreement_with(proposed: Mapping[str, str], final: Mapping[str, FinalExclusion],
                             ids: Sequence[str]) -> float:
    """Share of kept responses where Call 1's proposal matches the final in/out decision."""
    if not ids:
        return 1.0
    return sum((rid in proposed) == (rid in final) for rid in ids) / len(ids)
