"""Reconcile stage (spec §4.5): combine the two coding passes into final labels.

| Pass A  | Pass B                  | Final label                        | Status     |
|---------|-------------------------|------------------------------------|------------|
| valid   | valid, same stance      | that stance                        | agreed     |
| valid   | valid, different stance | pass A's stance, pass B's as `alt` | contested  |
| valid   | dropped                 | pass A's stance, no `alt`          | contested  |
| dropped | valid                   | pass B's stance, no `alt`          | contested  |
| dropped | dropped                 | `other`                            | ambiguous  |

Without a second pass (``double_coding=false``) every valid pass-A label is
marked ``agreed``; ``count_poll(double_coded=False)`` then ignores the status.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from schema.report import OTHER, FinalLabel, Label

LabelsById = Mapping[str, Mapping[str, Label]]


@dataclass
class Reconciled:
    labels: dict[str, dict[str, FinalLabel]]
    pass_agreement: float | None  # share of labels that are agreed; None for one pass
    ambiguous: int


def _final(a: Label | None, b: Label | None, two_passes: bool) -> FinalLabel:
    if a and b:
        if a.stance == b.stance:
            return FinalLabel(stance=a.stance, status="agreed")
        return FinalLabel(stance=a.stance, status="contested", alt=b.stance)
    if a:
        return FinalLabel(stance=a.stance, status="contested" if two_passes else "agreed")
    if b:
        return FinalLabel(stance=b.stance, status="contested")
    return FinalLabel(stance=OTHER, status="ambiguous")


def reconcile(
    ids: Sequence[str],
    sub_ids: Sequence[str],
    pass_a: LabelsById,
    pass_b: LabelsById | None,
) -> Reconciled:
    two_passes = pass_b is not None
    labels: dict[str, dict[str, FinalLabel]] = {}
    agreed = total = ambiguous = 0
    for rid in ids:
        for sub in sub_ids:
            a = pass_a.get(rid, {}).get(sub)
            b = pass_b.get(rid, {}).get(sub) if pass_b is not None else None
            final = _final(a, b, two_passes)
            labels.setdefault(rid, {})[sub] = final
            total += 1
            agreed += final.status == "agreed"
            ambiguous += final.status == "ambiguous"
    agreement = (agreed / total if total else 1.0) if two_passes else None
    return Reconciled(labels, agreement, ambiguous)
