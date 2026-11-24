"""Stance matching (spec §8.2).

A system invents its own stance ids (and, for multi-issue polls, its own
sub-issue ids), so before scoring each reported stance is matched one-to-one to
a key view by the number of responses they share (Hungarian algorithm).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

MIN_SPLIT_PART = 2


@dataclass(frozen=True)
class Matching:
    mapping: dict[str, str]  # reported stance -> key view
    missed: list[str]  # key views with members but no matched stance
    extra: list[str]  # reported stances with members but no matched view
    splits: list[str]  # key views spread over 2+ reported stances (>= 2 responses each)
    overlap: int  # responses shared by matched pairs


def match_stances(
    true_labels: Mapping[str, str],
    reported_members: Mapping[str, Sequence[str]],
    views: Sequence[str],
) -> Matching:
    """Match reported stances to key views.

    ``true_labels`` maps respondent id -> key view (other values, such as
    not_addressed, are ignored). ``reported_members`` maps reported stance -> ids.
    """
    present_views = [v for v in views if any(lab == v for lab in true_labels.values())]
    stances = [s for s, ids in reported_members.items() if ids]
    matrix = np.zeros((len(present_views), len(stances)), dtype=int)
    for j, stance in enumerate(stances):
        for rid in reported_members[stance]:
            view = true_labels.get(rid)
            if view in present_views:
                matrix[present_views.index(view), j] += 1

    mapping: dict[str, str] = {}
    overlap = 0
    if matrix.size:
        rows, cols = linear_sum_assignment(matrix, maximize=True)
        for r, c in zip(rows, cols):
            if matrix[r, c] > 0:
                mapping[stances[c]] = present_views[r]
                overlap += int(matrix[r, c])

    matched_views = set(mapping.values())
    splits = [
        v for i, v in enumerate(present_views) if int((matrix[i] >= MIN_SPLIT_PART).sum()) >= 2
    ] if matrix.size else []
    return Matching(
        mapping=mapping,
        missed=[v for v in present_views if v not in matched_views],
        extra=[s for s in stances if s not in mapping],
        splits=splits,
        overlap=overlap,
    )


def match_sub_issues(
    truth_subs: Sequence[tuple[str, Mapping[str, str], Sequence[str]]],
    report_subs: Sequence[tuple[str, Mapping[str, Sequence[str]]]],
) -> dict[str, str]:
    """Pair truth sub-issues with reported ones, maximising matched overlap.

    ``truth_subs``: (id, true_labels, views). ``report_subs``: (id, reported_members).
    Returns truth sub-issue id -> reported sub-issue id; unpaired ones are left out.
    """
    if not truth_subs or not report_subs:
        return {}
    if len(truth_subs) == 1 and len(report_subs) == 1:
        return {truth_subs[0][0]: report_subs[0][0]}
    gain = np.zeros((len(truth_subs), len(report_subs)), dtype=int)
    for i, (_, labels, views) in enumerate(truth_subs):
        for j, (_, members) in enumerate(report_subs):
            gain[i, j] = match_stances(labels, members, views).overlap
    rows, cols = linear_sum_assignment(gain, maximize=True)
    return {truth_subs[r][0]: report_subs[c][0] for r, c in zip(rows, cols)}
