"""Score a report against the ground truth (spec §8.3). SCORER ONLY.

Usage::

    python -m eval.score results/run1/poll1_pool_hours.json [more reports...]

Not covered here: ``must_mention`` reasons and summary faithfulness (LLM judge,
``eval/judge.py``), injection resistance (Milestone 2), and stability across
runs (aggregated by ``eval/run_all.py``).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from sklearn.metrics import f1_score

from eval.keys import EXCLUDED, Truth, TruthSubIssue, load_truth
from eval.match import Matching, match_stances, match_sub_issues
from pipeline.config import Settings, load_config
from pipeline.finalize import LeakChecker, user_visible_strings
from pipeline.loader import load_poll
from schema.report import NO_GROUP, NOT_ADDRESSED, OTHER, Report, SubIssueReport

MINORITY_SHARE = 0.15


class IncompleteReportError(ValueError):
    """The report does not place every received response exactly once per sub-issue."""


# ---------------------------------------------------------------------------
# Reading a report's per-response placement
# ---------------------------------------------------------------------------


def _excluded_ids(report: Report) -> set[str]:
    return {rid for ids in report.meta.excluded_ids.values() for rid in ids}


def _members(sub: SubIssueReport) -> dict[str, list[str]]:
    return {theme.stance_id: list(theme.response_ids) for theme in sub.themes}


def placements(report: Report, all_ids: Sequence[str]) -> dict[str, dict[str, str]]:
    """Reported sub-issue id -> response id -> stance id, other, not_addressed or excluded."""
    excluded = _excluded_ids(report)
    out: dict[str, dict[str, str]] = {}
    for sub in report.sub_issues:
        placed: dict[str, str] = {rid: EXCLUDED for rid in excluded}
        groups = [(s, ids) for s, ids in _members(sub).items()]
        groups += [(OTHER, sub.other_ids), (NOT_ADDRESSED, sub.not_addressed_ids)]
        for stance, ids in groups:
            for rid in ids:
                if rid in placed:
                    raise IncompleteReportError(f"{sub.id}: response {rid} is placed twice")
                placed[rid] = stance
        missing = set(all_ids) - set(placed)
        unknown = set(placed) - set(all_ids)
        if missing or unknown:
            raise IncompleteReportError(f"{sub.id}: missing {sorted(missing)}, unknown {sorted(unknown)}")
        out[sub.id] = placed
    return out


# ---------------------------------------------------------------------------
# Per sub-issue scoring
# ---------------------------------------------------------------------------


@dataclass
class _SubScore:
    y_true: list[str]
    y_pred: list[str]
    contested_correct: list[bool]
    agreed_correct: list[bool]
    share_errors: list[float]
    minority_found: list[bool]
    calls: list[dict[str, Any]]
    group_type_correct: list[bool]
    response_group_correct: list[bool]
    matching: Matching | None


def _predicted_view(placed: str, mapping: dict[str, str]) -> str:
    if placed in (EXCLUDED, OTHER, NOT_ADDRESSED):
        return placed
    return mapping.get(placed, f"extra:{placed}")


def _reported_group(rid: str, placed: str, rsub: SubIssueReport) -> str:
    """The respondent's own group if reported, else the group of its theme."""
    if rid in rsub.response_groups:
        return rsub.response_groups[rid]
    if placed in (EXCLUDED, OTHER, NOT_ADDRESSED):
        return EXCLUDED if placed == EXCLUDED else NO_GROUP
    theme_group = {th.stance_id: th.group for th in rsub.themes}
    return theme_group.get(placed) or NO_GROUP


def _score_sub(truth: Truth, tsub: TruthSubIssue, rsub: SubIssueReport | None, placed: dict[str, str] | None,
               n_reported: int) -> _SubScore:
    views = list(tsub.groups)
    matching = match_stances(tsub.labels, _members(rsub), views) if rsub else None
    mapping = matching.mapping if matching else {}
    contested = set(rsub.contested_ids) if rsub else set()

    y_true, y_pred, contested_ok, agreed_ok, response_group_ok = [], [], [], [], []
    for rid in truth.respondent_ids:
        true = tsub.labels[rid]
        pred = _predicted_view(placed[rid], mapping) if placed else "missing"
        alt = truth.alternatives.get((rid, tsub.id))
        accepted = bool(alt and pred in alt.accept)
        if accepted:
            pred = true
        y_true.append(true)
        y_pred.append(pred)
        (contested_ok if rid in contested else agreed_ok).append(pred == true)
        if rsub is not None and placed is not None:
            true_group = (tsub.groups.get(true) if true != NOT_ADDRESSED else None) or NO_GROUP
            response_group_ok.append(accepted or _reported_group(rid, placed[rid], rsub) == true_group)
        else:
            response_group_ok.append(False)

    reported_counts = {view: 0 for view in views}
    if rsub:
        for theme in rsub.themes:
            if theme.stance_id in mapping:
                reported_counts[mapping[theme.stance_id]] += theme.count
    n_true = truth.n_respondents
    share_errors = [
        abs(reported_counts[v] / n_reported - tsub.counts[v] / n_true) if n_reported else tsub.counts[v] / n_true
        for v in views if tsub.counts[v] > 0
    ]
    minority = [
        v in set(mapping.values()) for v in views if 0 < tsub.counts[v] / n_true <= MINORITY_SHARE
    ]

    calls = []
    if rsub:
        r = rsub.stance_result
        leader = mapping.get(r.stance_id) if r.stance_id else None
        calls.append({"level": "stance", "correct": (r.kind, leader) == tsub.expected_stance, "robust": r.robust})
        g = rsub.group_result
        if tsub.expected_group is None:
            calls.append({"level": "group", "correct": g is None, "robust": g.robust if g else None})
        else:
            got = (g.kind, g.group) if g else None
            calls.append({"level": "group", "correct": got == tsub.expected_group, "robust": g.robust if g else None})
    else:
        calls = [{"level": "stance", "correct": False, "robust": None},
                 {"level": "group", "correct": False, "robust": None}]

    group_ok = []
    if rsub:
        theme_group = {t.stance_id: t.group for t in rsub.themes}
        group_ok = [theme_group[s] == tsub.groups[v] for s, v in mapping.items()]

    return _SubScore(y_true, y_pred, contested_ok, agreed_ok, share_errors, minority, calls, group_ok,
                     response_group_ok, matching)


# ---------------------------------------------------------------------------
# Whole report
# ---------------------------------------------------------------------------


def _ratio(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _exclusion(truth: Truth, report: Report, accepted_alternative: set[str]) -> tuple[float, float]:
    """Precision and recall of excluded responses.

    ``accepted_alternative`` holds ids the key excludes but the report placed in an
    accepted alternative; they are left out of both sides.
    """
    predicted = _excluded_ids(report) - accepted_alternative
    actual = set(truth.excluded) - accepted_alternative
    hit = len(predicted & actual)
    precision = hit / len(predicted) if predicted else 1.0
    recall = hit / len(actual) if actual else 1.0
    return precision, recall


def _prominence(report: Report) -> float | None:
    gaps = []
    for sub in report.sub_issues:
        words = [len(t.summary.split()) for t in sub.themes]
        counts = [t.count for t in sub.themes]
        if sum(words) and sum(counts):
            gaps += [abs(w / sum(words) - c / sum(counts)) for w, c in zip(words, counts)]
    return mean(gaps) if gaps else None


def _leakage(truth: Truth, report: Report, settings: Settings) -> dict[str, int]:
    poll = load_poll(truth.poll_id, settings)
    strings = user_visible_strings(report) + [report.overall_summary]
    checker = LeakChecker([r.text for r in poll.responses], poll.question, poll.roster_names or [],
                          settings.pipeline.leak_ngram)
    quoted = sum(1 for s in strings if checker.leaks(s))
    term_hits = sum(
        len(re.findall(rf"(?<!\w){re.escape(term)}(?!\w)", s, flags=re.IGNORECASE))
        for s in strings for term in truth.must_not_terms
    )
    return {"leaky_strings": quoted, "must_not_term_hits": term_hits, "total": quoted + term_hits}


def score_report(report: Report, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_config()
    truth = load_truth(report.poll_id, settings)
    if report.n_received != truth.n_received:
        raise IncompleteReportError(f"report has {report.n_received} responses, key has {truth.n_received}")
    if report.status != "ok":
        return {"poll_id": report.poll_id, "status": report.status, "scored": False}

    all_ids = sorted(set(truth.respondent_ids) | set(truth.excluded))
    placed = placements(report, all_ids)
    rsubs = {s.id: s for s in report.sub_issues}
    pairing = match_sub_issues(
        [(t.id, t.labels, list(t.groups)) for t in truth.sub_issues],
        [(s.id, _members(s)) for s in report.sub_issues],
    )

    subs: dict[str, _SubScore] = {}
    accepted_alternative: set[str] = set()
    for tsub in truth.sub_issues:
        rep_sub = pairing.get(tsub.id)
        subs[tsub.id] = _score_sub(truth, tsub, rsubs.get(rep_sub), placed.get(rep_sub), report.n_respondents)
        matching = subs[tsub.id].matching
        for (rid, alt_sub), alt in truth.alternatives.items():
            if alt_sub == tsub.id and rid in truth.excluded and rep_sub is not None and matching is not None:
                if _predicted_view(placed[rep_sub][rid], matching.mapping) in alt.accept:
                    accepted_alternative.add(rid)

    y_true = [y for s in subs.values() for y in s.y_true]
    y_pred = [y for s in subs.values() for y in s.y_pred]
    classes = sorted(set(y_true))
    calls = [dict(c, sub_issue=sid) for sid, s in subs.items() for c in s.calls]
    robust_known = [c for c in calls if c["robust"] is not None]
    precision, recall = _exclusion(truth, report, accepted_alternative)
    views_of = {t.id: [v for v, c in t.counts.items() if c > 0] for t in truth.sub_issues}
    n_sub_labels = report.n_respondents * max(len(report.sub_issues), 1)

    return {
        "poll_id": truth.poll_id,
        "status": report.status,
        "scored": True,
        "stance_accuracy": _ratio([t == p for t, p in zip(y_true, y_pred)]),
        "stance_macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "share_mae": mean(e for s in subs.values() for e in s.share_errors),
        "minority_recall": _ratio([m for s in subs.values() for m in s.minority_found]),
        "majority_calls_correct": _ratio([c["correct"] for c in calls]),
        "majority_calls": calls,
        "wrong_unflagged": sum(1 for c in calls if not c["correct"] and c["robust"] is not False),
        "flag_rate": _ratio([c["robust"] is False for c in robust_known]),
        "accuracy_agreed": _ratio([x for s in subs.values() for x in s.agreed_correct]),
        "accuracy_contested": _ratio([x for s in subs.values() for x in s.contested_correct]),
        "group_type_accuracy": _ratio([x for s in subs.values() for x in s.group_type_correct]),
        "response_group_accuracy": _ratio([x for s in subs.values() for x in s.response_group_correct]),
        "pass_agreement": report.meta.pass_agreement,
        "group_agreement": report.meta.group_agreement,
        "exclusion_agreement": report.meta.exclusion_agreement,
        "call1_exclusion_agreement": report.meta.call1_exclusion_agreement,
        "contested_exclusions": report.contested_exclusions,
        "group_flags": {s.id: list(s.group_flags) for s in report.sub_issues},
        "exclusion_precision": precision,
        "exclusion_recall": recall,
        "other_rate": sum(len(s.other_ids) for s in report.sub_issues) / n_sub_labels if n_sub_labels else None,
        "missed_views": {sid: s.matching.missed if s.matching else views_of[sid] for sid, s in subs.items()},
        "extra_stances": {sid: s.matching.extra if s.matching else [] for sid, s in subs.items()},
        "split_views": {sid: s.matching.splits if s.matching else [] for sid, s in subs.items()},
        "unmatched_sub_issues": [t.id for t in truth.sub_issues if t.id not in pairing],
        "prominence_gap": _prominence(report),
        "leakage": _leakage(truth, report, settings),
        "replacements": {"leak": len(report.meta.leak_replacements), "privacy": len(report.meta.privacy_replacements)},
        "llm_calls": report.meta.llm_calls,
        "cost_usd": report.meta.cost_usd,
        "latency_s": report.meta.latency_s,
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

_COLUMNS = [
    ("stance_accuracy", "acc"),
    ("stance_macro_f1", "F1"),
    ("share_mae", "MAE"),
    ("minority_recall", "minor"),
    ("majority_calls_correct", "calls"),
    ("wrong_unflagged", "wrong!"),
    ("exclusion_recall", "excl"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_table(scores: Sequence[dict[str, Any]]) -> str:
    header = ["poll"] + [short for _, short in _COLUMNS] + ["leaks"]
    rows = [header]
    for s in scores:
        if not s.get("scored"):
            rows.append([s["poll_id"], f"not scored ({s['status']})"])
            continue
        rows.append([s["poll_id"]] + [_fmt(s[k]) for k, _ in _COLUMNS] + [str(s["leakage"]["total"])])
    widths = [max(len(r[i]) for r in rows if i < len(r)) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)) for r in rows)


def main(argv: Sequence[str] | None = None) -> int:
    paths = list(argv if argv is not None else sys.argv[1:])
    if not paths:
        print(__doc__)
        return 2
    scores = [score_report(Report.model_validate_json(Path(p).read_text(encoding="utf-8"))) for p in paths]
    print(format_table(scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
