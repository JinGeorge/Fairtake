"""Validate stage (spec §4.2 span validation, §4.4 label checks) and redaction.

No LLM calls. Texts passed in are the texts the LLM actually saw: scrubbed, and
for the coding passes also with injection spans removed. Comparisons ignore
case, whitespace and the ``&lt;`` / ``&gt;`` escapes added when wrapping.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from schema.report import NO_GROUP, NOT_ADDRESSED, Label, RedactSpan, Stance

REMOVED = "[removed]"
_WHITESPACE = re.compile(r"\s+")


def unescape(text: str) -> str:
    return text.replace("&lt;", "<").replace("&gt;", ">")


def normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", unescape(text)).strip().casefold()


def evidence_ok(evidence: str, text: str) -> bool:
    ev = normalise(evidence)
    return bool(ev) and ev in normalise(text)


# ---------------------------------------------------------------------------
# Labels from one coding pass
# ---------------------------------------------------------------------------


@dataclass
class PassValidation:
    labels: dict[str, dict[str, Label]]  # valid labels only: id -> sub-issue -> label
    failed_ids: list[str]  # ids with no valid exclusion and at least one missing or invalid label
    problems: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)  # valid exclusion votes: id -> reason


def validate_pass(
    labels: Iterable[Label],
    expected_ids: Sequence[str],
    codebook: Mapping[str, Sequence[Stance]],
    texts: Mapping[str, str],
    exclusions: Mapping[str, str] | None = None,
) -> PassValidation:
    """Check one pass (spec §4.4).

    Every id needs either an exclusion vote with no labels, or exactly one label
    per sub-issue with a known stance, a group and real evidence.
    """
    valid_stances = {sub: {s.id for s in stances} for sub, stances in codebook.items()}
    expected = set(expected_ids)
    seen: dict[tuple[str, str], list[Label]] = {}
    labelled: set[str] = set()
    problems: list[str] = []

    for label in labels:
        if label.id not in expected:
            problems.append(f"unexpected id {label.id!r}")
        elif label.sub_issue not in valid_stances:
            problems.append(f"{label.id}: unknown sub-issue {label.sub_issue!r}")
            labelled.add(label.id)
        else:
            seen.setdefault((label.id, label.sub_issue), []).append(label)
            labelled.add(label.id)

    excluded: dict[str, str] = {}
    for rid, reason in (exclusions or {}).items():
        if rid not in expected:
            problems.append(f"unexpected excluded id {rid!r}")
        elif rid in labelled:
            problems.append(f"{rid}: both excluded and labelled")
        else:
            excluded[rid] = reason

    valid: dict[str, dict[str, Label]] = {}
    failed: list[str] = []
    for rid in expected_ids:
        if rid in excluded:
            continue
        if rid in (exclusions or {}):  # excluded and labelled: the whole id fails
            failed.append(rid)
            continue
        all_ok = True
        for sub in codebook:
            problem = _check_one(rid, sub, seen.get((rid, sub), []), valid_stances[sub], texts[rid])
            if problem:
                problems.append(problem)
                all_ok = False
            else:
                valid.setdefault(rid, {})[sub] = seen[(rid, sub)][0]
        if not all_ok:
            failed.append(rid)
    return PassValidation(valid, failed, problems, excluded)


def _check_one(rid: str, sub: str, candidates: list[Label], stances: set[str], text: str) -> str | None:
    if len(candidates) != 1:
        return f"{rid}/{sub}: expected 1 label, got {len(candidates)}"
    label = candidates[0]
    if label.stance not in stances:
        return f"{rid}/{sub}: unknown stance {label.stance!r}"
    if label.group is None:
        return f"{rid}/{sub}: group is missing"
    if label.stance == NOT_ADDRESSED and label.group != NO_GROUP:
        return f"{rid}/{sub}: not_addressed must have group none"
    if label.evidence.strip():
        if not evidence_ok(label.evidence, text):
            return f"{rid}/{sub}: evidence is not in the response"
    elif label.stance != NOT_ADDRESSED:
        return f"{rid}/{sub}: evidence is empty"
    return None


def merge_retry(first: PassValidation, retry: PassValidation, sub_ids: Sequence[str]) -> PassValidation:
    """Fill gaps in the first attempt with valid results from the retry call.

    A retried id that the retry excludes is excluded, and its partial labels
    from the first attempt are dropped, so an id is never both.
    """
    labels = {rid: dict(subs) for rid, subs in first.labels.items()}
    excluded = dict(first.excluded)
    retried = set(first.failed_ids)
    for rid, reason in retry.excluded.items():
        if rid in retried:
            excluded[rid] = reason
            labels.pop(rid, None)
    for rid, subs in retry.labels.items():
        if rid in excluded:
            continue
        for sub, label in subs.items():
            labels.setdefault(rid, {}).setdefault(sub, label)
    failed = [
        rid for rid in first.failed_ids if rid not in excluded and len(labels.get(rid, {})) < len(sub_ids)
    ]
    return PassValidation(labels, failed, first.problems + retry.problems, excluded)


# ---------------------------------------------------------------------------
# Redaction spans
# ---------------------------------------------------------------------------


def span_pattern(span: str) -> re.Pattern[str] | None:
    words = unescape(span).split()
    if not words:
        return None
    return re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)


@dataclass
class SpanCheck:
    spans: dict[str, list[RedactSpan]]  # spans that were located
    withheld: list[str]  # ids with an unlocatable span: text kept out of Call 3
    problems: list[str] = field(default_factory=list)


def validate_spans(redact_spans: Mapping[str, Sequence[RedactSpan]], texts: Mapping[str, str]) -> SpanCheck:
    located: dict[str, list[RedactSpan]] = {}
    withheld: list[str] = []
    problems: list[str] = []
    for rid, spans in redact_spans.items():
        if rid not in texts:
            problems.append(f"redaction for unknown id {rid!r}")
            continue
        good = []
        for span in spans:
            pattern = span_pattern(span.span)
            if pattern and pattern.search(texts[rid]):
                good.append(span)
            else:
                problems.append(f"{rid}: span not found ({span.reason})")
        if good:
            located[rid] = good
        if len(good) < len(spans):
            withheld.append(rid)
    return SpanCheck(located, withheld, problems)


def apply_redactions(text: str, spans: Sequence[RedactSpan], reasons: Iterable[str] | None = None) -> str:
    """Replace each span with ``[removed]``; optionally only spans with given reasons."""
    allowed = set(reasons) if reasons is not None else None
    for span in spans:
        if allowed is not None and span.reason not in allowed:
            continue
        pattern = span_pattern(span.span)
        if pattern:
            text = pattern.sub(REMOVED, text)
    return text


def redaction_counts(spans: Mapping[str, Sequence[RedactSpan]]) -> dict[str, int]:
    counts = Counter(span.reason for group in spans.values() for span in group)
    return {reason: counts.get(reason, 0) for reason in ("abuse", "identifying", "injection")}
