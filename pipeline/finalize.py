"""Finalize stage (spec §4.9) and report assembly.

No LLM calls. The orchestrator uses it like this:

1. ``find_leaks()`` on Call 3's output; if anything leaks, regenerate Call 3 once.
2. ``finalize()`` builds the report, applying code fallbacks to whatever still fails.
3. ``user_visible_strings()`` goes to Call 4; ``finalize()`` runs again with the
   strings Call 4 flagged, so the headline is rebuilt from the replaced labels.

Replacements are recorded by location (``"main.support.summary"``), never by
content, because the content is what failed the check.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence

from pipeline.config import PipelineSettings
from pipeline.count import CountResult, SubIssueCount
from pipeline.prepare import SCRUB_TOKENS
from pipeline.validate import REMOVED
from pipeline.wording import (
    GROUP_FALLBACK,
    group_sentence,
    respondents,
    stance_sentence,
    unsettled_sentence,
)
from schema.report import (
    NOT_ADDRESSED,
    OTHER,
    RESERVED_STANCES,
    CommonGround,
    Excluded,
    Meta,
    Report,
    Stance,
    SubIssue,
    SubIssueReport,
    Theme,
    WriteOutput,
)

_TOKEN = re.compile(r"[a-z0-9]+")
_FORBIDDEN_MARKERS = tuple(t.lower() for t in (*SCRUB_TOKENS, REMOVED))
_MIN_NAME_PART = 3


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def ngrams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class LeakChecker:
    """Flags text that quotes a response or names someone (spec §4.9 step 2)."""

    def __init__(self, response_texts: Iterable[str], question: str, roster_names: Sequence[str], n: int):
        self.n = n
        exempt = ngrams(tokens(question), n)
        self._source: set[tuple[str, ...]] = set()
        for text in response_texts:
            self._source |= ngrams(tokens(text), n)
        self._source -= exempt
        self._names = [tuple(tokens(name)) for name in roster_names if tokens(name)]
        self._name_parts = {w for name in self._names for w in name if len(w) >= _MIN_NAME_PART}

    def leaks(self, text: str) -> bool:
        lowered = text.lower()
        if any(marker in lowered for marker in _FORBIDDEN_MARKERS):
            return True
        words = tokens(text)
        if ngrams(words, self.n) & self._source:
            return True
        if self._name_parts & set(words):
            return True
        joined = " ".join(words)
        return any(" ".join(name) in joined for name in self._names)


def find_leaks(written: WriteOutput, checker: LeakChecker) -> list[str]:
    """Strings in Call 3's output that fail the check, for one regeneration."""
    out = [s for per_sub in written.summaries.values() for s in per_sub.values() if checker.leaks(s)]
    out += [cg.point for points in written.common_ground.values() for cg in points if checker.leaks(cg.point)]
    return out


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


class _Gate:
    """Runs every LLM-written string through the leak check and Call 4's flags."""

    def __init__(self, checker: LeakChecker, flagged: Collection[str], meta: Meta):
        self.checker, self.flagged, self.meta = checker, set(flagged), meta

    def ok(self, location: str, text: str) -> bool:
        if text in self.flagged:
            self.meta.privacy_replacements.append(location)
            return False
        if self.checker.leaks(text):
            self.meta.leak_replacements.append(location)
            return False
        return True


def _common_ground(
    points: Sequence,
    sub: SubIssueCount,
    groups: Mapping[str, str | None],
    gate: _Gate,
    min_group: int,
) -> list[CommonGround]:
    stance_of = {rid: s for s, members in sub.stance_members.items() if s != NOT_ADDRESSED for rid in members}
    use_groups = any(g is not None for s, g in groups.items() if s not in RESERVED_STANCES)
    kept = []
    for i, point in enumerate(points):
        ids = list(dict.fromkeys(point.ids))
        if len(ids) < min_group or any(rid not in stance_of for rid in ids):
            continue
        if use_groups:
            distinct = {groups[stance_of[rid]] for rid in ids} - {None}
        else:
            distinct = {stance_of[rid] for rid in ids}
        if len(distinct) < 2:
            continue
        if gate.ok(f"{sub.id}.common_ground.{i}", point.point):
            kept.append(CommonGround(point=point.point, response_ids=ids))
    return kept


def _sub_issue_report(
    index: int,
    total_subs: int,
    sub_meta: SubIssue,
    stances: Sequence[Stance],
    sub: SubIssueCount,
    written: WriteOutput | None,
    n: int,
    gate: _Gate,
    settings: PipelineSettings,
) -> tuple[SubIssueReport, dict[str, str]]:
    sid = sub.id
    description = sub_meta.description
    if not gate.ok(f"{sid}.description", description):
        description = "The question" if total_subs == 1 else f"Part {index + 1} of the question"

    groups = {s.id: s.group for s in stances}
    labels: dict[str, str] = {}
    definitions: dict[str, str] = {}
    used_labels: set[str] = set()
    for s in stances:
        if s.id in RESERVED_STANCES:
            continue
        fallback = GROUP_FALLBACK[s.group]
        label = s.label if gate.ok(f"{sid}.{s.id}.label", s.label) else fallback
        if label in used_labels:  # e.g. two stances fell back to the same text
            k = 2
            while f"{label} ({k})" in used_labels:
                k += 1
            label = f"{label} ({k})"
        used_labels.add(label)
        labels[s.id] = label
        definitions[s.id] = s.definition if gate.ok(f"{sid}.{s.id}.definition", s.definition) else fallback

    summaries = (written.summaries.get(sid, {}) if written else {})
    themes = []
    for s in stances:
        count = sub.stance_counts.get(s.id, 0)
        if s.id in RESERVED_STANCES or count == 0:
            continue
        summary = summaries.get(s.id)
        if count < settings.min_group:
            summary = definitions[s.id]
        elif summary is None:
            gate.meta.leak_replacements.append(f"{sid}.{s.id}.summary:missing")
            summary = definitions[s.id]
        elif not gate.ok(f"{sid}.{s.id}.summary", summary):
            summary = definitions[s.id]
        themes.append(
            Theme(
                stance_id=s.id,
                label=labels[s.id],
                group=s.group,
                summary=summary,
                count=count,
                of_total=n,
                of_addressed=sub.addressed,
                x_of_y=f"{count} of {n} {respondents(n)}",
                kind=sub.theme_kinds[s.id],
                contested=sub.stance_contested[s.id] if sub.contested_count is not None else None,
                response_ids=list(sub.stance_members[s.id]),
            )
        )
    themes.sort(key=lambda t: (-t.count, t.stance_id))

    points = written.common_ground.get(sid, []) if written else []
    report = SubIssueReport(
        id=sid,
        description=description,
        not_addressed=sub.not_addressed,
        addressed=sub.addressed,
        other_count=sub.other_count,
        contested_count=sub.contested_count,
        stance_result=sub.stance_result,
        group_result=sub.group_result,
        themes=themes,
        common_ground=_common_ground(points, sub, groups, gate, settings.min_group),
        other_ids=list(sub.stance_members[OTHER]),
        not_addressed_ids=list(sub.stance_members[NOT_ADDRESSED]),
        contested_ids=list(sub.contested_ids),
    )
    return report, labels


def headline(
    n_respondents: int,
    n_invited: int | None,
    excluded: Excluded,
    subs: Sequence[SubIssueReport],
    labels: Mapping[str, Mapping[str, str]],
) -> str:
    """The overall summary, built from counts and checked labels only."""
    n = n_respondents
    parts = []
    if n_invited is not None:
        parts.append(f"{n} of {n_invited} invited members gave a usable response.")
    for sub in subs:
        prefix = f"{sub.description}: " if len(subs) > 1 else ""
        listing = "; ".join(f"{t.label}: {t.count} of {n}" for t in sub.themes)
        parts.append(f"{prefix}{listing}." if listing else f"{prefix}no views were expressed.")

        result = sub.stance_result
        lead_id = result.stance_id or (sub.themes[0].stance_id if sub.themes else None)
        lead = labels[sub.id].get(lead_id) if lead_id else None
        if result.robust is False and lead:
            parts.append(unsettled_sentence(lead, sub.contested_count or 0, result.kind, list(result.alt_kinds or [])))
        else:
            parts.append(stance_sentence(result.kind, lead))

        g = sub.group_result
        if g is not None:
            counts = f"{g.for_} for, {g.against} against"
            if g.conditional:
                counts += f", {g.conditional} in favour only with conditions"
            if g.alternative:
                counts += f", {g.alternative} proposing a different approach"
            parts.append(f"In total: {counts}. {group_sentence(g.kind, g.group, g.robust)}")
        if sub.other_count:
            parts.append(f"{sub.other_count} {respondents(sub.other_count)} gave a view that fits none of the groups above.")
        if sub.not_addressed:
            parts.append(f"{sub.not_addressed} did not address this.")
    if excluded.total:
        parts.append(
            f"{excluded.total} {'response was' if excluded.total == 1 else 'responses were'} not counted "
            "(empty, off-topic or without a position)."
        )
    return " ".join(parts)


def too_few_report(
    poll_id: str,
    n_received: int,
    n_respondents: int,
    excluded: Excluded,
    n_invited: int | None,
    meta: Meta | None = None,
) -> Report:
    return Report(
        poll_id=poll_id,
        status="too_few_responses",
        n_invited=n_invited,
        n_received=n_received,
        n_respondents=n_respondents,
        excluded=excluded,
        sub_issues=[],
        overall_summary=(
            f"{n_respondents} usable {'response was' if n_respondents == 1 else 'responses were'} received. "
            "That is too few to report views without risking identifying people."
        ),
        meta=meta or Meta(),
    )


def finalize(
    *,
    poll_id: str,
    question: str,
    count: CountResult,
    sub_issues: Sequence[SubIssue],
    codebook: Mapping[str, Sequence[Stance]],
    written: WriteOutput | None,
    response_texts: Iterable[str],
    roster_names: Sequence[str],
    n_invited: int | None,
    settings: PipelineSettings,
    meta: Meta | None = None,
    privacy_flagged: Collection[str] = (),
) -> Report:
    """Assemble the report.

    ``response_texts`` must include every raw and scrubbed response text, so the
    leakage check can compare against all of them.
    """
    meta = meta.model_copy(deep=True) if meta else Meta()
    meta.excluded_ids = {reason: list(ids) for reason, ids in count.excluded_ids.items()}
    if count.status != "ok":
        return too_few_report(poll_id, count.n_received, count.n_respondents, count.excluded, n_invited, meta)

    checker = LeakChecker(response_texts, question, roster_names, settings.leak_ngram)
    gate = _Gate(checker, privacy_flagged, meta)
    sub_by_id = {s.id: s for s in sub_issues}

    reports, labels = [], {}
    for i, sub in enumerate(count.sub_issues):
        report, sub_labels = _sub_issue_report(
            i, len(count.sub_issues), sub_by_id[sub.id], codebook[sub.id], sub, written,
            count.n_respondents, gate, settings,
        )
        reports.append(report)
        labels[sub.id] = sub_labels

    return Report(
        poll_id=poll_id,
        status="ok",
        n_invited=n_invited,
        n_received=count.n_received,
        n_respondents=count.n_respondents,
        excluded=count.excluded,
        ambiguous_count=count.ambiguous_count,
        sub_issues=reports,
        overall_summary=headline(count.n_respondents, n_invited, count.excluded, reports, labels),
        meta=meta,
    )


def user_visible_strings(report: Report) -> list[str]:
    """Every LLM-written string a user will see: the input for Call 4."""
    out: list[str] = []
    for sub in report.sub_issues:
        out.append(sub.description)
        for theme in sub.themes:
            out += [theme.label, theme.summary]
        out += [cg.point for cg in sub.common_ground]
    return list(dict.fromkeys(out))
