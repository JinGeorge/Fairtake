"""Naive baseline (spec §7): one LLM call that also writes the counts and the calls.

Same prompt and placement as the strong baseline, but the model also reports
``n_respondents``, a count for every theme and the majority call at both
levels, and the report uses those numbers as given, even where they disagree
with the model's own response ids. That disagreement is what this baseline
measures. Text still goes through ``finalize`` (leakage check, small-group
rule, template headline), so only the numbers differ from the strong baseline.
"""

from __future__ import annotations

from baseline.common import (
    Placement,
    codebook_for,
    load_prompt,
    place,
    response_texts,
    run_meta,
    summaries_for,
    user_message,
)
from pipeline.config import Settings
from pipeline.count import LEADING_KINDS, CountResult, SubIssueCount
from pipeline.finalize import finalize, too_few_report
from pipeline.llm import LLMClient, Usage
from pipeline.prepare import PreparedPoll, prepare
from pipeline.seeds import base_seed_for, call_seeds
from schema.report import (
    GROUP_TYPES,
    NOT_ADDRESSED,
    OTHER,
    Excluded,
    GroupResult,
    Meta,
    NaiveBaselineOutput,
    PollInput,
    Report,
    StanceResult,
    WriteOutput,
)

SYSTEM = "naive"

NAIVE_RULES = """\
- Also count. Give `n_respondents`: the number of responses that are not excluded and address at least one sub-issue. Give each theme a `count`.
- Make the majority call for each sub-issue, where share = count / n_respondents over the themes:
  - `majority` if the largest share is at least $clear;
  - else `narrow_majority` if it is above 0.5;
  - else `plurality` if it leads the second largest by at least $margin;
  - else `no_clear_majority`.
  Give `result_kind` and the id of the leading theme in `leading_theme` (null for `no_clear_majority`).
- Make the same call comparing only the `for` and `against` totals: `group_kind` and `leading_group`. Use null for both if no theme is `for` or `against`."""


def system_prompt(settings: Settings) -> str:
    rules = (NAIVE_RULES.replace("$clear", f"{settings.pipeline.clear_majority:.2f}")
             .replace("$margin", f"{settings.pipeline.plurality_margin:.2f}"))
    return load_prompt("strong_system", max_stances=settings.pipeline.max_stances, extra_rules=rules)


def _model_count(out: NaiveBaselineOutput, prepared: PreparedPoll, placement: Placement) -> CountResult:
    """Build the Count stage's result from the model's own numbers."""
    notes = placement.notes
    coded = [rid for rid in prepared.kept_ids if rid not in placement.exclusions]
    subs_ids = [s.id for s in placement.sub_issues]
    no_position = [rid for rid in coded if all(placement.placed[s][rid] == NOT_ADDRESSED for s in subs_ids)]
    respondents = [rid for rid in coded if rid not in set(no_position)]

    n = out.n_respondents
    if n <= 0:
        notes.append("n_respondents was not positive; replaced by the number of respondents placed")
        n = len(respondents)

    subs = []
    for sub_out, meta in zip(out.sub_issues, placement.sub_issues):
        tids = placement.theme_ids[meta.id]
        groups = {tid: t.group for t, tid in zip(sub_out.themes, tids)}
        members: dict[str, list[str]] = {tid: [] for tid in tids}
        members[OTHER], members[NOT_ADDRESSED] = [], []
        for rid in respondents:
            members[placement.placed[meta.id][rid]].append(rid)

        counts = {tid: max(t.count, 0) for t, tid in zip(sub_out.themes, tids)}
        counts[OTHER] = len(members[OTHER])
        counts[NOT_ADDRESSED] = len(members[NOT_ADDRESSED])
        totals = dict.fromkeys(GROUP_TYPES, 0)
        for tid, group in groups.items():
            if group is not None:
                totals[group] += counts[tid]

        leader = placement.raw_theme_ids[meta.id].get(sub_out.leading_theme or "")
        if sub_out.leading_theme and leader is None:
            notes.append(f"{meta.id}: leading_theme is not one of its themes")
        if sub_out.result_kind not in LEADING_KINDS:
            leader = None
        theme_kinds = {
            tid: (sub_out.result_kind if tid == leader else "minority") for tid in tids  # type: ignore[misc]
        }

        has_group = any(g in ("for", "against") for g in groups.values())
        group_result = None
        if has_group:
            kind = sub_out.group_kind or "no_clear_majority"
            if sub_out.group_kind is None:
                notes.append(f"{meta.id}: group_kind missing; reported as no_clear_majority")
            group_result = GroupResult(
                kind=kind, group=sub_out.leading_group if kind in LEADING_KINDS else None,
                for_=totals["for"], against=totals["against"],
                conditional=totals["conditional"], alternative=totals["alternative"],
            )

        subs.append(SubIssueCount(
            id=meta.id,
            stance_counts=counts,
            stance_members=members,
            stance_contested=dict.fromkeys(counts, 0),
            contested_ids=[],
            group_totals=totals,
            not_addressed=counts[NOT_ADDRESSED],
            addressed=max(n - counts[NOT_ADDRESSED], 0),
            other_count=counts[OTHER],
            contested_count=None,
            stance_result=StanceResult(kind=sub_out.result_kind, stance_id=leader),
            group_result=group_result,
            theme_kinds=theme_kinds,
            group_mismatch={},
            group_flags=[],
            response_groups={},
        ))

    reasons = {rid: ex.reason for rid, ex in placement.exclusions.items()}
    excluded = Excluded(
        no_content=len(prepared.no_content_ids),
        off_topic=sum(r == "off_topic" for r in reasons.values()),
        low_content=sum(r == "low_content" for r in reasons.values()),
        no_position=len(no_position),
    )
    excluded_ids = {
        "no_content": list(prepared.no_content_ids),
        "off_topic": [rid for rid, r in reasons.items() if r == "off_topic"],
        "low_content": [rid for rid, r in reasons.items() if r == "low_content"],
        "no_position": no_position,
    }
    return CountResult("ok", prepared.n_received, n, excluded, 0, respondents, subs, excluded_ids, None)


def run(poll: PollInput, settings: Settings, client: LLMClient,
        base_seed: int | None = None) -> tuple[Report, NaiveBaselineOutput | None]:
    seed = base_seed if base_seed is not None else base_seed_for(poll.poll_id)
    prepared = prepare(poll, settings.pipeline, settings.scrub)
    if prepared.status != "ok":
        return too_few_report(poll.poll_id, prepared.n_received, prepared.n_received, Excluded(),
                              poll.n_invited, Meta(base_seed=seed)), None

    usage = Usage()
    user = user_message(prepared, poll.sub_questions, call_seeds(seed)["pass_a"], settings.pipeline.shuffle)
    out = client.call_json("baseline", system_prompt(settings), user, NaiveBaselineOutput, usage,
                           seed=call_seeds(seed)["pass_a"])

    placement = place(prepared, out.exclusions, out.sub_issues)
    count = _model_count(out, prepared, placement)
    report = finalize(
        poll_id=poll.poll_id,
        question=poll.question,
        count=count,
        sub_issues=placement.sub_issues,
        codebook=codebook_for(out.sub_issues, placement),
        written=WriteOutput(summaries=summaries_for(out.sub_issues, placement)),
        response_texts=response_texts(prepared),
        roster_names=prepared.roster_names,
        n_invited=poll.n_invited,
        settings=settings.pipeline,
        meta=run_meta(usage, seed, placement.notes),
    )
    return report, out
