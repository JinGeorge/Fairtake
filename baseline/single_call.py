"""Strong baseline (spec §7): one LLM call, then the pipeline's own counting code.

The model sees every prepared response at once and must place every response
id in a theme, in ``not_addressed`` or in an exclusion. Counts, ``kind`` and
``x_of_y`` then come from ``count_poll`` and ``finalize``, exactly as for the
pipeline, so this isolates what the pipeline adds beyond "code counts". It has
one coding pass, so ``robust`` is null, and no privacy review call.
"""

from __future__ import annotations

from baseline.common import (
    codebook_for,
    load_prompt,
    place,
    response_texts,
    run_meta,
    summaries_for,
    user_message,
)
from pipeline.config import Settings
from pipeline.count import count_poll
from pipeline.finalize import finalize, too_few_report
from pipeline.llm import LLMClient, Usage
from pipeline.prepare import prepare
from pipeline.seeds import base_seed_for, call_seeds
from schema.report import Excluded, FinalLabel, Meta, PollInput, Report, StrongBaselineOutput, WriteOutput

SYSTEM = "strong"


def system_prompt(settings: Settings) -> str:
    return load_prompt("strong_system", max_stances=settings.pipeline.max_stances,
                       extra_rules="- Do not count responses or say which view is the majority; only group them.")


def run(poll: PollInput, settings: Settings, client: LLMClient,
        base_seed: int | None = None) -> tuple[Report, StrongBaselineOutput | None]:
    """Returns the report and the parsed model output (None if the poll was too small)."""
    seed = base_seed if base_seed is not None else base_seed_for(poll.poll_id)
    prepared = prepare(poll, settings.pipeline, settings.scrub)
    if prepared.status != "ok":
        return too_few_report(poll.poll_id, prepared.n_received, prepared.n_received, Excluded(),
                              poll.n_invited, Meta(base_seed=seed)), None

    usage = Usage()
    user = user_message(prepared, poll.sub_questions, call_seeds(seed)["pass_a"], settings.pipeline.shuffle)
    out = client.call_json("baseline", system_prompt(settings), user, StrongBaselineOutput, usage,
                           seed=call_seeds(seed)["pass_a"])

    placement = place(prepared, out.exclusions, out.sub_issues)
    codebook = codebook_for(out.sub_issues, placement)
    coded = [rid for rid in prepared.kept_ids if rid not in placement.exclusions]
    labels = {
        rid: {sid: FinalLabel(stance=placement.placed[sid][rid], status="agreed") for sid in codebook}
        for rid in coded
    }
    count = count_poll(
        n_received=prepared.n_received,
        no_content_ids=prepared.no_content_ids,
        kept_ids=prepared.kept_ids,
        exclusions=placement.exclusions,
        codebook=codebook,
        labels=labels,
        settings=settings.pipeline,
        double_coded=False,
    )
    report = finalize(
        poll_id=poll.poll_id,
        question=poll.question,
        count=count,
        sub_issues=placement.sub_issues,
        codebook=codebook,
        written=WriteOutput(summaries=summaries_for(out.sub_issues, placement)),
        response_texts=response_texts(prepared),
        roster_names=prepared.roster_names,
        n_invited=poll.n_invited,
        settings=settings.pipeline,
        meta=run_meta(usage, seed, placement.notes),
    )
    return report, out
