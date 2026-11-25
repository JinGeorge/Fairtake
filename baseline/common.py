"""Shared helpers for the single-call baselines (spec §7).

The baselines see the same prepared input as the pipeline (scrubbed, escaped,
wrapped, in seeded shuffled order) and are scored by the same scorer. They
never read answer keys.

LLM output is repaired in code where it breaks the "every response exactly
once" rule; each repair is recorded in ``meta.notes`` by response id or
location only, never by content.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from pipeline.llm import Usage
from pipeline.prepare import PreparedPoll
from pipeline.seeds import shuffled
from schema.report import (
    NOT_ADDRESSED,
    OTHER,
    RESERVED_STANCES,
    BaselineExclusion,
    BaselineSubIssue,
    FinalExclusion,
    Meta,
    Stance,
    SubIssue,
)

PROMPTS = Path(__file__).resolve().parent / "prompts"
_ID_CHARS = re.compile(r"[^a-z0-9_]+")
_MAX_ID = 40


def load_prompt(name: str, **values: object) -> str:
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    return Template(text).substitute(**{k: str(v) for k, v in values.items()})


def user_message(prepared: PreparedPoll, sub_questions: Sequence[str] | None, seed: int, shuffle: bool) -> str:
    ids = shuffled(prepared.kept_ids, seed) if shuffle else list(prepared.kept_ids)
    lines = [f"Question: {prepared.question}"]
    if sub_questions:
        lines.append("Sub-questions:")
        lines += [f"{i + 1}. {q}" for i, q in enumerate(sub_questions)]
    lines += ["", f"Responses ({len(ids)}):"]
    lines += [prepared.wrapped[rid] for rid in ids]
    return "\n".join(lines)


def clean_id(raw: str, used: set[str], fallback: str) -> str:
    """A valid, unique stance or sub-issue id (``^[a-z0-9_]{1,40}$``, not reserved)."""
    out = _ID_CHARS.sub("_", raw.strip().lower()).strip("_")[:_MAX_ID]
    if not out or out in used or out in RESERVED_STANCES:
        out = fallback
        k = 2
        while out in used:
            out = f"{fallback}_{k}"
            k += 1
    used.add(out)
    return out


@dataclass
class Placement:
    """Where each kept response landed, after repairs."""

    exclusions: dict[str, FinalExclusion]
    sub_issues: list[SubIssue]
    theme_ids: dict[str, list[str]]  # sub-issue id -> clean theme ids, in model order
    raw_theme_ids: dict[str, dict[str, str]]  # sub-issue id -> model theme id -> clean id
    placed: dict[str, dict[str, str]]  # sub-issue id -> response id -> theme id / other / not_addressed
    notes: list[str] = field(default_factory=list)


def place(prepared: PreparedPoll, exclusions_out: Sequence[BaselineExclusion],
          subs_out: Sequence[BaselineSubIssue]) -> Placement:
    kept = set(prepared.kept_ids)
    notes: list[str] = []

    exclusions: dict[str, FinalExclusion] = {}
    for ex in exclusions_out:
        if ex.id not in kept:
            notes.append(f"exclusions: unknown id {ex.id}")
        elif ex.id in exclusions:
            notes.append(f"exclusions: duplicate id {ex.id}")
        else:
            exclusions[ex.id] = FinalExclusion(reason=ex.reason, status="agreed")
    coded = [rid for rid in prepared.kept_ids if rid not in exclusions]

    used_subs: set[str] = set()
    sub_issues, theme_ids, raw_ids, placed = [], {}, {}, {}
    for i, sub in enumerate(subs_out):
        sid = clean_id(sub.id, used_subs, "main" if i == 0 else f"sub{i + 1}")
        sub_issues.append(SubIssue(id=sid, description=sub.description))
        used_themes: set[str] = set()
        raw_ids[sid] = {}
        theme_ids[sid] = []
        where: dict[str, str] = {}
        groups = [(t.id, t.response_ids) for t in sub.themes] + [(NOT_ADDRESSED, sub.not_addressed_ids)]
        for j, (raw, ids) in enumerate(groups):
            if j == len(sub.themes):  # the last group is not_addressed_ids
                tid = NOT_ADDRESSED
            else:
                tid = clean_id(raw, used_themes, f"theme{j + 1}")
                raw_ids[sid].setdefault(raw, tid)
                theme_ids[sid].append(tid)
            for rid in ids:
                if rid not in kept:
                    notes.append(f"{sid}: unknown id {rid}")
                elif rid in exclusions:
                    notes.append(f"{sid}: excluded id {rid} also placed")
                elif rid in where:
                    notes.append(f"{sid}: duplicate id {rid}")
                else:
                    where[rid] = tid
        for rid in coded:
            if rid not in where:
                notes.append(f"{sid}: missing id {rid} counted as other")
                where[rid] = OTHER
        placed[sid] = where
    return Placement(exclusions, sub_issues, theme_ids, raw_ids, placed, notes)


def codebook_for(subs_out: Sequence[BaselineSubIssue], placement: Placement) -> dict[str, list[Stance]]:
    codebook = {}
    for sub, meta in zip(subs_out, placement.sub_issues):
        stances = [
            Stance(id=tid, label=t.label, group=t.group, definition=t.definition)
            for t, tid in zip(sub.themes, placement.theme_ids[meta.id])
        ]
        stances += [
            Stance(id=OTHER, label="Other", group=None, definition="On topic but fits no stance above"),
            Stance(id=NOT_ADDRESSED, label="Not addressed", group=None, definition="States no position"),
        ]
        codebook[meta.id] = stances
    return codebook


def summaries_for(subs_out: Sequence[BaselineSubIssue], placement: Placement) -> dict[str, dict[str, str]]:
    return {
        meta.id: {tid: t.summary for t, tid in zip(sub.themes, placement.theme_ids[meta.id])}
        for sub, meta in zip(subs_out, placement.sub_issues)
    }


def run_meta(usage: Usage, base_seed: int, notes: Sequence[str] = ()) -> Meta:
    return Meta(
        llm_calls=usage.n_calls,
        models=usage.models,
        base_seed=base_seed,
        cost_usd=usage.cost_usd,
        latency_s=usage.latency_s,
        notes=list(notes),
    )


def response_texts(prepared: PreparedPoll) -> list[str]:
    """Raw and scrubbed texts, for the leakage check."""
    return list(prepared.raw.values()) + list(prepared.scrubbed.values())
