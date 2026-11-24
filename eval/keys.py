"""Ground truth for the scorer: answer key + hand-written overlay (spec §8.1).

SCORER ONLY. Nothing under ``pipeline/`` or ``baseline/`` may import this module
or read the answer keys; ``tests/test_scorer.py`` checks that.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import Settings, load_config
from pipeline.count import majority_kind
from schema.report import NOT_ADDRESSED, Kind

EXCLUDED = "excluded"


@dataclass(frozen=True)
class TruthSubIssue:
    id: str
    labels: dict[str, str]  # respondent id -> key view, or not_addressed
    groups: dict[str, str | None]  # key view -> group type
    counts: dict[str, int]  # key view -> respondents holding it
    not_addressed: int
    expected_stance: tuple[Kind, str | None]
    expected_group: tuple[Kind, str | None] | None  # None for choice questions


@dataclass(frozen=True)
class Alternative:
    id: str
    sub_issue: str
    accept: frozenset[str]  # stance views or not_addressed; exclusion is always the key's answer
    reason: str


@dataclass(frozen=True)
class Truth:
    poll_id: str
    n_received: int
    n_respondents: int
    excluded: dict[str, str]  # id -> exclusion reason
    respondent_ids: list[str]
    sub_issues: list[TruthSubIssue]
    alternatives: dict[tuple[str, str], Alternative]
    must_mention: list[str]
    must_not: list[str]
    must_not_terms: list[str]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_truth(poll_id: str, settings: Settings | None = None) -> Truth:
    settings = settings or load_config()
    key = _read(settings.eval.path("answer_keys_dir") / f"{poll_id}_key.json")
    overlay = _read(settings.eval.path("overlays_dir") / f"{poll_id}.json")
    if key["poll_id"] != poll_id or overlay["poll_id"] != poll_id:
        raise ValueError(f"key or overlay does not belong to {poll_id}")

    views: dict[str, str] = key["labels"]
    excluded_views: dict[str, str] = overlay.get("excluded_views", {})
    sub_cfg: dict[str, dict] = overlay["sub_issues"]
    rename: dict[str, dict[str, str]] = overlay.get("rename", {})

    excluded = {rid: excluded_views[v] for rid, v in views.items() if v in excluded_views}
    per_sub: dict[str, dict[str, str]] = {sub: {} for sub in sub_cfg}
    for rid, view in sorted(views.items()):
        if rid in excluded:
            continue
        for sub in sub_cfg:
            if overlay.get("views_from_key_sub_issue_labels"):
                stance = key["sub_issue_labels"][view][sub]
            else:
                stance = view
            per_sub[sub][rid] = rename.get(sub, {}).get(stance, stance)

    # A respondent with no position on any sub-issue is excluded, as in the pipeline.
    for rid in list(per_sub[next(iter(sub_cfg))]):
        if all(per_sub[sub][rid] == NOT_ADDRESSED for sub in sub_cfg):
            excluded[rid] = "no_position"
            for sub in sub_cfg:
                del per_sub[sub][rid]

    respondents = sorted(rid for rid in views if rid not in excluded)
    n = len(respondents)
    clear, margin = settings.pipeline.clear_majority_exact, settings.pipeline.plurality_margin_exact

    subs = []
    for sub, cfg in sub_cfg.items():
        groups: dict[str, str | None] = cfg["groups"]
        labels = per_sub[sub]
        unknown = {s for s in labels.values() if s != NOT_ADDRESSED} - set(groups)
        if unknown:
            raise ValueError(f"{poll_id}/{sub}: views missing from the overlay: {sorted(unknown)}")
        tally = Counter(s for s in labels.values() if s != NOT_ADDRESSED)
        counts = {view: tally.get(view, 0) for view in groups}
        has_sides = any(g in ("for", "against") for g in groups.values())
        sides = {
            side: sum(c for view, c in counts.items() if groups[view] == side) for side in ("for", "against")
        }
        subs.append(
            TruthSubIssue(
                id=sub,
                labels=labels,
                groups=groups,
                counts=counts,
                not_addressed=sum(1 for s in labels.values() if s == NOT_ADDRESSED),
                expected_stance=majority_kind(counts, n, clear, margin),
                expected_group=majority_kind(sides, n, clear, margin) if has_sides else None,
            )
        )

    alternatives = {
        (a["id"], a["sub_issue"]): Alternative(a["id"], a["sub_issue"], frozenset(a["accept"]), a["reason"])
        for a in overlay.get("alternatives", [])
    }
    return Truth(
        poll_id=poll_id,
        n_received=len(views),
        n_respondents=n,
        excluded=excluded,
        respondent_ids=respondents,
        sub_issues=subs,
        alternatives=alternatives,
        must_mention=list(key.get("must_mention", [])),
        must_not=list(key.get("must_not", [])),
        must_not_terms=list(overlay.get("must_not_terms", [])),
    )
