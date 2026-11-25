"""Run systems on the dev polls and score them (spec §8.5).

Usage::

    python -m eval.run_all --systems strong,naive --runs 5
    python -m eval.run_all --systems strong --runs 1 --polls poll5_security_cameras

Writes ``results/<timestamp>/``:

- ``<system>/<poll>_run<k>.json``: every report
- ``<system>/<poll>_run<k>.output.json``: the parsed LLM output behind it
  (synthetic eval data only; never do this with real responses)
- ``scores.json``: per-poll, per-run scores, plus any errors
- ``summary.md``: mean ± sd per system over polls and runs
- ``alternatives_used.json``: overlay alternative labels the scorer accepted

The pipeline arm is added once ``pipeline/run.py`` exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from baseline import single_call, single_call_naive
from eval.score import score_report
from pipeline.config import REPO_ROOT, Settings, load_config
from pipeline.llm import LLMClient, LLMError
from pipeline.loader import list_poll_ids, load_poll
from pipeline.seeds import base_seed_for
from schema.report import PollInput, Report

SystemFn = Callable[[PollInput, Settings, LLMClient, int], tuple[Report, Any]]
SYSTEMS: dict[str, SystemFn] = {
    "strong": single_call.run,
    "naive": single_call_naive.run,
}
RUN_SEED_STEP = 10  # pass A, pass B and Call 1 use base, base+1, base+2; runs must not overlap

SUMMARY_METRICS = [
    ("stance_macro_f1", "macro-F1"),
    ("stance_accuracy", "accuracy"),
    ("share_mae", "share MAE"),
    ("minority_recall", "minority recall"),
    ("majority_calls_correct", "calls correct"),
    ("wrong_unflagged", "wrong & unflagged"),
    ("flag_rate", "flag rate"),
    ("group_type_accuracy", "group type acc."),
    ("response_group_accuracy", "response group acc."),
    ("exclusion_precision", "excl. precision"),
    ("exclusion_recall", "excl. recall"),
    ("leakage_total", "leakage"),
    ("llm_calls", "LLM calls"),
    ("cost_usd", "cost (USD)"),
    ("latency_s", "latency (s)"),
]


def run_seed(poll_id: str, run_index: int) -> int:
    return base_seed_for(poll_id) + RUN_SEED_STEP * (run_index - 1)


def _value(score: dict[str, Any], key: str) -> float | None:
    if key == "leakage_total":
        return score["leakage"]["total"]
    value = score.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _mean_sd(values: Sequence[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


def summarise(scores: Sequence[dict[str, Any]], systems: Sequence[str]) -> str:
    """Markdown table: one row per metric, one column per system (mean ± sd over every poll and run)."""
    lines = ["| metric | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for key, name in SUMMARY_METRICS:
        cells = []
        for system in systems:
            values = [v for s in scores if s["system"] == system and s.get("scored")
                      for v in [_value(s, key)] if v is not None]
            cells.append(_mean_sd(values))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    for system in systems:
        rows = [s for s in scores if s["system"] == system]
        failed = [s for s in rows if "error" in s]
        lines.append(f"- {system}: {len(rows) - len(failed)} of {len(rows)} runs scored"
                     + (f"; {len(failed)} failed" if failed else ""))
    return "\n".join(lines) + "\n"


def run_all(systems: Sequence[str], runs: int, polls: Sequence[str], settings: Settings, client: LLMClient,
            out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, Any]] = []
    for system in systems:
        folder = out_dir / system
        folder.mkdir(exist_ok=True)
        for poll_id in polls:
            poll = load_poll(poll_id, settings)
            for k in range(1, runs + 1):
                row: dict[str, Any] = {"system": system, "poll_id": poll_id, "run": k}
                try:
                    report, output = SYSTEMS[system](poll, settings, client, run_seed(poll_id, k))
                except LLMError as err:
                    row["error"] = str(err)
                    scores.append(row)
                    print(f"{system} {poll_id} run {k}: failed ({err})", file=sys.stderr)
                    continue
                report.meta.run_index = 1  # eval runs are repeats for stability, not re-runs (spec §6)
                (folder / f"{poll_id}_run{k}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
                if output is not None:
                    (folder / f"{poll_id}_run{k}.output.json").write_text(output.model_dump_json(indent=2),
                                                                         encoding="utf-8")
                row.update(score_report(report, settings))
                scores.append(row)
                print(f"{system} {poll_id} run {k}: done", file=sys.stderr)

    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(summarise(scores, systems), encoding="utf-8")
    used = sorted({a for s in scores for a in s.get("alternatives_used", [])})
    (out_dir / "alternatives_used.json").write_text(json.dumps(used, indent=2), encoding="utf-8")
    return scores


def main(argv: Sequence[str] | None = None) -> int:
    settings = load_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--systems", default="strong,naive", help="comma-separated: " + ", ".join(SYSTEMS))
    parser.add_argument("--runs", type=int, default=settings.eval.n_runs)
    parser.add_argument("--polls", default="", help="comma-separated poll ids (default: all dev polls)")
    parser.add_argument("--out", default="", help="output folder (default: results/<timestamp>)")
    args = parser.parse_args(argv)

    systems = [s for s in args.systems.split(",") if s]
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        parser.error(f"unknown systems: {sorted(unknown)}")
    polls = [p for p in args.polls.split(",") if p] or list_poll_ids(settings)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / settings.eval.results_dir / stamp

    client = LLMClient(settings.llm)
    try:
        client.model_for("baseline")  # fail before any call if the model is not set
    except LLMError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    run_all(systems, args.runs, polls, settings, client, out_dir)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
