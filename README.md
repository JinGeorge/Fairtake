# Fairtake

Anonymous feedback consolidator for groups such as HOAs/strata committees, clubs and workplaces.

Members submit free-text views on an issue through single-use links. Only the AI ever sees individual responses; the admin sees only a response count. When the poll closes, an AI pipeline produces a fair report: every view held, with exact "X of Y respondents" counts, a correct majority / no-majority call, and common ground.

## The idea

LLM summaries tend to under-represent minority views and misstate close splits. In Fairtake the LLM only **reads and labels**: it proposes a codebook of stances, then labels each response with one stance and an evidence quote, in two independent passes. **Code does the rest**: it checks every label, counts, applies fixed majority rules, flags any majority call that could change with the responses the two passes disagree on, orders the views, and blocks any quoted or identifying text from reaching the report.

An evaluation harness compares this against a strong single-call baseline (same counting code) and a naive one (the model writes its own counts). The full spec is in [Fairtake_pipelinev2.md](Fairtake_pipelinev2.md) (v2.3). The earlier [FAIRTAKE_PIPELINE.md](FAIRTAKE_PIPELINE.md) (v2.1) is kept for history only.

## Components

1. Admin dashboard: create poll, invite links, response count only
2. Anonymous, mobile-friendly response form (single-use link)
3. Shareable report page
4. Invite/anonymity service: single-use tokens, never linked to responses
5. Encrypted response storage (Postgres)
6. AI pipeline: scrub personal details, design a stance codebook, label each response twice, validate and reconcile, count and apply majority rules with a robustness check, write per-stance summaries, run privacy and common-ground checks, assemble report
7. Evaluation harness

## Evaluation set

`evalset/polls/` holds five synthetic polls of 40 responses each with known opinion splits. Answer keys in `evalset/answer_keys/` are used only by the scorer and are never given to the pipeline.

Metrics include per-response stance accuracy (macro-F1), share error, minority recall, majority-call correctness, confident-wrong calls (wrong and not flagged as fragile), privacy leakage and `must_not` compliance (poll 5), plus stability across runs and cost.

## Status

Evaluation set done. Pipeline spec revised to v2.2 (stance coding with double coding). Next: Milestone 1, starting with the schema and deterministic counting code with unit tests, then the scorer, baselines and pipeline, scored on the 5 polls as a dev set.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Put API keys in `.env` (gitignored).
