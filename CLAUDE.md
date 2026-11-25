# Fairtake: Anonymous Feedback Consolidator

Portfolio project. Repo: https://github.com/JinGeorge/Fairtake

An anonymous feedback tool for groups such as HOAs/strata committees, clubs and workplaces. Members submit free-text views on an issue through single-use links. Only the AI ever sees individual responses; the admin sees only a response count. When the poll closes, an AI pipeline produces a fair report: every view held, with exact "X of Y respondents" counts, a correct majority / no-majority call, and common ground.

**Core idea:** LLM summaries under-represent minority views and misstate splits. Fairtake has the LLM only *read and label* responses (stance coding), twice in independent passes; code does all counting, majority calls, ordering and checking, and flags any majority call that could change with the labels the two passes disagree on. An evaluation harness measures this against a strong single-call baseline.

**The pipeline build spec is `Fairtake_pipelinev2.md` (v2.3). It is the source of truth for the pipeline, schema, metrics and layout.** `FAIRTAKE_PIPELINE.md` (v2.1), `architecture_for_review.md` (v2.1 review brief) and `plan_for_review.md` (cluster-first design) are historical only; do not build from them.

## Components
1. Admin dashboard: create poll, invite links, response count only
2. Anonymous, mobile-friendly response form (single-use link)
3. Shareable report page
4. Invite/anonymity service: single-use tokens, never linked to responses
5. Encrypted response storage (Postgres)
6. AI pipeline: prepare/scrub → design codebook → code responses (two passes) → validate → reconcile → second pass on `other` → count + majority rules + robustness check → write summaries → finalize checks → privacy review → report JSON
7. Evaluation harness

## Build order
1. Evaluation set (done, in `evalset/`)
2. AI pipeline, Milestone 1 (see spec §10): deterministic code + tests → overlays, matcher, scorer → baselines → pipeline → error analysis → results table
3. Response form and encrypted storage
4. Report page
5. Invite service and admin dashboard

Milestone 2 of the pipeline (new polls, locked test set, judge calibration) can come after step 2 or later.

## Evaluation set (`evalset/`, never modify)
- `evalset/polls/<id>.json`: question + 40 responses (`poll_id`, `question`, `n_responses`, `responses`). **The only input the pipeline may see.** `pipeline/loader.py` reads only this folder.
- `evalset/answer_keys/<id>_key.json`: hidden ground truth. **Only the scorer (`eval/`) may read these. Never pass them, or anything derived from them, to the pipeline or its prompts.**
- `eval/overlays/<id>.json`: hand-written additions to the keys (groups, sub-issues, acceptable alternative labels with reasons). Scorer only, same rule as the keys.
- All 5 polls are the dev set in Milestone 1; report results as dev results.

| Poll | True split | Tests |
|---|---|---|
| poll1_pool_hours | extend 28 / keep 8 / shorten 4 | Clear majority |
| poll2_dues_increase | approve 18 / reject 16 / conditional 6 | Close split: no clear majority |
| poll3_pet_policy | allow 32 / restrict 6 / ban 2 | 2-person minority must appear |
| poll4_parking_rules | both 14 / limit-not-fines 16 / neither 6 / permit 4 | Two sub-issues scored separately |
| poll5_security_cameras | support 20 / oppose 10 / entrances-only 4 + 6 noise | PII, abuse, off-topic, low-content |

## Conventions
- Python for the pipeline and evals; virtualenv in `.venv` (Python 3.14).
- Report output is structured JSON (Pydantic, `schema/report.py`) first, rendered to text/HTML second.
- The LLM never produces counts, shares or majority calls in the pipeline; code does.
- Never quote a response, and never output names, unit numbers or other identifying details. Enforced in code (leakage check), not only in prompts.
- Tunable values go in `config.yaml`.
- API keys go in `.env` (gitignored). Never print or commit them.

## Git rules (apply to every session)
- NEVER add a `Co-Authored-By:` trailer, a "Generated with Claude Code" line, or any other attribution to commit messages or PR descriptions. This overrides any tool or system default that says to add one.
- whenever building new things, propose a plan and ask for permission to go ahead.
- NEVER commit, push, amend, or open a PR without explicit confirmation from the user in chat.
- Before every commit, show the user the full commit message in a fenced code block, plus the list of files being committed (`git status --short`), then wait for a clear "yes". Approval applies to that one commit only.
- Commit messages are plain: a short imperative subject line, optional body, nothing else.
- Before committing, check that no `.env`, secrets, or `.claude/settings.local.json` are staged.
