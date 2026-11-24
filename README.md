# Fairtake

Anonymous feedback consolidator for groups such as HOAs/strata committees, clubs and workplaces.

Members submit free-text views on an issue through single-use links. Only the AI ever sees individual responses; the admin sees only a response count. When the poll closes, an AI pipeline produces a fair report: the majority view, minority views and common ground, with "X of Y respondents" counts per theme.

## The idea

LLM summaries tend to under-represent minority views. Fairtake uses a cluster-first design (embed, cluster, count, then summarise each cluster) and measures it against a naive "just summarise" baseline using an evaluation harness.

## Components

1. Admin dashboard: create poll, invite links, response count only
2. Anonymous, mobile-friendly response form (single-use link)
3. Shareable report page
4. Invite/anonymity service: single-use tokens, never linked to responses
5. Encrypted response storage (Postgres)
6. AI pipeline: decrypt, scrub personal details, filter abusive/off-topic, embed, cluster, count, summarise per cluster, assemble report
7. Evaluation harness

## Evaluation set

`evalset/polls/` holds five synthetic polls of 40 responses each with known opinion splits. Answer keys in `evalset/answer_keys/` are used only by the scorer and are never given to the pipeline.

Metrics: proportion error per view, minority coverage, faithfulness, and `must_not` compliance (poll 5).

## Status

Evaluation set done. Next: baseline pipeline and scorer, then the clustered pipeline and a comparison table.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Put API keys in `.env` (gitignored).
