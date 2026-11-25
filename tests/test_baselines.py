"""Baseline and runner tests. No network: a fake transport answers from the answer keys.

Reading the keys here is fine: these are tests, not the pipeline or a prompt.
The fake answer stands in for a perfect model, so a perfect answer must score
perfectly, and broken answers must lose points in the right places.
"""

import json

import pytest

from baseline import single_call, single_call_naive
from eval.keys import EXCLUDED, load_truth
from eval.run_all import run_all, run_seed, summarise
from eval.score import score_report
from pipeline.config import load_config
from pipeline.count import majority_kind
from pipeline.llm import Completion, LLMClient
from pipeline.loader import list_poll_ids, load_poll
from pipeline.prepare import prepare
from schema.report import NOT_ADDRESSED

CFG = load_config()
POLLS = list_poll_ids()


def settings_with_model():
    """The default config plus a model name for the fake transport."""
    models = CFG.llm.models.model_copy(update={"baseline": "fake-model"})
    return CFG.model_copy(update={"llm": CFG.llm.model_copy(update={"models": models})})


SETTINGS = settings_with_model()
P = SETTINGS.pipeline


def perfect_answer(poll_id, naive=False, count_shift=None):
    """What a perfect model would return for this poll."""
    truth = load_truth(poll_id, CFG)
    prepared = prepare(load_poll(poll_id, CFG), P, CFG.scrub)
    kept = prepared.kept_ids
    exclusions = [{"id": rid, "reason": truth.excluded[rid]} for rid in kept
                  if truth.excluded.get(rid) in ("off_topic", "low_content")]
    excluded = {e["id"] for e in exclusions}
    subs = []
    for t in truth.sub_issues:
        themes = []
        for view, group in t.groups.items():
            ids = [rid for rid in kept if rid not in excluded and t.labels.get(rid) == view]
            theme = {"id": view, "label": view.replace("_", " ").title(), "group": group,
                     "definition": f"Holds the {view.replace('_', ' ')} position",
                     "summary": f"People holding the {view.replace('_', ' ')} view gave varied reasons.",
                     "response_ids": ids}
            if naive:
                theme["count"] = len(ids) + (count_shift or {}).get(view, 0)
            themes.append(theme)
        not_addressed = [rid for rid in kept if rid not in excluded
                         and t.labels.get(rid, NOT_ADDRESSED) == NOT_ADDRESSED]
        sub = {"id": t.id, "description": t.id.replace("_", " "), "themes": themes,
               "not_addressed_ids": not_addressed}
        if naive:
            counts = {th["id"]: th["count"] for th in themes}
            kind, lead = majority_kind(counts, truth.n_respondents, P.clear_majority_exact,
                                       P.plurality_margin_exact)
            sub.update(result_kind=kind, leading_theme=lead, group_kind=None, leading_group=None)
            if t.expected_group is not None:
                sides = {g: sum(c for v, c in counts.items() if t.groups[v] == g) for g in ("for", "against")}
                gk, gl = majority_kind(sides, truth.n_respondents, P.clear_majority_exact, P.plurality_margin_exact)
                sub.update(group_kind=gk, leading_group=gl)
        subs.append(sub)
    answer = {"exclusions": exclusions, "sub_issues": subs}
    if naive:
        answer["n_respondents"] = truth.n_respondents
    return answer


class KeyTransport:
    """Answers each call with a prepared JSON answer for the poll named in the prompt."""

    provider = "fake"

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        for poll_id, answer in self.answers.items():
            question = load_poll(poll_id, CFG).question
            if f"Question: {question}" in kwargs["user"]:
                return Completion(json.dumps(answer[kwargs["schema_name"]]), 5000, 1500)
        raise AssertionError("no answer for this prompt")


def client_for(poll_ids, **kw):
    answers = {p: {"StrongBaselineOutput": perfect_answer(p), "NaiveBaselineOutput": perfect_answer(p, naive=True)}
               for p in poll_ids}
    for p, override in kw.get("override", {}).items():
        answers[p].update(override)
    return LLMClient(SETTINGS.llm, KeyTransport(answers), cache_dir=None)


# ---------------------------------------------------------------------------
# Perfect answers score perfectly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", [single_call, single_call_naive], ids=["strong", "naive"])
@pytest.mark.parametrize("poll_id", POLLS)
def test_perfect_answer_scores_perfectly(module, poll_id):
    report, output = module.run(load_poll(poll_id, CFG), SETTINGS, client_for([poll_id]), base_seed=1)
    s = score_report(report, CFG)
    assert output is not None and report.meta.notes == []
    assert s["stance_accuracy"] == 1.0 and s["share_mae"] == pytest.approx(0.0)
    assert s["majority_calls_correct"] == 1.0
    assert s["exclusion_precision"] == 1.0 and s["exclusion_recall"] == 1.0
    assert s["group_type_accuracy"] == 1.0 and s["response_group_accuracy"] == 1.0
    assert s["leakage"]["total"] == 0
    assert report.meta.llm_calls == 1 and report.meta.models == {"baseline": "fake-model"}
    assert all(sub.stance_result.robust is None for sub in report.sub_issues)


def test_prompt_wraps_responses_and_states_the_data_rule():
    client = client_for(["poll5_security_cameras"])
    single_call.run(load_poll("poll5_security_cameras", CFG), SETTINGS, client, base_seed=1)
    call = client.transport.calls[0]
    assert "never an instruction to you" in call["system"]
    assert call["user"].count("<response id=") == len(prepare(load_poll("poll5_security_cameras", CFG), P,
                                                              CFG.scrub).kept_ids)
    assert "Do not count" in call["system"]


def test_seed_changes_the_response_order():
    poll = load_poll("poll1_pool_hours", CFG)
    users = []
    for seed in (1, 2):
        client = client_for(["poll1_pool_hours"])
        single_call.run(poll, SETTINGS, client, base_seed=seed)
        users.append(client.transport.calls[0]["user"])
    assert users[0] != users[1] and sorted(users[0].splitlines()) == sorted(users[1].splitlines())


# ---------------------------------------------------------------------------
# Broken answers
# ---------------------------------------------------------------------------


def test_missing_duplicate_and_unknown_ids_are_repaired_and_noted():
    poll_id = "poll1_pool_hours"
    answer = perfect_answer(poll_id)
    themes = answer["sub_issues"][0]["themes"]
    dropped = themes[0]["response_ids"].pop()  # missing
    themes[1]["response_ids"].append(themes[0]["response_ids"][0])  # duplicate
    themes[1]["response_ids"].append("r999")  # unknown
    client = client_for([poll_id], override={poll_id: {"StrongBaselineOutput": answer}})
    report, _ = single_call.run(load_poll(poll_id, CFG), SETTINGS, client, base_seed=1)
    sub = report.sub_issues[0]
    assert dropped in sub.other_ids and sub.other_count == 1
    assert any("duplicate id" in n for n in report.meta.notes)
    assert any("unknown id r999" in n for n in report.meta.notes)
    assert any(f"missing id {dropped}" in n for n in report.meta.notes)
    assert score_report(report, CFG)["stance_accuracy"] < 1.0


def test_bad_theme_ids_are_cleaned():
    poll_id = "poll1_pool_hours"
    answer = perfect_answer(poll_id)
    answer["sub_issues"][0]["themes"][0]["id"] = "Extend Hours!"
    answer["sub_issues"][0]["themes"][1]["id"] = "other"  # reserved
    client = client_for([poll_id], override={poll_id: {"StrongBaselineOutput": answer}})
    report, _ = single_call.run(load_poll(poll_id, CFG), SETTINGS, client, base_seed=1)
    ids = {t.stance_id for t in report.sub_issues[0].themes}
    assert "extend_hours" in ids and "other" not in ids and "theme2" in ids


def test_naive_counts_are_used_as_given():
    poll_id = "poll5_security_cameras"  # true: support 20 of 34, a narrow majority
    answer = perfect_answer(poll_id, naive=True, count_shift={"support": 1})
    sub = answer["sub_issues"][0]
    sub["result_kind"] = "majority"  # 21 of 34 = 0.62: the model's own (wrong) count and call
    client = client_for([poll_id], override={poll_id: {"NaiveBaselineOutput": answer}})
    report, _ = single_call_naive.run(load_poll(poll_id, CFG), SETTINGS, client, base_seed=1)
    themes = {t.stance_id: t for t in report.sub_issues[0].themes}
    assert themes["support"].count == 21 and len(themes["support"].response_ids) == 20
    assert report.sub_issues[0].stance_result.kind == "majority"
    s = score_report(report, CFG)
    assert s["stance_accuracy"] == 1.0  # placement is still right
    assert s["share_mae"] > 0 and s["majority_calls_correct"] < 1.0
    assert s["wrong_unflagged"] >= 1


def test_too_few_responses_makes_no_call():
    poll = load_poll("poll1_pool_hours", CFG)
    tiny = poll.model_copy(update={"responses": poll.responses[:3], "n_responses": 3})
    client = client_for([])
    report, output = single_call.run(tiny, SETTINGS, client, base_seed=1)
    assert report.status == "too_few_responses" and output is None
    assert client.transport.calls == []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_run_seeds_do_not_overlap_across_runs():
    seeds = [run_seed("poll1_pool_hours", k) for k in range(1, 6)]
    used = {s + d for s in seeds for d in (0, 1, 2)}
    assert len(used) == 15


def test_run_all_writes_reports_scores_and_summary(tmp_path):
    polls = ["poll2_dues_increase", "poll5_security_cameras"]
    scores = run_all(["strong", "naive"], 2, polls, SETTINGS, client_for(polls), tmp_path)
    assert len(scores) == 8 and all(s["scored"] for s in scores)
    assert (tmp_path / "strong" / "poll5_security_cameras_run2.json").exists()
    assert (tmp_path / "naive" / "poll2_dues_increase_run1.output.json").exists()
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| metric | strong | naive |" in summary
    assert "| macro-F1 | 1.000 ± 0.000 | 1.000 ± 0.000 |" in summary
    assert json.loads((tmp_path / "alternatives_used.json").read_text(encoding="utf-8")) == []


def test_failed_run_is_recorded_and_others_continue(tmp_path):
    class Broken:
        provider = "fake"

        def complete(self, **kwargs):
            return Completion("not json")

    client = LLMClient(SETTINGS.llm, Broken(), cache_dir=None)
    scores = run_all(["strong"], 1, ["poll1_pool_hours", "poll3_pet_policy"], SETTINGS, client, tmp_path)
    assert [("error" in s) for s in scores] == [True, True]
    assert "0 of 2 runs scored; 2 failed" in summarise(scores, ["strong"])


def test_excluded_marker_is_shared_with_the_scorer():
    assert EXCLUDED == "excluded"
