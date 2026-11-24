"""Scorer tests: perfect reports built from the answer keys score perfectly, and
deliberately broken reports lose points in the right places."""

import json
from pathlib import Path

import pytest

from eval.keys import load_truth
from eval.score import IncompleteReportError, format_table, main, score_report
from pipeline.config import REPO_ROOT, load_config
from pipeline.count import count_poll
from pipeline.finalize import finalize
from pipeline.loader import list_poll_ids, load_poll
from schema.report import FinalLabel, Stance, SubIssue, WriteOutput

CFG = load_config()
P = CFG.pipeline
POLLS = list_poll_ids()


def sid(view):
    return view if view == "not_addressed" else f"s_{view}"


def report_from_truth(poll_id, *, relabel=None, extra_stances=(), contested=(), sub_ids=None,
                      reverse_subs=False, summaries=None):
    """Build a report through the real count + finalize code from the answer key.

    relabel: {rid: stance_id} overrides for the first sub-issue.
    extra_stances: [(stance_id, group)] added to the first sub-issue's codebook.
    contested: [(rid, alt_stance_id)] marked contested on the first sub-issue.
    """
    truth = load_truth(poll_id, CFG)
    poll = load_poll(poll_id, CFG)
    sub_ids = sub_ids or {}
    rsub = {t.id: sub_ids.get(t.id, t.id) for t in truth.sub_issues}
    first = truth.sub_issues[0].id

    codebook, sub_meta = {}, []
    for t in truth.sub_issues:
        stances = [Stance(id=sid(v), label=v.replace("_", " ").title(), group=g, definition=f"Holds the {v} view")
                   for v, g in t.groups.items()]
        if t.id == first:
            stances += [Stance(id=s, label=s, group=g, definition=f"Another {s} view") for s, g in extra_stances]
        stances += [Stance(id="other", label="Other", definition="none"),
                    Stance(id="not_addressed", label="Not addressed", definition="none")]
        codebook[rsub[t.id]] = stances
        sub_meta.append(SubIssue(id=rsub[t.id], description=t.id.replace("_", " ")))
    if reverse_subs:
        codebook = dict(reversed(list(codebook.items())))

    triage = {rid: r for rid, r in truth.excluded.items() if r in ("off_topic", "low_content")}
    labels = {}
    for rid in sorted(truth.respondent_ids + [r for r, why in truth.excluded.items() if why == "no_position"]):
        labels[rid] = {}
        for t in truth.sub_issues:
            view = t.labels.get(rid, "not_addressed")
            labels[rid][rsub[t.id]] = FinalLabel(stance=sid(view), status="agreed")
    for rid, stance in (relabel or {}).items():
        labels[rid][rsub[first]] = FinalLabel(stance=stance, status="agreed")
    for rid, alt in contested:
        stance = labels[rid][rsub[first]].stance
        labels[rid][rsub[first]] = FinalLabel(stance=stance, status="contested", alt=alt)

    count = count_poll(
        n_received=len(poll.responses), no_content_ids=[], kept_ids=[r.id for r in poll.responses],
        triage=triage, codebook=codebook, labels=labels, settings=P,
    )
    return finalize(
        poll_id=poll_id, question=poll.question, count=count, sub_issues=sub_meta, codebook=codebook,
        written=WriteOutput(summaries=summaries) if summaries else None,
        response_texts=[r.text for r in poll.responses], roster_names=[], n_invited=None, settings=P,
    )


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


EXPECTED = {
    "poll1_pool_hours": {"main": (("majority", "extend"), None)},
    "poll2_dues_increase": {"main": (("no_clear_majority", None), ("no_clear_majority", None))},
    "poll3_pet_policy": {"main": (("majority", "allow"), None)},
    "poll4_parking_rules": {
        "visitor_limit": (("majority", "support"), ("majority", "for")),
        "fines": (("narrow_majority", "oppose"), ("narrow_majority", "against")),
    },
    "poll5_security_cameras": {"main": (("narrow_majority", "support"), ("narrow_majority", "for"))},
}


@pytest.mark.parametrize("poll_id", POLLS)
def test_expected_majority_calls_match_spec_table(poll_id):
    truth = load_truth(poll_id, CFG)
    got = {t.id: (t.expected_stance, t.expected_group) for t in truth.sub_issues}
    assert got == EXPECTED[poll_id]


def test_truth_counts_agree_with_answer_keys():
    for poll_id in POLLS:
        key = json.loads((REPO_ROOT / "evalset" / "answer_keys" / f"{poll_id}_key.json").read_text("utf-8"))
        truth = load_truth(poll_id, CFG)
        if "sub_issue_counts" in key:
            for t in truth.sub_issues:
                want = {("not_addressed" if v == "unclear" else v): c for v, c in key["sub_issue_counts"][t.id].items()}
                expected = {**{v: 0 for v in t.counts}, "not_addressed": 0, **want}
                assert {**t.counts, "not_addressed": t.not_addressed} == expected
        else:
            want = {v: c for v, c in key["true_counts"].items() if v in truth.sub_issues[0].groups}
            assert truth.sub_issues[0].counts == want
    assert load_truth("poll5_security_cameras", CFG).n_respondents == 34


def test_pipeline_and_baseline_never_import_the_scorer():
    for folder in ("pipeline", "baseline"):
        for path in (REPO_ROOT / folder).glob("**/*.py"):
            source = path.read_text(encoding="utf-8")
            assert "import eval" not in source and "from eval" not in source, path
            assert "load_truth" not in source, path


# ---------------------------------------------------------------------------
# Perfect reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("poll_id", POLLS)
def test_perfect_report_scores_perfectly(poll_id):
    s = score_report(report_from_truth(poll_id), CFG)
    assert s["stance_accuracy"] == 1.0 and s["stance_macro_f1"] == 1.0
    assert s["share_mae"] == 0.0
    assert s["minority_recall"] in (1.0, None)
    assert s["majority_calls_correct"] == 1.0 and s["wrong_unflagged"] == 0
    assert s["group_type_accuracy"] == 1.0
    assert s["exclusion_precision"] == 1.0 and s["exclusion_recall"] == 1.0
    assert all(not v for v in s["missed_views"].values())
    assert all(not v for v in s["extra_stances"].values())
    assert all(not v for v in s["split_views"].values())
    assert s["leakage"]["total"] == 0


def test_sub_issues_are_matched_by_content_not_id_or_order():
    report = report_from_truth(
        "poll4_parking_rules", sub_ids={"visitor_limit": "limit_rule", "fines": "fine_policy"}, reverse_subs=True
    )
    s = score_report(report, CFG)
    assert s["stance_accuracy"] == 1.0 and s["majority_calls_correct"] == 1.0
    assert s["unmatched_sub_issues"] == []


# ---------------------------------------------------------------------------
# Broken reports
# ---------------------------------------------------------------------------


def test_merged_minority_is_missed():
    truth = load_truth("poll3_pet_policy", CFG)
    ban = [rid for rid, v in truth.sub_issues[0].labels.items() if v == "ban"]
    s = score_report(report_from_truth("poll3_pet_policy", relabel={rid: "s_allow" for rid in ban}), CFG)
    assert s["missed_views"] == {"main": ["ban"]}
    assert s["minority_recall"] == 0.5  # restrict found, ban lost
    assert s["stance_accuracy"] == 38 / 40
    assert s["stance_macro_f1"] < 1.0


def test_split_view_is_reported():
    truth = load_truth("poll1_pool_hours", CFG)
    extend = [rid for rid, v in truth.sub_issues[0].labels.items() if v == "extend"]
    relabel = {rid: "s_extend_b" for rid in extend[:14]}
    s = score_report(report_from_truth("poll1_pool_hours", relabel=relabel, extra_stances=[("s_extend_b", None)]), CFG)
    assert s["split_views"] == {"main": ["extend"]}
    assert len(s["extra_stances"]["main"]) == 1
    assert s["stance_accuracy"] == pytest.approx(26 / 40)


def _poll2_shift():
    truth = load_truth("poll2_dues_increase", CFG)
    return [rid for rid, v in truth.sub_issues[0].labels.items() if v == "reject"][:3]


def test_wrong_unflagged_call_is_counted():
    shifted = _poll2_shift()  # approve 21/40 = 0.525: a narrow majority that is not there
    s = score_report(report_from_truth("poll2_dues_increase", relabel={r: "s_approve" for r in shifted}), CFG)
    assert s["majority_calls_correct"] == 0.0
    assert s["wrong_unflagged"] == 2


def test_wrong_but_flagged_call_is_not_counted():
    shifted = _poll2_shift()
    report = report_from_truth(
        "poll2_dues_increase",
        relabel={r: "s_approve" for r in shifted},
        contested=[(r, "s_reject") for r in shifted],
    )
    s = score_report(report, CFG)
    assert s["majority_calls_correct"] == 0.0
    assert s["wrong_unflagged"] == 0 and s["flag_rate"] == 1.0
    assert s["accuracy_contested"] == 0.0 and s["accuracy_agreed"] == 1.0


def test_accepted_alternative_for_abusive_response():
    s = score_report(report_from_truth("poll5_security_cameras", relabel={"r04": "s_oppose"}), CFG)
    assert s["exclusion_precision"] == 1.0 and s["exclusion_recall"] == 1.0
    assert s["stance_accuracy"] == 1.0


def test_unaccepted_placement_of_excluded_response():
    s = score_report(report_from_truth("poll5_security_cameras", relabel={"r04": "s_support"}), CFG)
    assert s["exclusion_recall"] == pytest.approx(5 / 6)


def test_must_not_terms_are_caught():
    summaries = {"main": {"s_support": "Some blame Kovac for the rubbish problem."}}
    s = score_report(report_from_truth("poll5_security_cameras", summaries=summaries), CFG)
    assert s["leakage"]["must_not_term_hits"] == 1


def test_incomplete_report_is_rejected():
    report = report_from_truth("poll1_pool_hours")
    report.sub_issues[0].themes[0].response_ids.pop()
    with pytest.raises(IncompleteReportError):
        score_report(report, CFG)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_cli_prints_a_table(tmp_path, capsys):
    path = tmp_path / "poll1.json"
    path.write_text(report_from_truth("poll1_pool_hours").model_dump_json(), encoding="utf-8")
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "poll1_pool_hours" in out and "1.00" in out
    assert "poll" in format_table([])
