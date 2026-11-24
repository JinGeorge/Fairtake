from fractions import Fraction

import pytest

from pipeline.config import load_config
from pipeline.count import count_poll, majority_kind
from schema.report import FinalLabel, Stance

CFG = load_config()
P = CFG.pipeline
CLEAR, MARGIN = P.clear_majority_exact, P.plurality_margin_exact


# ---------------------------------------------------------------------------
# Majority rule edge cases (n = 100 keeps the shares readable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "counts, expected",
    [
        ({"a": 60, "b": 40}, ("majority", "a")),  # exactly CLEAR_MAJORITY
        ({"a": 59, "b": 41}, ("narrow_majority", "a")),
        ({"a": 51, "b": 49}, ("narrow_majority", "a")),
        ({"a": 50, "b": 50}, ("no_clear_majority", None)),  # 50/50 tie
        ({"a": 50, "b": 40, "c": 10}, ("plurality", "a")),  # exactly 0.5 is not a majority
        ({"a": 45, "b": 35, "c": 20}, ("plurality", "a")),  # gap exactly PLURALITY_MARGIN
        ({"a": 49, "b": 40, "c": 11}, ("no_clear_majority", None)),  # gap 0.09
        ({"a": 40, "b": 40, "c": 20}, ("no_clear_majority", None)),  # tie at the top
        ({}, ("no_clear_majority", None)),
    ],
)
def test_majority_kind_edges(counts, expected):
    assert majority_kind(counts, 100, CLEAR, MARGIN) == expected


def test_tie_never_becomes_plurality_even_with_zero_margin():
    assert majority_kind({"a": 40, "b": 40, "c": 20}, 100, CLEAR, Fraction(0)) == ("no_clear_majority", None)


def test_exact_threshold_at_awkward_denominator():
    # 21 of 35 is exactly 0.6; float division would be fine here, fractions make sure.
    assert majority_kind({"a": 21, "b": 14}, 35, CLEAR, MARGIN)[0] == "majority"


# ---------------------------------------------------------------------------
# Helpers to build polls
# ---------------------------------------------------------------------------


def st(sid, group=None):
    return Stance(id=sid, label=sid, group=group, definition=sid)


def book(*stances):
    return list(stances) + [st("other"), st("not_addressed")]


def build(groups_of_ids, sub="main"):
    """{"support": 20, ...} -> ids and single-issue labels, in order."""
    labels, i = {}, 0
    for stance, n in groups_of_ids.items():
        for _ in range(n):
            i += 1
            labels[f"r{i:02d}"] = {sub: FinalLabel(stance=stance, status="agreed")}
    return labels


def run(labels, codebook, *, no_content=(), triage=None, extra_kept=(), double_coded=True):
    triage = triage or {}
    kept = list(labels) + list(triage) + list(extra_kept)
    return count_poll(
        n_received=len(kept) + len(no_content),
        no_content_ids=list(no_content),
        kept_ids=kept,
        triage=triage,
        codebook=codebook,
        labels=labels,
        settings=P,
        double_coded=double_coded,
    )


# ---------------------------------------------------------------------------
# Evalset-shaped polls
# ---------------------------------------------------------------------------


def test_choice_question_has_no_group_result():  # poll 1
    codebook = {"main": book(st("extend"), st("keep"), st("shorten"))}
    res = run(build({"extend": 28, "keep": 8, "shorten": 4}), codebook)
    sub = res.sub_issues[0]
    assert res.n_respondents == 40
    assert (sub.stance_result.kind, sub.stance_result.stance_id) == ("majority", "extend")
    assert sub.group_result is None
    assert sub.theme_kinds == {"extend": "majority", "keep": "minority", "shorten": "minority"}


def test_close_split_conditional_not_merged():  # poll 2
    codebook = {"main": book(st("approve", "for"), st("reject", "against"), st("conditional", "conditional"))}
    sub = run(build({"approve": 18, "reject": 16, "conditional": 6}), codebook).sub_issues[0]
    assert sub.stance_result.kind == "no_clear_majority"
    g = sub.group_result
    assert (g.kind, g.for_, g.against, g.conditional) == ("no_clear_majority", 18, 16, 6)
    assert all(k == "minority" for k in sub.theme_kinds.values())


def test_multi_issue_counting_and_denominators():  # poll 4
    views = {
        "both": ("support", "support", 14),
        "limit_not_fines": ("support", "oppose", 16),
        "neither": ("oppose", "oppose", 6),
        "permit": ("alternative", "not_addressed", 4),
    }
    labels, i = {}, 0
    for limit, fines, n in views.values():
        for _ in range(n):
            i += 1
            labels[f"r{i:02d}"] = {
                "visitor_limit": FinalLabel(stance=limit, status="agreed"),
                "fines": FinalLabel(stance=fines, status="agreed"),
            }
    codebook = {
        "visitor_limit": book(st("support", "for"), st("oppose", "against"), st("alternative", "alternative")),
        "fines": book(st("support", "for"), st("oppose", "against")),
    }
    res = run(labels, codebook)
    limit, fines = res.sub_issues
    assert res.n_respondents == 40
    assert (limit.stance_result.kind, limit.stance_result.stance_id) == ("majority", "support")
    assert limit.group_result.alternative == 4
    assert (fines.stance_result.kind, fines.stance_result.stance_id) == ("narrow_majority", "oppose")
    assert (fines.group_result.kind, fines.group_result.group) == ("narrow_majority", "against")
    assert (fines.not_addressed, fines.addressed) == (4, 36)


def test_noise_exclusions_and_no_position():  # poll 5
    codebook = {"main": book(st("support", "for"), st("oppose", "against"), st("entrances_only", "conditional"))}
    labels = build({"support": 20, "oppose": 10, "entrances_only": 4})
    labels["r_abuse"] = {"main": FinalLabel(stance="not_addressed", status="agreed")}
    triage = {"r_off1": "off_topic", "r_off2": "off_topic", "r_low": "low_content"}
    res = run(labels, codebook, triage=triage, no_content=["r_idk", "r_meh"])
    assert res.excluded.model_dump() == {"no_content": 2, "off_topic": 2, "low_content": 1, "no_position": 1}
    assert res.n_respondents == 34
    sub = res.sub_issues[0]
    assert sub.stance_result.kind == "narrow_majority"  # 20/34 = 0.588
    assert (sub.group_result.kind, sub.group_result.group) == ("narrow_majority", "for")
    assert sub.stance_result.robust is True


# ---------------------------------------------------------------------------
# Robustness check
# ---------------------------------------------------------------------------


def _poll5_with_contested(n_contested_oppose):
    codebook = {"main": book(st("support", "for"), st("oppose", "against"), st("entrances_only", "conditional"))}
    labels = build({"support": 20, "oppose": 10, "entrances_only": 4})
    oppose_ids = [rid for rid, lab in labels.items() if lab["main"].stance == "oppose"]
    for rid in oppose_ids[:n_contested_oppose]:
        labels[rid] = {"main": FinalLabel(stance="oppose", status="contested", alt="support")}
    return run(labels, codebook)


def test_contested_label_that_crosses_threshold():
    sub = _poll5_with_contested(1).sub_issues[0]  # 20/34 narrow; 21/34 = 0.618 majority
    assert sub.stance_result.kind == "narrow_majority"
    assert sub.stance_result.robust is False
    assert sub.stance_result.alt_kinds == ["majority"]
    assert sub.group_result.robust is False
    assert sub.contested_count == 1


def test_contested_label_that_does_not_cross():
    codebook = {"main": book(st("extend"), st("keep"), st("shorten"))}
    labels = build({"extend": 28, "keep": 8, "shorten": 4})
    labels["r40"] = {"main": FinalLabel(stance="shorten", status="contested", alt="keep")}
    sub = run(labels, codebook).sub_issues[0]
    assert sub.stance_result.robust is True and sub.stance_result.alt_kinds == []


def test_too_many_contested_skips_enumeration():
    n = P.sens_max_contested + 1
    codebook = {"main": book(st("extend"), st("keep"))}
    labels = build({"extend": 30, "keep": 10})
    for rid in list(labels)[:n]:
        labels[rid] = {"main": FinalLabel(stance="extend", status="contested", alt="keep")}
    sub = run(labels, codebook).sub_issues[0]
    assert sub.stance_result.robust is False and sub.stance_result.alt_kinds == []


def test_single_pass_has_no_robustness():
    sub = run(build({"a": 30, "b": 10}), {"main": book(st("a"), st("b"))}, double_coded=False).sub_issues[0]
    assert sub.stance_result.robust is None and sub.contested_count is None


# ---------------------------------------------------------------------------
# Size gate and input checks
# ---------------------------------------------------------------------------


def test_size_gate_after_exclusions():
    labels = build({"a": 3, "b": 1})
    res = run(labels, {"main": book(st("a"), st("b"))}, triage={"x1": "off_topic", "x2": "off_topic"})
    assert res.status == "too_few_responses"
    assert res.n_respondents == 4 and res.sub_issues == []


def test_unknown_stance_rejected():
    labels = build({"a": 5})
    labels["r01"] = {"main": FinalLabel(stance="made_up", status="agreed")}
    with pytest.raises(ValueError, match="unknown stance"):
        run(labels, {"main": book(st("a"))})


def test_missing_label_rejected():
    with pytest.raises(ValueError, match="exactly one label"):
        run(build({"a": 5}), {"main": book(st("a"))}, extra_kept=["r_unlabelled"])
