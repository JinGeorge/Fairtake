from fractions import Fraction

import pytest

from pipeline.config import load_config
from pipeline.count import count_poll, majority_kind
from schema.report import FinalExclusion, FinalLabel, Stance

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


def run(labels, codebook, *, no_content=(), triage=None, exclusions=None, alt_exclusions=None,
        extra_kept=(), double_coded=True):
    """``triage`` is shorthand for agreed exclusions: {id: reason}."""
    exclusions = dict(exclusions or {})
    exclusions.update({rid: FinalExclusion(reason=r, status="agreed") for rid, r in (triage or {}).items()})
    kept = list(labels) + list(exclusions) + list(extra_kept)
    return count_poll(
        n_received=len(kept) + len(no_content),
        no_content_ids=list(no_content),
        kept_ids=kept,
        exclusions=exclusions,
        alt_exclusions=alt_exclusions,
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


# ---------------------------------------------------------------------------
# v2.3: contested exclusions move the denominator in the robustness check
# ---------------------------------------------------------------------------


def _cameras():
    return {"main": book(st("support", "for"), st("oppose", "against"), st("entrances_only", "conditional"))}


def test_contested_exclusion_that_crosses_threshold():
    # 21 of 35 = 0.60 majority. If pass B was right to count r_x as opposed: 21 of 36 = 0.583.
    labels = build({"support": 21, "oppose": 10, "entrances_only": 4})
    alt = {"main": FinalLabel(stance="oppose", status="contested", group="against")}
    exclusions = {"r_x": FinalExclusion(reason="off_topic", status="contested", alt=alt)}
    res = run(labels, _cameras(), exclusions=exclusions)
    sub = res.sub_issues[0]
    assert res.n_respondents == 35 and res.contested_exclusions == 1
    assert sub.stance_result.kind == "majority"
    assert sub.stance_result.robust is False and sub.stance_result.alt_kinds == ["narrow_majority"]
    assert sub.contested_count == 0  # excluded responses are not in any theme


def test_contested_exclusion_without_alternative_changes_nothing():
    labels = build({"support": 21, "oppose": 10, "entrances_only": 4})
    exclusions = {"r_x": FinalExclusion(reason="off_topic", status="contested")}
    sub = run(labels, _cameras(), exclusions=exclusions).sub_issues[0]
    assert sub.stance_result.robust is True


def test_counted_response_that_pass_b_excluded():
    # 21 of 35 majority; without r01 (support) it is 20 of 34 = 0.588.
    labels = build({"support": 21, "oppose": 10, "entrances_only": 4})
    labels["r01"] = {"main": FinalLabel(stance="support", status="contested", group="for", group_status="contested")}
    res = run(labels, _cameras(), alt_exclusions={"r01": "off_topic"})
    sub = res.sub_issues[0]
    assert res.contested_exclusions == 1
    assert sub.stance_result.robust is False and sub.stance_result.alt_kinds == ["narrow_majority"]
    assert sub.group_result.robust is False


def test_no_position_flip_recomputes_the_denominator():
    # 21 of 36 = 0.583. If pass B's not_addressed is right, r22 has no position and drops out:
    # 21 of 35 = 0.60. Holding n fixed (v2.2) would have called this robust.
    labels = build({"support": 21, "oppose": 11, "entrances_only": 4})
    labels["r22"] = {"main": FinalLabel(stance="oppose", status="contested", alt="not_addressed",
                                        group="against", group_status="contested", alt_group="none")}
    sub = run(labels, _cameras()).sub_issues[0]
    assert sub.stance_result.kind == "narrow_majority"
    assert sub.stance_result.robust is False and sub.stance_result.alt_kinds == ["majority"]


def test_robustness_reports_too_few_respondents():
    labels = build({"a": 4, "b": 1})
    labels["r05"] = {"main": FinalLabel(stance="b", status="contested", alt="not_addressed")}
    sub = run(labels, {"main": book(st("a"), st("b"))}).sub_issues[0]  # alt version: only 4 respondents
    assert sub.stance_result.robust is False


def test_combination_limit_skips_enumeration():
    labels = build({"a": 30, "b": 10})
    for rid in list(labels)[: P.sens_max_contested + 1]:
        labels[rid] = {"main": FinalLabel(stance="a", status="contested", alt="b")}
    sub = run(labels, {"main": book(st("a"), st("b"))}).sub_issues[0]
    assert sub.stance_result.robust is False and sub.stance_result.alt_kinds == []


# ---------------------------------------------------------------------------
# v2.3: per-response groups and the mismatch check
# ---------------------------------------------------------------------------


def _with_groups(labels, groups):
    for rid, group in groups.items():
        labels[rid] = {"main": labels[rid]["main"].model_copy(update={"group": group, "group_status": "agreed"})}
    return labels


def test_group_totals_use_per_response_groups():
    labels = build({"support": 21, "oppose": 10, "entrances_only": 4})
    _with_groups(labels, {"r01": "conditional", "r02": "conditional"})  # supporters who attach conditions
    sub = run(labels, _cameras()).sub_issues[0]
    g = sub.group_result
    assert (g.for_, g.against, g.conditional) == (19, 10, 6)
    assert g.kind == "narrow_majority"  # 19 of 35
    assert sub.stance_result.kind == "majority"  # 21 of 35: stance counts are unchanged
    assert sub.group_mismatch == {"support": 2, "oppose": 0, "entrances_only": 0}
    assert sub.group_flags == ["support"]
    # With the stance's own mapping it would be 21 of 35 = 0.60: the call is not settled.
    assert g.robust is False and g.alt_kinds == ["majority"]
    assert sub.response_groups["r01"] == "conditional" and sub.response_groups["r03"] == "for"


def test_one_mismatch_is_not_flagged():
    labels = _with_groups(build({"support": 25, "oppose": 10}), {"r01": "conditional"})
    sub = run(labels, _cameras()).sub_issues[0]
    assert sub.group_mismatch["support"] == 1 and sub.group_flags == []
    assert sub.group_result.robust is True  # 24 or 25 of 35 is a clear majority either way


def test_other_counts_toward_its_group_and_is_never_mismatched():
    labels = build({"support": 20, "oppose": 10, "other": 2})
    _with_groups(labels, {"r31": "for", "r32": "for"})
    sub = run(labels, _cameras()).sub_issues[0]
    assert sub.group_result.for_ == 22
    assert sub.group_flags == []


def test_choice_question_flags_mismatches_but_has_no_group_result():
    labels = _with_groups(build({"extend": 28, "keep": 8, "shorten": 4}), {"r01": "for", "r02": "for"})
    sub = run(labels, {"main": book(st("extend"), st("keep"), st("shorten"))}).sub_issues[0]
    assert sub.group_result is None
    assert sub.group_flags == ["extend"]


def test_single_pass_still_flags_mismatches():
    labels = _with_groups(build({"support": 21, "oppose": 10}), {"r01": "conditional", "r02": "conditional"})
    sub = run(labels, _cameras(), double_coded=False).sub_issues[0]
    assert sub.group_flags == ["support"] and sub.group_result.robust is None


def test_contested_group_counts_as_contested():
    labels = build({"support": 21, "oppose": 14})
    labels["r01"] = {"main": FinalLabel(stance="support", status="agreed", group="for",
                                        group_status="contested", alt_group="conditional")}
    sub = run(labels, _cameras()).sub_issues[0]
    assert sub.contested_count == 1 and sub.stance_contested["support"] == 1
    assert sub.stance_result.robust is True  # the stance itself is agreed
    assert sub.group_result.robust is False  # for 21 or 20 of 35: majority or narrow


# ---------------------------------------------------------------------------
# v2.3: input checks
# ---------------------------------------------------------------------------


def test_exclusion_of_unknown_response_rejected():
    with pytest.raises(ValueError, match="not kept"):
        count_poll(n_received=5, no_content_ids=[], kept_ids=list(build({"a": 5})),
                   exclusions={"zz": FinalExclusion(reason="off_topic", status="agreed")},
                   codebook={"main": book(st("a"))}, labels=build({"a": 5}), settings=P)


def test_alt_exclusion_must_name_a_counted_response():
    with pytest.raises(ValueError, match="alt_exclusions"):
        run(build({"a": 5}), {"main": book(st("a"))}, triage={"x": "off_topic"}, alt_exclusions={"x": "off_topic"})
