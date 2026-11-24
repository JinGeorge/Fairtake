import pytest

from pipeline.config import load_config
from pipeline.count import count_poll
from pipeline.finalize import LeakChecker, find_leaks, finalize, user_visible_strings
from pipeline.render import public_dict, render_html, render_text
from schema.report import CommonGroundIn, FinalLabel, Meta, Stance, SubIssue, WriteOutput

CFG = load_config()
P = CFG.pipeline
QUESTION = "Should the building install security cameras in common areas?"


def st(sid, group, label, definition):
    return Stance(id=sid, label=label, group=group, definition=definition)


def reserved():
    return [st("other", None, "Other", "Fits no stance"), st("not_addressed", None, "Not addressed", "No position")]


def camera_codebook(support_label="Support"):
    return {
        "main": [
            st("support", "for", support_label, "Wants cameras installed"),
            st("oppose", "against", "Oppose", "Does not want cameras"),
            st("entrances_only", "conditional", "Entrances only", "Supports cameras only at entrances"),
        ]
        + reserved()
    }


def camera_texts():
    texts = {}
    for i in range(1, 21):
        texts[f"r{i:02d}"] = f"Yes please, number {i} supports this for safety"
    for i in range(21, 31):
        texts[f"r{i:02d}"] = f"No thanks from {i}, I value privacy"
    for i in range(31, 35):
        texts[f"r{i:02d}"] = f"Only at the doors, says {i}"
    texts["r21"] = "Cameras are an invasion of everyone's privacy in the hallways"
    return texts


def camera_labels(contested_oppose=0):
    labels = {}
    for i in range(1, 35):
        stance = "support" if i <= 20 else "oppose" if i <= 30 else "entrances_only"
        labels[f"r{i:02d}"] = {"main": FinalLabel(stance=stance, status="agreed")}
    for i in range(21, 21 + contested_oppose):
        labels[f"r{i:02d}"] = {"main": FinalLabel(stance="oppose", status="contested", alt="support")}
    return labels


def camera_count(labels, codebook):
    return count_poll(
        n_received=34,
        no_content_ids=[],
        kept_ids=list(labels),
        triage={},
        codebook=codebook,
        labels=labels,
        settings=P,
    )


WRITTEN = WriteOutput(
    summaries={
        "main": {
            "support": "Supporters want better security for shared spaces.",
            "oppose": "Opponents say cameras are an invasion of everyone's privacy in the hallways.",
            "entrances_only": "Some would accept cameras at entry points only.",
        }
    },
    common_ground={
        "main": [
            CommonGroundIn(point="Clear rules on who may view footage.", ids=["r01", "r02", "r22"]),
            CommonGroundIn(point="Shared by one side only.", ids=["r01", "r02", "r03"]),
            CommonGroundIn(point="Backed by two people.", ids=["r01", "r22"]),
            CommonGroundIn(point="Cites an unknown id.", ids=["r01", "r22", "r99"]),
        ]
    },
)


def build_report(labels=None, codebook=None, written=WRITTEN, **kwargs):
    codebook = codebook or camera_codebook()
    labels = labels or camera_labels()
    texts = camera_texts()
    return finalize(
        poll_id="cameras",
        question=QUESTION,
        count=camera_count(labels, codebook),
        sub_issues=[SubIssue(id="main", description="Whether to install cameras")],
        codebook=codebook,
        written=written,
        response_texts=list(texts.values()),
        roster_names=kwargs.pop("roster_names", ["Daniel Kovac"]),
        n_invited=kwargs.pop("n_invited", None),
        settings=P,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------


def test_leak_checker():
    checker = LeakChecker(
        ["Install security cameras in common areas now, I was burgled twice"], QUESTION, ["Daniel Kovac"], 5
    )
    assert checker.leaks("One said it is needed now, I was burgled twice")  # 5-word quote
    assert not checker.leaks("A resident says I was burgled twice recently")  # only 4 words shared
    assert not checker.leaks("Some want to install security cameras in common areas")  # from the question
    assert checker.leaks("Concerns about [NAME] were raised")
    assert checker.leaks("Text with [removed] in it")
    assert checker.leaks("One complaint was about Kovac")
    assert not checker.leaks("Residents want better lighting")


def test_find_leaks_lists_offending_strings():
    checker = LeakChecker(camera_texts().values(), QUESTION, [], 5)
    assert find_leaks(WRITTEN, checker) == [WRITTEN.summaries["main"]["oppose"]]


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def test_report_basics_and_fallbacks():
    report = build_report()
    sub = report.sub_issues[0]
    assert report.status == "ok" and report.n_respondents == 34
    assert [t.stance_id for t in sub.themes] == ["support", "oppose", "entrances_only"]  # by count
    themes = {t.stance_id: t for t in sub.themes}
    assert themes["support"].x_of_y == "20 of 34 respondents"
    assert themes["oppose"].summary == "Does not want cameras"  # quoted a response
    assert report.meta.leak_replacements == ["main.oppose.summary"]


def test_common_ground_rules():
    points = [cg.point for cg in build_report().sub_issues[0].common_ground]
    assert points == ["Clear rules on who may view footage."]


def test_small_group_gets_definition():
    labels = camera_labels()
    for rid in ["r33", "r34"]:  # entrances_only drops to 2 members
        labels[rid] = {"main": FinalLabel(stance="support", status="agreed")}
    themes = {t.stance_id: t for t in build_report(labels=labels).sub_issues[0].themes}
    assert themes["entrances_only"].count == 2
    assert themes["entrances_only"].summary == "Supports cameras only at entrances"


def test_leaky_label_falls_back_to_group_text():
    report = build_report(codebook=camera_codebook(support_label="Agrees with [NAME]"))
    themes = {t.stance_id: t for t in report.sub_issues[0].themes}
    assert themes["support"].label == "Supports the proposal"
    assert "Supports the proposal: 20 of 34" in report.overall_summary
    assert "main.support.label" in report.meta.leak_replacements


def test_privacy_flags_replace_and_are_recorded():
    flagged = {WRITTEN.summaries["main"]["support"]}
    report = build_report(privacy_flagged=flagged)
    themes = {t.stance_id: t for t in report.sub_issues[0].themes}
    assert themes["support"].summary == "Wants cameras installed"
    assert report.meta.privacy_replacements == ["main.support.summary"]


def test_no_response_text_reaches_the_report():
    report = build_report()
    checker = LeakChecker(camera_texts().values(), QUESTION, ["Daniel Kovac"], P.leak_ngram)
    assert not [s for s in user_visible_strings(report) + [report.overall_summary] if checker.leaks(s)]


# ---------------------------------------------------------------------------
# Headline template
# ---------------------------------------------------------------------------


def test_headline_robust():
    text = build_report().overall_summary
    assert text.startswith("Support: 20 of 34; Oppose: 10 of 34; Entrances only: 4 of 34.")
    assert '"Support" has a narrow majority, not a clear one.' in text
    assert "In total: 20 for, 10 against, 4 in favour only with conditions." in text


def test_headline_not_robust():
    text = build_report(labels=camera_labels(contested_oppose=1)).overall_summary
    assert "1 response was hard to classify" in text
    assert "lies between a narrow majority, not a clear one and a clear majority" in text
    assert "the result is not settled" in text  # group level


def test_headline_with_response_rate():
    assert build_report(n_invited=50).overall_summary.startswith("34 of 50 invited members gave a usable response.")


def test_too_few_report():
    labels = {f"r{i}": {"main": FinalLabel(stance="support", status="agreed")} for i in range(1, 4)}
    count = count_poll(
        n_received=3, no_content_ids=[], kept_ids=list(labels), triage={},
        codebook=camera_codebook(), labels=labels, settings=P,
    )
    report = finalize(
        poll_id="tiny", question=QUESTION, count=count, sub_issues=[SubIssue(id="main", description="d")],
        codebook=camera_codebook(), written=None, response_texts=[], roster_names=[], n_invited=None, settings=P,
    )
    assert report.status == "too_few_responses" and report.sub_issues == []
    assert "too few" in report.overall_summary


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_public_view_strips_internal_fields():
    data = public_dict(build_report(meta=Meta(run_index=2, llm_calls=5)))
    assert "meta" not in data and data["run_index"] == 2
    sub = data["sub_issues"][0]
    assert not {"other_ids", "not_addressed_ids", "contested_ids"} & set(sub)
    assert all("response_ids" not in t for t in sub["themes"])
    assert all("response_ids" not in cg for cg in sub["common_ground"])


def test_render_text_and_html():
    report = build_report(codebook=camera_codebook(support_label="Yes <b>please</b>"))
    text = render_text(report)
    assert "Yes <b>please</b>: 20 of 34 respondents" in text
    page = render_html(report)
    assert "&lt;b&gt;please&lt;/b&gt;" in page and "<b>please</b>" not in page
    assert "r01" not in page and "r01" not in text
