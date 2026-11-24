import pytest
from pydantic import ValidationError

from pipeline.config import REPO_ROOT
from pipeline.loader import ForbiddenPathError, list_poll_ids, load_poll, resolve_poll_path
from schema.report import DesignOutput, GroupResult, PollInput, Stance, SubIssue

KEYS_DIR = REPO_ROOT / "evalset" / "answer_keys"


def test_lists_the_five_polls():
    assert list_poll_ids() == [
        "poll1_pool_hours",
        "poll2_dues_increase",
        "poll3_pet_policy",
        "poll4_parking_rules",
        "poll5_security_cameras",
    ]


@pytest.mark.parametrize("poll_id", list_poll_ids())
def test_every_poll_loads(poll_id):
    poll = load_poll(poll_id)
    assert poll.poll_id == poll_id
    assert len(poll.responses) == 40


def test_refuses_answer_key_path():
    with pytest.raises(ForbiddenPathError):
        resolve_poll_path(KEYS_DIR / "poll1_pool_hours_key.json")


def test_refuses_traversal_in_poll_id():
    with pytest.raises(ValueError):
        resolve_poll_path("../answer_keys/poll1_pool_hours_key")


def test_refuses_path_outside_polls_dir():
    with pytest.raises(ForbiddenPathError):
        resolve_poll_path(REPO_ROOT / "evalset" / "README.md")


def test_missing_poll_raises():
    with pytest.raises(FileNotFoundError):
        load_poll("no_such_poll")


def test_duplicate_response_ids_rejected():
    with pytest.raises(ValidationError):
        PollInput(poll_id="p", question="q", responses=[{"id": "r1", "text": "a"}, {"id": "r1", "text": "b"}])


def test_response_id_cannot_break_the_wrapper():
    with pytest.raises(ValidationError):
        PollInput(poll_id="p", question="q", responses=[{"id": 'r1" evil="', "text": "a"}])


def test_group_result_serialises_for_as_keyword():
    g = GroupResult(kind="majority", group="for", for_=30, against=6, conditional=0, alternative=4)
    dumped = g.model_dump()
    assert dumped["for"] == 30 and "for_" not in dumped
    assert GroupResult.model_validate(dumped).for_ == 30


def _stance(sid, group=None):
    return Stance(id=sid, label=sid, group=group, definition=sid)


def test_codebook_needs_reserved_stances():
    with pytest.raises(ValidationError):
        DesignOutput(
            sub_issues=[SubIssue(id="main", description="d")],
            codebook={"main": [_stance("support", "for"), _stance("other")]},
        )


def test_reserved_stance_cannot_have_a_group():
    with pytest.raises(ValidationError):
        DesignOutput(
            sub_issues=[SubIssue(id="main", description="d")],
            codebook={"main": [_stance("support", "for"), _stance("other", "for"), _stance("not_addressed")]},
        )
