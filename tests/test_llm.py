"""LLM wrapper tests. No network: every provider client here is a fake."""

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from pipeline.config import load_config
from pipeline.llm import (
    AnthropicTransport,
    Completion,
    LLMClient,
    LLMError,
    OpenAITransport,
    Usage,
    api_key,
    strict_json_schema,
)
from schema.report import NaiveBaselineOutput, StrongBaselineOutput

CFG = load_config()


def llm_settings(**updates):
    models = CFG.llm.models.model_copy(update={"baseline": "fake-model"})
    return CFG.llm.model_copy(update={"models": models, **updates})


class Answer(BaseModel):
    title: str  # a property named like a JSON-schema keyword must survive
    n: int
    note: str | None


class FakeTransport:
    provider = "fake"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return Completion(self.texts.pop(0), input_tokens=1000, output_tokens=200)


# ---------------------------------------------------------------------------
# Strict schema
# ---------------------------------------------------------------------------


def _objects(node):
    if isinstance(node, dict):
        if "properties" in node:
            yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _objects(item)


@pytest.mark.parametrize("model", [Answer, StrongBaselineOutput, NaiveBaselineOutput])
def test_strict_schema_requires_every_property(model):
    schema = strict_json_schema(model)
    objects = list(_objects(schema))
    assert objects
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    assert "minItems" not in json.dumps(schema)


def test_strict_schema_keeps_property_names_that_look_like_keywords():
    assert "title" in strict_json_schema(Answer)["properties"]


def test_strict_schema_rejects_maps():
    class WithMap(BaseModel):
        counts: dict[str, int]

    with pytest.raises(ValueError, match="free-form"):
        strict_json_schema(WithMap)


# ---------------------------------------------------------------------------
# Client: validation, repair, cache, cost
# ---------------------------------------------------------------------------


def test_valid_answer_is_parsed_and_logged():
    transport = FakeTransport('{"title": "x", "n": 3, "note": null}')
    client = LLMClient(llm_settings(), transport, cache_dir=None)
    usage = Usage()
    out = client.call_json("baseline", "sys", "user", Answer, usage, seed=7)
    assert out.n == 3
    call = transport.calls[0]
    assert (call["model"], call["seed"], call["schema_name"]) == ("fake-model", 7, "Answer")
    assert usage.n_calls == 1 and usage.calls[0].repaired is False
    assert usage.cost_usd is None  # no price configured


def test_one_repair_then_success():
    transport = FakeTransport('{"title": "x"}', '{"title": "x", "n": 1, "note": "ok"}')
    usage = Usage()
    out = LLMClient(llm_settings(), transport, cache_dir=None).call_json("baseline", "s", "u", Answer, usage)
    assert out.n == 1
    assert "did not match the required format" in transport.calls[1]["user"]
    assert usage.calls[0].repaired and usage.calls[0].input_tokens == 2000


def test_second_failure_raises():
    transport = FakeTransport("not json", '{"n": "three"}')
    with pytest.raises(LLMError, match="after one repair"):
        LLMClient(llm_settings(), transport, cache_dir=None).call_json("baseline", "s", "u", Answer, Usage())


def test_cache_serves_repeat_calls(tmp_path):
    transport = FakeTransport('{"title": "x", "n": 3, "note": null}')
    client = LLMClient(llm_settings(), transport, cache_dir=tmp_path / "cache")
    usage = Usage()
    for _ in range(2):
        client.call_json("baseline", "s", "u", Answer, usage, seed=1)
    assert len(transport.calls) == 1
    assert [c.cached for c in usage.calls] == [False, True]
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_cache_key_changes_with_seed(tmp_path):
    transport = FakeTransport('{"title": "x", "n": 3, "note": null}', '{"title": "y", "n": 4, "note": null}')
    client = LLMClient(llm_settings(), transport, cache_dir=tmp_path)
    assert client.call_json("baseline", "s", "u", Answer, Usage(), seed=1).n == 3
    assert client.call_json("baseline", "s", "u", Answer, Usage(), seed=2).n == 4


def test_cost_from_prices():
    data = llm_settings().model_dump()
    data["prices"] = {"fake-model": {"input": 2.0, "output": 10.0}}
    settings = type(CFG.llm).model_validate(data)
    usage = Usage()
    LLMClient(settings, FakeTransport('{"title": "x", "n": 1, "note": null}'), cache_dir=None).call_json(
        "baseline", "s", "u", Answer, usage)
    assert usage.cost_usd == pytest.approx((1000 * 2 + 200 * 10) / 1e6)


def test_unset_model_is_refused():
    models = CFG.llm.models.model_copy(update={"call1_design": "CHANGE_ME"})
    client = LLMClient(CFG.llm.model_copy(update={"models": models}), FakeTransport(), cache_dir=None)
    with pytest.raises(LLMError, match="llm.models.call1_design"):
        client.call_json("call1_design", "s", "u", Answer, Usage())


def test_missing_api_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("FAIRTAKE_TEST_KEY", raising=False)
    with pytest.raises(LLMError, match="FAIRTAKE_TEST_KEY"):
        api_key("FAIRTAKE_TEST_KEY")


# ---------------------------------------------------------------------------
# Transports, with fake SDK clients
# ---------------------------------------------------------------------------


def _openai_client(content="{}", finish="stop", refusal=None):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content=content, refusal=refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                               usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5))

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), captured


ARGS = dict(model="m", system="sys", user="usr", schema_name="Answer", schema={"type": "object"},
            seed=2**40 + 3, temperature=0, max_output_tokens=500)


def test_openai_request_shape():
    client, captured = _openai_client('{"a": 1}')
    out = OpenAITransport(llm_settings(), client).complete(**ARGS)
    assert out == Completion('{"a": 1}', 11, 5)
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["seed"] < 2**31 and captured["temperature"] == 0
    assert captured["max_completion_tokens"] == 500


def test_openai_omits_temperature_when_null():
    client, captured = _openai_client()
    OpenAITransport(llm_settings(), client).complete(**{**ARGS, "temperature": None})
    assert "temperature" not in captured


@pytest.mark.parametrize("kwargs, match", [({"refusal": "no"}, "refused"), ({"finish": "length"}, "cut off")])
def test_openai_refusal_and_truncation(kwargs, match):
    client, _ = _openai_client(**kwargs)
    with pytest.raises(LLMError, match=match):
        OpenAITransport(llm_settings(), client).complete(**ARGS)


def test_anthropic_uses_a_forced_tool_call():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        block = SimpleNamespace(type="tool_use", input={"a": 1})
        return SimpleNamespace(content=[block], stop_reason="tool_use",
                               usage=SimpleNamespace(input_tokens=9, output_tokens=4))

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    out = AnthropicTransport(llm_settings(), client).complete(**ARGS)
    assert json.loads(out.text) == {"a": 1} and out.input_tokens == 9
    assert captured["tool_choice"] == {"type": "tool", "name": "Answer"}
    assert "seed" not in captured
