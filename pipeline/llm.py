"""Provider wrapper for every LLM call (spec §3, §9).

One entry point, ``LLMClient.call_json``: send a system and a user prompt, get
back a validated Pydantic object. The wrapper handles structured output,
retries, one schema-repair attempt, an eval-only response cache, and a log of
model, tokens, latency and cost for every call.

Transports talk to a provider. ``OpenAITransport`` (also any OpenAI-compatible
endpoint via ``base_url``) and ``AnthropicTransport`` are provided; tests use a
fake transport, so no test makes a network call.

Privacy: the cache stores model output, which can contain respondent text
(evidence quotes). It is for the eval harness on synthetic data only; set
``llm.cache_dir: null`` in production (spec §4.1 persistence rule). Prompts,
outputs and API keys are never logged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from pipeline.config import MODEL_PLACEHOLDER, REPO_ROOT, LLMSettings

T = TypeVar("T", bound=BaseModel)

_UNSUPPORTED_STRICT_KEYS = ("default", "title", "pattern", "minLength", "maxLength", "minItems", "maxItems",
                            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "format")
REPAIR_NOTE = (
    "\n\nYour previous answer did not match the required format. Problems:\n{problems}\n"
    "Return the complete answer again, corrected."
)


class LLMError(RuntimeError):
    """A call failed after its retries, or the provider refused or truncated it."""


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class Transport(Protocol):
    provider: str

    def complete(self, *, model: str, system: str, user: str, schema_name: str, schema: dict[str, Any],
                 seed: int | None, temperature: float | None, max_output_tokens: int) -> Completion: ...


@dataclass
class CallLog:
    call: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float | None
    cached: bool
    repaired: bool


@dataclass
class Usage:
    """Totals for one report, written into ``Meta``."""

    calls: list[CallLog] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def cost_usd(self) -> float | None:
        costs = [c.cost_usd for c in self.calls]
        return None if any(c is None for c in costs) else round(sum(costs), 6)  # type: ignore[arg-type]

    @property
    def latency_s(self) -> float:
        return round(sum(c.latency_s for c in self.calls), 3)

    @property
    def models(self) -> dict[str, str]:
        return {c.call: c.model for c in self.calls}


# ---------------------------------------------------------------------------
# JSON schema for structured output
# ---------------------------------------------------------------------------


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema, made acceptable to OpenAI strict mode.

    Every object gets ``additionalProperties: false`` and lists all its
    properties as required; keywords strict mode rejects are dropped (the
    Pydantic model still checks them after the call). Free-form maps
    (``dict[str, X]``) are not allowed, so LLM output models use lists.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def walk(node: Any) -> None:  # node is a schema, never a map of property names
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        for key in _UNSUPPORTED_STRICT_KEYS:
            node.pop(key, None)
        if node.get("type") == "object" or "properties" in node:
            if "properties" not in node:
                raise ValueError(f"{model.__name__}: free-form objects are not allowed in strict mode")
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for key, value in node.items():
            if key in ("properties", "$defs"):
                for sub in value.values():
                    walk(sub)
            elif key != "required":
                walk(value)

    walk(schema)
    return schema


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


def api_key(env_var: str) -> str:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is in requirements.txt
        pass
    else:
        load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise LLMError(f"no API key: set {env_var} in .env (see .env.example)")
    return key


class OpenAITransport:
    provider = "openai"

    def __init__(self, settings: LLMSettings, client: Any = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key(settings.api_key_env), base_url=settings.base_url,
                            timeout=settings.timeout_s, max_retries=settings.max_retries)
        self.client = client
        self.strict = settings.strict_schema

    def complete(self, *, model, system, user, schema_name, schema, seed, temperature, max_output_tokens):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": self.strict},
            },
            "max_completion_tokens": max_output_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed % (2**31)
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError("the model refused the request")
        if choice.finish_reason == "length":
            raise LLMError("the answer was cut off at max_output_tokens")
        usage = response.usage
        return Completion(choice.message.content or "", getattr(usage, "prompt_tokens", 0) or 0,
                          getattr(usage, "completion_tokens", 0) or 0)


class AnthropicTransport:
    provider = "anthropic"

    def __init__(self, settings: LLMSettings, client: Any = None):
        if client is None:
            from anthropic import Anthropic

            kwargs: dict[str, Any] = {"api_key": api_key(settings.api_key_env), "timeout": settings.timeout_s,
                                      "max_retries": settings.max_retries}
            if settings.base_url:
                kwargs["base_url"] = settings.base_url
            client = Anthropic(**kwargs)
        self.client = client

    def complete(self, *, model, system, user, schema_name, schema, seed, temperature, max_output_tokens):
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": user}],
            "tools": [{"name": schema_name, "description": "Return the answer.", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": schema_name},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.messages.create(**kwargs)  # no seed parameter on this API
        if response.stop_reason == "max_tokens":
            raise LLMError("the answer was cut off at max_output_tokens")
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return Completion(json.dumps(block.input), response.usage.input_tokens, response.usage.output_tokens)
        raise LLMError("the model did not return a tool call")


def make_transport(settings: LLMSettings) -> Transport:
    if settings.provider == "openai":
        return OpenAITransport(settings)
    if settings.provider == "anthropic":
        return AnthropicTransport(settings)
    raise ValueError(f"unknown provider {settings.provider!r}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(self, settings: LLMSettings, transport: Transport | None = None,
                 cache_dir: str | Path | None = "config"):
        self.settings = settings
        self._transport = transport
        if cache_dir == "config":
            cache_dir = settings.cache_dir
        self.cache_dir = (REPO_ROOT / cache_dir) if cache_dir else None

    @property
    def transport(self) -> Transport:
        if self._transport is None:  # created on first use, so no key is needed until a real call
            self._transport = make_transport(self.settings)
        return self._transport

    def model_for(self, call: str) -> str:
        model = getattr(self.settings.models, call)
        if model == MODEL_PLACEHOLDER:
            raise LLMError(f"set llm.models.{call} in config.yaml before running LLM calls")
        return model

    def _cost(self, model: str, completion: Completion) -> float | None:
        price = self.settings.prices.get(model)
        if price is None:
            return None
        return (completion.input_tokens * price.input + completion.output_tokens * price.output) / 1_000_000

    def _cache_path(self, payload: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _complete(self, payload: dict[str, Any]) -> tuple[Completion, bool]:
        path = self._cache_path(payload)
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Completion(**data), True
        completion = self.transport.complete(**{k: v for k, v in payload.items() if k != "provider"})
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(completion)), encoding="utf-8")
        return completion, False

    def call_json(self, call: str, system: str, user: str, output_model: type[T], usage: Usage,
                  seed: int | None = None) -> T:
        """One structured call. ``call`` names the model slot in ``llm.models``."""
        model = self.model_for(call)
        payload = {
            "provider": self.settings.provider,
            "model": model,
            "system": system,
            "user": user,
            "schema_name": output_model.__name__,
            "schema": strict_json_schema(output_model),
            "seed": seed,
            "temperature": self.settings.temperature,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        start = time.perf_counter()
        completion, cached = self._complete(payload)
        tokens_in, tokens_out = completion.input_tokens, completion.output_tokens
        cost = self._cost(model, completion)
        repaired = False
        try:
            result = output_model.model_validate_json(completion.text)
        except ValidationError as err:
            repaired = True
            problems = "\n".join(f"- {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()[:20])
            payload = {**payload, "user": user + REPAIR_NOTE.format(problems=problems)}
            completion, cached_repair = self._complete(payload)
            cached = cached and cached_repair
            tokens_in += completion.input_tokens
            tokens_out += completion.output_tokens
            repair_cost = self._cost(model, completion)
            cost = None if cost is None or repair_cost is None else cost + repair_cost
            try:
                result = output_model.model_validate_json(completion.text)
            except ValidationError as err2:
                raise LLMError(f"{call}: output did not match {output_model.__name__} after one repair "
                               f"({err2.error_count()} problems)") from None
        usage.calls.append(CallLog(call, model, tokens_in, tokens_out, round(time.perf_counter() - start, 3),
                                   cost, cached, repaired))
        return result
