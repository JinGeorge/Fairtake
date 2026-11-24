"""Load config.yaml into typed, read-only settings.

The file is chosen in this order:
1. the path passed to ``load_config()``
2. the ``FAIRTAKE_CONFIG`` environment variable
3. ``config.yaml`` at the repository root

Unknown keys are rejected, so a typo in the file fails loudly instead of being ignored.
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
CONFIG_ENV_VAR = "FAIRTAKE_CONFIG"
MODEL_PLACEHOLDER = "CHANGE_ME"


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderSettings(_Settings):
    provider: str
    api_key_env: str
    base_url: str | None = None
    timeout_s: float = Field(default=120, gt=0)
    max_retries: int = Field(default=3, ge=0)


class CallModels(_Settings):
    call1_design: str
    call1b_other: str
    call2a_code: str
    call2b_code: str
    call3_write: str
    call4_privacy: str
    baseline: str


class LLMSettings(ProviderSettings):
    cache_dir: str | None = None
    models: CallModels


class JudgeSettings(ProviderSettings):
    model: str


class PipelineSettings(_Settings):
    min_respondents: int = Field(ge=1)
    min_words: int = Field(ge=0)
    max_stances: int = Field(ge=1)
    coding_batch: int = Field(ge=1)
    other_recode_min: int = Field(ge=1)
    clear_majority: float = Field(gt=0.5, le=1.0)
    plurality_margin: float = Field(ge=0.0, lt=1.0)
    min_group: int = Field(ge=1)
    leak_ngram: int = Field(ge=3)
    sens_max_contested: int = Field(ge=0, le=20)
    shuffle: bool
    double_coding: bool
    validation_retry: bool
    other_second_pass: bool
    privacy_review: bool

    @property
    def clear_majority_exact(self) -> Fraction:
        """Threshold as an exact fraction, so 24/40 compares equal to 0.60."""
        return Fraction(str(self.clear_majority))

    @property
    def plurality_margin_exact(self) -> Fraction:
        return Fraction(str(self.plurality_margin))


class ScrubSettings(_Settings):
    unit_pattern: str
    email_pattern: str
    phone_min_digits: int = Field(ge=5)


class EvalSettings(_Settings):
    n_runs: int = Field(ge=1)
    polls_dir: str
    answer_keys_dir: str
    overlays_dir: str
    results_dir: str

    def path(self, name: str) -> Path:
        """Resolve one of the directory settings against the repository root."""
        return (REPO_ROOT / getattr(self, name)).resolve()


class Settings(_Settings):
    llm: LLMSettings
    judge: JudgeSettings
    pipeline: PipelineSettings
    scrub: ScrubSettings
    eval: EvalSettings


def config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(CONFIG_ENV_VAR)
    return Path(env) if env else DEFAULT_CONFIG_PATH


def load_config(path: str | Path | None = None) -> Settings:
    source = config_path(path)
    with open(source, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{source} does not contain a YAML mapping")
    return Settings.model_validate(data)
