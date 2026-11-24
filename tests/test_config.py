import pytest
import yaml
from pydantic import ValidationError

from pipeline.config import CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH, load_config


def _default_data():
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_default_config_loads():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.llm.provider == "openai"
    assert cfg.llm.api_key_env == "OPENAI_API_KEY"
    assert cfg.pipeline.clear_majority == 0.60
    assert cfg.pipeline.min_group == 3


def test_thresholds_are_exact_fractions():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    from fractions import Fraction

    assert cfg.pipeline.clear_majority_exact == Fraction(3, 5)
    assert cfg.pipeline.plurality_margin_exact == Fraction(1, 10)


def test_provider_switch_via_env_var(tmp_path, monkeypatch):
    data = _default_data()
    data["llm"]["provider"] = "anthropic"
    data["llm"]["api_key_env"] = "ANTHROPIC_API_KEY"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(_write(tmp_path, data)))
    cfg = load_config()
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.api_key_env == "ANTHROPIC_API_KEY"


def test_openai_compatible_endpoint_via_base_url(tmp_path):
    data = _default_data()
    data["llm"]["base_url"] = "https://example.invalid/v1"
    cfg = load_config(_write(tmp_path, data))
    assert cfg.llm.base_url == "https://example.invalid/v1"


def test_unknown_key_is_rejected(tmp_path):
    data = _default_data()
    data["pipeline"]["clear_majorty"] = 0.6  # typo
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))


def test_clear_majority_must_be_above_half(tmp_path):
    data = _default_data()
    data["pipeline"]["clear_majority"] = 0.5
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, data))
