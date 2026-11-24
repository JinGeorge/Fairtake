"""The only way the pipeline reads poll data.

It reads poll files from the configured polls folder and refuses anything else,
in particular anything under an ``answer_keys`` folder. Answer keys are for the
scorer only and must never reach the pipeline or its prompts.
"""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.config import Settings, load_config
from schema.report import PollInput

_POLL_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_FORBIDDEN_PART = "answer_keys"


class ForbiddenPathError(PermissionError):
    """Raised when something asks the loader for a file it must not read."""


def _polls_dir(settings: Settings | None) -> Path:
    settings = settings or load_config()
    polls_dir = settings.eval.path("polls_dir")
    keys_dir = settings.eval.path("answer_keys_dir")
    if _is_forbidden(polls_dir) or polls_dir.is_relative_to(keys_dir):
        raise ForbiddenPathError(f"polls_dir points into the answer keys: {polls_dir}")
    return polls_dir


def _is_forbidden(path: Path) -> bool:
    return any(part.lower() == _FORBIDDEN_PART for part in path.parts)


def resolve_poll_path(poll: str | Path, settings: Settings | None = None) -> Path:
    """Turn a poll id or a path into a checked path inside the polls folder.

    A ``str`` is treated as a poll id (``"poll1_pool_hours"``). A ``Path`` is
    resolved as given. Either way the result must sit inside the polls folder.
    """
    polls_dir = _polls_dir(settings)
    if isinstance(poll, Path):
        candidate = poll.resolve()
    else:
        if not _POLL_ID.match(poll):
            raise ValueError(f"not a valid poll id: {poll!r}")
        candidate = (polls_dir / f"{poll}.json").resolve()

    if _is_forbidden(candidate):
        raise ForbiddenPathError(f"the pipeline may not read answer keys: {candidate}")
    if not candidate.is_relative_to(polls_dir):
        raise ForbiddenPathError(f"the pipeline may only read from {polls_dir}: {candidate}")
    return candidate


def load_poll(poll: str | Path, settings: Settings | None = None) -> PollInput:
    path = resolve_poll_path(poll, settings)
    return PollInput.model_validate_json(path.read_text(encoding="utf-8"))


def list_poll_ids(settings: Settings | None = None) -> list[str]:
    return sorted(p.stem for p in _polls_dir(settings).glob("*.json"))
