"""Prepare stage (spec §4.1): size gate, scrub, drop empty, escape and wrap.

No LLM calls. Raw text stays in the returned object, in memory only, for the
leakage check; it is never written anywhere by this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pipeline.config import PipelineSettings, ScrubSettings
from schema.report import PollInput

NAME_TOKEN = "[NAME]"
UNIT_TOKEN = "[UNIT]"
EMAIL_TOKEN = "[EMAIL]"
PHONE_TOKEN = "[PHONE]"
SCRUB_TOKENS = (NAME_TOKEN, UNIT_TOKEN, EMAIL_TOKEN, PHONE_TOKEN)

# A run of digits and phone separators that starts and ends with a digit.
_PHONE_CANDIDATE = re.compile(r"(?<![\w+])\+?\d[\d\s().-]{5,}\d(?!\w)")
_HAS_WORD_CHAR = re.compile(r"\w")
_MIN_NAME_PART = 3


@dataclass
class PreparedPoll:
    poll_id: str
    question: str
    status: Literal["ok", "too_few_responses"]
    n_received: int
    n_invited: int | None
    roster_names: list[str]
    raw: dict[str, str] = field(repr=False)
    scrubbed: dict[str, str] = field(repr=False)
    kept_ids: list[str]
    no_content_ids: list[str]
    wrapped: dict[str, str] = field(repr=False)


def word_count(text: str) -> int:
    return sum(1 for token in text.split() if _HAS_WORD_CHAR.search(token))


def _scrub_phones(text: str, min_digits: int) -> str:
    def replace(match: re.Match[str]) -> str:
        digits = sum(ch.isdigit() for ch in match.group(0))
        return PHONE_TOKEN if digits >= min_digits else match.group(0)

    return _PHONE_CANDIDATE.sub(replace, text)


def _name_patterns(roster_names: list[str]) -> list[re.Pattern[str]]:
    """Full names case-insensitively, then single name parts case-sensitively.

    Single parts are case-sensitive so that a roster name like "Rose" does not
    scrub the word "rose". Longest patterns come first.
    """
    full: set[str] = set()
    parts: set[str] = set()
    for name in roster_names:
        words = name.split()
        if not words:
            continue
        full.add(" ".join(words))
        parts.update(w for w in words if len(w) >= _MIN_NAME_PART)

    patterns = []
    for name in sorted(full, key=len, reverse=True):
        body = r"\s+".join(re.escape(w) for w in name.split())
        patterns.append(re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE))
    for part in sorted(parts, key=len, reverse=True):
        patterns.append(re.compile(rf"(?<!\w){re.escape(part)}(?!\w)"))
    return patterns


def scrub(text: str, roster_names: list[str], settings: ScrubSettings) -> str:
    out = re.sub(settings.email_pattern, EMAIL_TOKEN, text)
    out = _scrub_phones(out, settings.phone_min_digits)
    out = re.sub(settings.unit_pattern, UNIT_TOKEN, out)
    for pattern in _name_patterns(roster_names):
        out = pattern.sub(NAME_TOKEN, out)
    return out


def escape(text: str) -> str:
    """Stop a respondent from closing the wrapper or opening a fake one."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def wrap(response_id: str, text: str) -> str:
    return f'<response id="{response_id}">{escape(text)}</response>'


def prepare(poll: PollInput, pipeline: PipelineSettings, scrub_settings: ScrubSettings) -> PreparedPoll:
    roster = list(poll.roster_names or [])
    raw = {r.id: r.text for r in poll.responses}
    n_received = len(raw)

    if n_received < pipeline.min_respondents:
        return PreparedPoll(
            poll_id=poll.poll_id,
            question=poll.question,
            status="too_few_responses",
            n_received=n_received,
            n_invited=poll.n_invited,
            roster_names=roster,
            raw=raw,
            scrubbed={},
            kept_ids=[],
            no_content_ids=[],
            wrapped={},
        )

    scrubbed = {rid: scrub(text, roster, scrub_settings) for rid, text in raw.items()}
    kept, no_content = [], []
    for rid, text in scrubbed.items():
        (kept if word_count(text) >= pipeline.min_words else no_content).append(rid)

    return PreparedPoll(
        poll_id=poll.poll_id,
        question=poll.question,
        status="ok",
        n_received=n_received,
        n_invited=poll.n_invited,
        roster_names=roster,
        raw=raw,
        scrubbed=scrubbed,
        kept_ids=kept,
        no_content_ids=no_content,
        wrapped={rid: wrap(rid, scrubbed[rid]) for rid in kept},
    )
