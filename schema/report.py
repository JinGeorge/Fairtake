"""Pydantic models shared by the pipeline, the baselines and the scorer (spec §5).

Sections:
- Input: a poll as the pipeline receives it.
- LLM outputs: the JSON each LLM call must return.
- Report: the final report.
- Internal: working types passed between pipeline stages.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OTHER = "other"
NOT_ADDRESSED = "not_addressed"
RESERVED_STANCES = frozenset({OTHER, NOT_ADDRESSED})

GroupType = Literal["for", "against", "conditional", "alternative"]
GROUP_TYPES: tuple[str, ...] = ("for", "against", "conditional", "alternative")
NO_GROUP = "none"
ResponseGroup = Literal["for", "against", "conditional", "alternative", "none"]
TriageReason = Literal["off_topic", "low_content"]
RedactReason = Literal["abuse", "identifying", "injection"]
Kind = Literal["majority", "narrow_majority", "plurality", "no_clear_majority"]
KIND_ORDER: tuple[str, ...] = ("majority", "narrow_majority", "plurality", "no_clear_majority")
ThemeKind = Literal["majority", "narrow_majority", "plurality", "minority"]
LabelStatus = Literal["agreed", "contested", "ambiguous"]
ReportStatus = Literal["ok", "too_few_responses"]

RESPONSE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
STANCE_ID_PATTERN = r"^[a-z0-9_]{1,40}$"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class ResponseIn(_Model):
    id: str = Field(pattern=RESPONSE_ID_PATTERN)
    text: str


class PollInput(_Model):
    poll_id: str
    question: str
    sub_questions: list[str] | None = None
    roster_names: list[str] | None = None
    n_invited: int | None = Field(default=None, ge=0)
    n_responses: int | None = None  # present in evalset files; checked, not used
    responses: list[ResponseIn]

    @model_validator(mode="after")
    def _check_responses(self) -> PollInput:
        ids = [r.id for r in self.responses]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate response ids")
        if self.n_responses is not None and self.n_responses != len(self.responses):
            raise ValueError(
                f"n_responses is {self.n_responses} but there are {len(self.responses)} responses"
            )
        return self


# ---------------------------------------------------------------------------
# LLM outputs
# ---------------------------------------------------------------------------


class Stance(_Model):
    id: str = Field(pattern=STANCE_ID_PATTERN)
    label: str
    group: GroupType | None = None
    definition: str


class SubIssue(_Model):
    id: str = Field(pattern=STANCE_ID_PATTERN)
    description: str


class RedactSpan(_Model):
    span: str = Field(min_length=1)
    reason: RedactReason


def check_codebook(sub_issues: list[SubIssue], codebook: dict[str, list[Stance]]) -> None:
    """Structural rules every codebook must meet (spec §4.2)."""
    sub_ids = [s.id for s in sub_issues]
    if len(sub_ids) != len(set(sub_ids)):
        raise ValueError("duplicate sub-issue ids")
    if set(codebook) != set(sub_ids):
        raise ValueError("codebook keys must match the sub-issue ids")
    for sub_id, stances in codebook.items():
        ids = [s.id for s in stances]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate stance ids in sub-issue {sub_id!r}")
        missing = RESERVED_STANCES - set(ids)
        if missing:
            raise ValueError(f"sub-issue {sub_id!r} is missing {sorted(missing)}")
        for s in stances:
            if s.id in RESERVED_STANCES and s.group is not None:
                raise ValueError(f"reserved stance {s.id!r} must have group null")


class DesignOutput(_Model):
    """Call 1."""

    proposed_exclusions: dict[str, TriageReason] = Field(default_factory=dict)  # a proposal only (v2.3)
    redact_spans: dict[str, list[RedactSpan]] = Field(default_factory=dict)
    sub_issues: list[SubIssue] = Field(min_length=1)
    codebook: dict[str, list[Stance]]

    @model_validator(mode="after")
    def _check(self) -> DesignOutput:
        check_codebook(self.sub_issues, self.codebook)
        return self


class Label(_Model):
    """One label from a coding pass (Calls 2a, 2b, 1b).

    ``group`` is the group type of this response, judged from its text (v2.3).
    It is optional in the schema so a missing group fails only this label in
    validation, not the whole call.
    """

    id: str
    sub_issue: str
    stance: str
    group: ResponseGroup | None = None
    evidence: str = ""


class CodingOutput(_Model):
    """Calls 2a and 2b: an exclusion vote, or labels, for every response."""

    exclusions: dict[str, TriageReason] = Field(default_factory=dict)
    labels: list[Label] = Field(default_factory=list)


class OtherPassOutput(_Model):
    """Call 1b: new stances per sub-issue, plus labels for the `other` responses."""

    new_stances: dict[str, list[Stance]] = Field(default_factory=dict)
    labels: list[Label]


class CommonGroundIn(_Model):
    point: str
    ids: list[str]


class WriteOutput(_Model):
    """Call 3."""

    summaries: dict[str, dict[str, str]]
    common_ground: dict[str, list[CommonGroundIn]] = Field(default_factory=dict)


class PrivacyFlag(_Model):
    string: str
    reason: str


class PrivacyOutput(_Model):
    """Call 4."""

    flagged: list[PrivacyFlag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Baseline LLM outputs (spec §7). Lists only, no maps, so they work with
# OpenAI strict structured output; every field is required.
# ---------------------------------------------------------------------------


class BaselineExclusion(_Model):
    id: str
    reason: TriageReason


class BaselineTheme(_Model):
    id: str
    label: str
    group: GroupType | None
    definition: str
    summary: str
    response_ids: list[str]


class BaselineSubIssue(_Model):
    id: str
    description: str
    themes: list[BaselineTheme]
    not_addressed_ids: list[str]


class StrongBaselineOutput(_Model):
    """Strong baseline: every response in a theme or excluded; code counts."""

    exclusions: list[BaselineExclusion]
    sub_issues: list[BaselineSubIssue] = Field(min_length=1)


class NaiveTheme(BaselineTheme):
    count: int


class NaiveSubIssue(_Model):
    id: str
    description: str
    themes: list[NaiveTheme]
    not_addressed_ids: list[str]
    result_kind: Kind
    leading_theme: str | None
    group_kind: Kind | None
    leading_group: Literal["for", "against"] | None


class NaiveBaselineOutput(_Model):
    """Naive baseline: the model also writes the counts and the majority calls."""

    n_respondents: int
    exclusions: list[BaselineExclusion]
    sub_issues: list[NaiveSubIssue] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class Excluded(_Model):
    no_content: int = 0
    off_topic: int = 0
    low_content: int = 0
    no_position: int = 0

    @property
    def total(self) -> int:
        return self.no_content + self.off_topic + self.low_content + self.no_position


class StanceResult(_Model):
    kind: Kind
    stance_id: str | None = None
    robust: bool | None = None
    alt_kinds: list[Kind] | None = None


class GroupResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_by_name=True, validate_by_alias=True, serialize_by_alias=True
    )

    kind: Kind
    group: GroupType | None = None
    robust: bool | None = None
    alt_kinds: list[Kind] | None = None
    for_: int = Field(alias="for", ge=0)
    against: int = Field(ge=0)
    conditional: int = Field(ge=0)
    alternative: int = Field(ge=0)


class Theme(_Model):
    stance_id: str
    label: str
    group: GroupType | None = None
    summary: str
    count: int = Field(ge=0)
    of_total: int = Field(ge=0)
    of_addressed: int = Field(ge=0)
    x_of_y: str
    kind: ThemeKind
    contested: int | None = None
    group_mismatch: int | None = None  # members whose own group differs from this stance's group
    response_ids: list[str] = Field(default_factory=list)


class CommonGround(_Model):
    point: str
    response_ids: list[str]


class SubIssueReport(_Model):
    id: str
    description: str
    not_addressed: int = Field(ge=0)
    addressed: int = Field(ge=0)
    other_count: int = Field(ge=0)
    contested_count: int | None = None
    group_flags: list[str] = Field(default_factory=list)  # stance ids whose group type looks wrong
    stance_result: StanceResult
    group_result: GroupResult | None = None
    themes: list[Theme]
    common_ground: list[CommonGround] = Field(default_factory=list)
    # Internal, for evaluation and audit; stripped from the public view.
    other_ids: list[str] = Field(default_factory=list)
    not_addressed_ids: list[str] = Field(default_factory=list)
    contested_ids: list[str] = Field(default_factory=list)
    response_groups: dict[str, ResponseGroup] = Field(default_factory=dict)


class Meta(_Model):
    run_index: int = 1
    llm_calls: int = 0
    models: dict[str, str] = Field(default_factory=dict)
    base_seed: int | None = None
    cost_usd: float | None = None
    latency_s: float | None = None
    pass_agreement: float | None = None
    group_agreement: float | None = None
    exclusion_agreement: float | None = None
    call1_exclusion_agreement: float | None = None
    other_rate_before: float | None = None
    other_rate_after: float | None = None
    redactions: dict[str, int] = Field(default_factory=dict)
    excluded_ids: dict[str, list[str]] = Field(default_factory=dict)
    withheld_from_summary: list[str] = Field(default_factory=list)
    leak_replacements: list[str] = Field(default_factory=list)
    privacy_replacements: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # repairs made to LLM output, by id or location only


class Report(_Model):
    poll_id: str
    status: ReportStatus
    n_invited: int | None = None
    n_received: int = Field(ge=0)
    n_respondents: int = Field(ge=0)
    excluded: Excluded
    contested_exclusions: int | None = None
    ambiguous_count: int = Field(default=0, ge=0)
    sub_issues: list[SubIssueReport] = Field(default_factory=list)
    overall_summary: str
    meta: Meta = Field(default_factory=Meta)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


class FinalLabel(_Model):
    """A reconciled label for one response on one sub-issue (spec §4.5).

    ``group`` is the response's own group; ``None`` means it was not labelled
    per response (the baselines) and the stance's group is used instead.
    """

    stance: str
    status: LabelStatus
    alt: str | None = None
    group: ResponseGroup | None = None
    group_status: LabelStatus | None = None
    alt_group: ResponseGroup | None = None


class FinalExclusion(_Model):
    """A reconciled exclusion for one response (spec §4.5, v2.3).

    ``alt`` holds pass B's labels when pass B labelled the response instead.
    """

    reason: TriageReason
    status: Literal["agreed", "contested"]
    alt: dict[str, FinalLabel] | None = None
