"""Fixed wording used by the headline template and the renderer.

Kept in one place so a majority call is always described the same way.
"""

from __future__ import annotations

KIND_PHRASE = {
    "majority": "a clear majority",
    "narrow_majority": "a narrow majority, not a clear one",
    "plurality": "the largest view but not a majority",
    "no_clear_majority": "no clear majority",
}

GROUP_FALLBACK = {
    "for": "Supports the proposal",
    "against": "Opposes the proposal",
    "conditional": "Supports with conditions",
    "alternative": "Proposes a different approach",
    None: "A distinct view",
}


def respondents(n: int) -> str:
    return "respondent" if n == 1 else "respondents"


def stance_sentence(kind: str, lead: str | None) -> str:
    if kind == "majority":
        return f'A clear majority favours "{lead}".'
    if kind == "narrow_majority":
        return f'"{lead}" has a narrow majority, not a clear one.'
    if kind == "plurality":
        return f'"{lead}" is the largest view but not a majority.'
    return "There is no clear majority."


def unsettled_sentence(lead: str, contested: int, kind: str, alt_kinds: list[str]) -> str:
    base = f'"{lead}" is the largest view, but {contested} {"response was" if contested == 1 else "responses were"} hard to classify'
    if alt_kinds:
        alts = " or ".join(KIND_PHRASE[k] for k in alt_kinds)
        return f"{base}, and the result lies between {KIND_PHRASE[kind]} and {alts}."
    return f"{base}, so the result is not settled."


def group_sentence(kind: str, group: str | None, robust: bool | None) -> str:
    if robust is False:
        return "Comparing those for and against, the result is not settled."
    side = {"for": "in favour", "against": "against"}.get(group or "", "")
    if kind == "majority":
        return f"Comparing those for and against, a clear majority is {side}."
    if kind == "narrow_majority":
        return f"Comparing those for and against, a narrow majority, not a clear one, is {side}."
    if kind == "plurality":
        return f"Comparing those for and against, more are {side}, but not a majority."
    return "Comparing those for and against, there is no clear majority either way."
