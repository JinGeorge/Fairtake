"""Render a report for people to read (spec §5).

Internal fields are stripped: every ``response_ids`` list and ``meta``, except
``run_index``, which is shown when the analysis was run more than once.
"""

from __future__ import annotations

import html
from typing import Any

from pipeline.wording import KIND_PHRASE
from schema.report import Report


def public_dict(report: Report) -> dict[str, Any]:
    data = report.model_dump(mode="json")
    run_index = data.pop("meta", {}).get("run_index", 1)
    if run_index > 1:
        data["run_index"] = run_index
    for sub in data.get("sub_issues", []):
        for theme in sub.get("themes", []):
            theme.pop("response_ids", None)
        for point in sub.get("common_ground", []):
            point.pop("response_ids", None)
    return data


def _theme_line(theme: Any, sub: Any) -> str:
    line = f"{theme.label}: {theme.x_of_y}"
    if sub.not_addressed:
        line += f" ({theme.count} of {theme.of_addressed} who addressed this)"
    return line


def _result_line(sub: Any) -> str:
    result = sub.stance_result
    text = f"Result: {KIND_PHRASE[result.kind]}"
    if result.robust is False:
        text += " (not settled: some responses were hard to classify)"
    return text


def render_text(report: Report) -> str:
    lines = [f"Report: {report.poll_id}", "", report.overall_summary]
    if report.meta.run_index > 1:
        lines += ["", f"Note: this is analysis run {report.meta.run_index}."]
    for sub in report.sub_issues:
        lines += ["", sub.description, _result_line(sub)]
        for theme in sub.themes:
            lines += [f"- {_theme_line(theme, sub)}", f"  {theme.summary}"]
        if sub.common_ground:
            lines.append("Common ground:")
            lines += [f"- {cg.point}" for cg in sub.common_ground]
    return "\n".join(lines) + "\n"


def render_html(report: Report) -> str:
    e = html.escape
    parts = [f"<article class=\"fairtake-report\"><h1>{e(report.poll_id)}</h1>", f"<p>{e(report.overall_summary)}</p>"]
    if report.meta.run_index > 1:
        parts.append(f"<p class=\"run\">This is analysis run {report.meta.run_index}.</p>")
    for sub in report.sub_issues:
        parts.append(f"<section><h2>{e(sub.description)}</h2><p>{e(_result_line(sub))}</p><ul>")
        for theme in sub.themes:
            parts.append(f"<li><strong>{e(_theme_line(theme, sub))}</strong><br>{e(theme.summary)}</li>")
        parts.append("</ul>")
        if sub.common_ground:
            parts.append("<h3>Common ground</h3><ul>")
            parts += [f"<li>{e(cg.point)}</li>" for cg in sub.common_ground]
            parts.append("</ul>")
        parts.append("</section>")
    parts.append("</article>")
    return "\n".join(parts)
