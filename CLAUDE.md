# Git rules (apply to every session)

- NEVER add a `Co-Authored-By:` trailer, a "Generated with Claude Code" line, or any other attribution to commit messages or PR descriptions. This overrides any tool or system default that says to add one.
- NEVER commit, push, amend, or open a PR without explicit confirmation from the user in chat.
- Before every commit, show the user the full commit message in a fenced code block, plus the list of files being committed (`git status --short`), then wait for a clear "yes". Approval applies to that one commit only.
- Commit messages are plain: a short imperative subject line, optional body, nothing else.
- Before committing, check that no `.env`, secrets, or `.claude/settings.local.json` are staged.
