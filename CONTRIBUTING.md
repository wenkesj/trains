# Contributing

Install and run checks using the [README](README.md#contributing).

Use Conventional Commits for commit subjects and PR titles:

```text
feat(tracing): add a session adapter
fix!: correct dataset output format
chore: update dependencies
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore`, `revert`. A scope is optional; `!` marks a breaking
change. Add a `BREAKING CHANGE:` footer when explaining a breaking change.

Branch names use the same types followed by a slash and a lowercase kebab-case
description, such as `feat/session-adapter`, `fix/dataset-output`, or
`chore/update-dependencies`.

The PR conventions check validates the branch, PR title, and every commit
subject. Rebase or reword nonconforming commits before merging. PRs target
`main` and require @wenkesj's code owner approval once the staged protection
settings are applied. New changes dismiss previous approvals. Direct pushes
to `main` are prohibited by that protection, including for admins. Use squash
or rebase merges to preserve linear history.

PyPI publishing is disabled. Install from source and keep credentials in an
ignored local `.env` or a secret manager; never commit API keys.

See [automation and protection setup](.github/AUTOMATION.md).
