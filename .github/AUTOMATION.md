# Automation

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| CI | Pull requests, pushes to `main`, manual | Ruff, Pygate, types, CPU tests, package builds and docs tests/build |
| Dependency security | Lockfile changes, weekly, manual | Native uv audit for every project; npm audit for docs |
| Independent examples | Example/library/tooling changes on PRs and `main`, manual | Per-project Ruff, Pygate, types, CPU tests, and critical-path doctests |
| GitHub Pages | Manual, disabled by default | Publish static docs when explicitly enabled |

Run the same steps locally with `task --list`. Examples:

```sh
task lint
task test:python doctest:python
task doctest:examples
task examples:quality
task example:lint example:type example:test EXAMPLE=examples/harbor
task docs:test docs:pages
task packages:build packages:check
task package:build PACKAGE=trains-sdk
task security:python
```

Each example needs `uv sync --project PATH --locked --all-extras` before local checks.
The example matrix discovers every project, excluding Git submodules and generated files.

`DIST_DIR` and `DOCS_BASE` override output paths and the Pages base.
Python workflows run `uv sync` directly, then `uv run --locked task …`. Lightweight
jobs install Task once with `uv tool install`, then run `task …`. Upload and
deployment actions remain in the workflows.

Workflows use Linux runners, timeouts, cancellation, and three-day artifacts.
GPU tests and hosted provider calls are excluded from CI. Image building is not
part of these workflows.
Dependency audits fail on findings. Python auditing uses uv 0.13.6 with the
`audit-command` preview and frozen lockfiles. Scan coverage follows uv's supported
dependency sources; no custom requirements conversion is used.

Pages requires a public repository, `PAGES_ENABLED=true` in repository variables,
and Settings → Pages → GitHub Actions. The public site is hosted at https://wenkesj.github.io/trains/.
CI checks the `/trains/` build and generated Python code reference without publishing them.

PyPI publishing is disabled. Install from source as described in the project README.
CI still builds and checks distributions; no PyPI credentials are needed.

Actions are enabled for this repository. GitHub Free includes 2,000 private-repo
minutes/month; Pro includes 3,000. Check your remaining allowance under
[Billing and licensing](https://github.com/settings/billing).

## Main branch protection

`CODEOWNERS` assigns all files to @wenkesj. `branch-protection.json` defines:

- Pull requests with at least one approval and required code owner review.
- Stale approvals dismissed after changes; all conversations resolved.
- Required CI checks (`Python quality`, `Docs build`, `PR conventions`) with an up-to-date branch.
- Enforcement for admins, with force pushes and branch deletion disabled.

These files do not activate protection by themselves. After the initial commit
(including CODEOWNERS and the workflows) is on `main` and the checks have run,
a repository admin can apply the prepared settings:

```sh
gh api --method PUT repos/wenkesj/trains/branches/main/protection \
  --input .github/branch-protection.json
```

This replaces existing protection for `main`; review existing settings first.
Private repositories require a GitHub plan supporting branch protection.
Required reviews prevent direct pushes, including by admins. GitHub does not
allow authors to approve their own PRs: a PR authored by @wenkesj will need a
policy change or a different eligible code owner before it can merge.
See [GitHub's protection documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
