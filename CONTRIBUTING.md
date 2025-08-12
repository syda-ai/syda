# Contributing to **syda**

First off, thanks for taking the time to help make **syda – the Synthetic Data Generation Library** better!  We value every issue report, pull‑request, and idea.

> **TL;DR – Five Quick Rules**
>
> 1. **Open an issue** before a large change so we can discuss design.
> 2. **Fork** the repo and create a topic branch off `main`.
> 3. **Write tests** (pytest) and run `pre‑commit run --all-files`.
> 4. **Sign every commit**: `git commit -s` *(DCO compliance)*.
> 5. **Submit a PR** that passes CI with a clear, single‑sentence title.
>
> That's it!  Read on for the details.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
3. [Project Setup](#project-setup)
4. [Coding & Style Guide](#coding--style-guide)
5. [Testing](#testing)
6. [Pull‑Request Checklist](#pull‑request-checklist)
7. [Issue Reports](#issue-reports)
8. [Commit Message Convention](#commit-message-convention)
9. [Documentation](#documentation)
10. [Community & Code of Conduct](#community--code-of-conduct)

---

## Getting Started

```bash
# 1. Fork the repository on GitHub and clone your fork
$ git clone https://github.com/<your-user>/syda.git
$ cd syda

# 2. Install dev dependencies
$ python -m venv .venv && source .venv/bin/activate
$ pip install -e ".[dev]"

# 3. Install pre‑commit hooks
$ pre-commit install
```

All commands below assume you are in the virtual environment.

---

## Developer Certificate of Origin (DCO)

We use a **DCO** instead of a CLA.  Each commit must be "signed off" to certify origin:

```bash
$ git commit -s -m "feat: add SQLAlchemy FK support"
```

The `-s` flag automatically adds the required `Signed-off-by:` trailer using your Git username and email.  CI will fail if any commit in the PR lacks a proper sign‑off.

---

## Project Setup

| Task                             | Command                 |
| -------------------------------- | ----------------------- |
| Install library in editable mode | `pip install -e ".[dev]`      |
| Run unit tests                   | `pytest`                |
| Run tests with coverage          | `pytest --cov=syda`     |
| Build docs locally               | `mkdocs serve`          |
| Format code                      | `isort . && autopep8 .` |
| Static analysis                  | `flake8 . && mypy .`    |

> **Tip:** all of the above are run automatically in CI; passing locally saves time.

---

## Coding & Style Guide

* **Python ≥ 3.8**.
* Type annotations encouraged; `mypy` runs in CI.
* Public APIs must include docstrings in Google style.
* Avoid breaking API changes; if inevitable, update `CHANGELOG.md` and docs.

---

## Testing

We strive for **>70 % coverage**.  Please:

1. Add or update **pytest** unit tests for every new feature or bugfix.
2. Use factories/fakes—never real credentials or customer data.
3. Run `pytest --cov=syda` before pushing.

---

## Pull‑Request Checklist

* [ ] PR title follows *Conventional Commits* (`feat: …`, `fix: …`, `docs: …`).
* [ ] All commits are signed (`git commit -s`).
* [ ] Branch is up‑to‑date with `main`.
* [ ] `pre‑commit run --all-files` passes.
* [ ] `pytest` and coverage tests pass.
* [ ] Added/updated docstrings and MkDocs pages.
* [ ] If the PR changes the public API, updated `CHANGELOG.md`.
* [ ] Linked the related Issue (e.g., `Fixes #42`).

---

## Issue Reports

Good bug reports help us fix problems faster. Please include:

* **Environment details:** OS, Python version, database drivers.
* **Steps to reproduce** (minimal code snippet or failing test).
* **Expected vs. actual behavior**.
* **Screenshots / stack traces** where applicable.

Feature requests should explain *why* the feature is valuable and any implementation ideas.

---

## Commit Message Convention

We follow **Conventional Commits** to automate changelogs and semantic versioning.

```
<type>[optional scope]: <description>

[optional body]
[optional footer(
```

Common `type` values: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

Example:

```
feat(generator): support YAML schema foreign‑keys

Adds FK resolution when loading a YAML schema so that referential
integrity is preserved automatically.

Fixes #128
```

---

## Documentation

User docs live in `docs/` and are built with **MkDocs + Material**.

```bash
$ mkdocs serve  # live reload at http://127.0.0.1:8000/
```

Large API surfaces should include usage examples.  Images must be SVG or compressed PNG.

---

## Community & Code of Conduct

We adhere to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).  Be respectful, inclusive, and professional.

Join the discussion in **GitHub Discussions** or on Matrix (#syda\:matrix.org).  For security issues, email **[security@syda.ai](mailto:security@syda.ai)** – please **do not** open a public issue.

---

### Thank You ❤️

Your time and expertise make **syda** better for everyone.  We’re excited to review your contribution!
