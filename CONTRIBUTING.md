# Contributing to MirrorFit

Thanks for your interest in contributing! This project powers real-time virtual
try-on experiences, and we welcome improvements across the whole stack —
computer vision, rendering, the API, web demos, documentation, and tests.

## Getting started

```bash
git clone <repo-url> && cd ar_virtual_tryon
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate
pip install -r requirements.txt
python tools/generate_sample_clothes.py
pytest
```

The full test suite runs headless (synthetic camera, no webcam or MediaPipe
needed) and must pass before opening a PR.

## Where to help

- Issues labelled **`good first issue`** are scoped, self-contained tasks —
  a great entry point.
- Issues labelled **`help wanted`** are larger features where design input is
  welcome — comment on the issue before starting big work.

## Ground rules

1. **Tests first for bug fixes.** Add a failing test that reproduces the bug,
   then fix it. The suite must stay green (`pytest`).
2. **Keep the engine single-owner.** MediaPipe graphs are driven only by the
   engine's worker thread — new components should expose thread-safe snapshot
   methods rather than touching the graphs directly (see `TryOnEngine`).
3. **Match the existing style**: type hints everywhere, docstrings on public
   API, config via `config.yaml` (never hard-code paths or thresholds), and
   logging via `get_logger(__name__)`.
4. **Garment assets:** don't commit third-party branded product photos unless
   you have rights to redistribute them. Procedurally generated samples and
   CC0/own assets are fine.

## Pull requests

- One concern per PR; describe the *why*, not just the *what*.
- Include before/after notes for visual changes (screenshots help a lot here).
- Keep PRs small — under ~400 lines of diff where possible.

Thanks again for helping make virtual try-on better!
