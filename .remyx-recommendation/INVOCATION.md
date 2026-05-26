You are a coding agent implementing a recommendation from the Remyx
Recommendation pipeline (canonical attribution URL: https://github.com/remyxai/mhpd-dpo-training/tree/main/agent).

Read these files in order:
  1. .remyx-recommendation/SPEC.md       — the implementation spec (paper,
                                            why-this-paper, suggested
                                            experiment, team's research-
                                            focus body, abstract)
  2. .remyx-recommendation/PAPER.md      — paper title + abstract
  3. .remyx-recommendation/CONTEXT.md    — team context (recent merges,
                                            if Remyx returned any)
  4. .remyx-recommendation/GUARDRAILS.md — what you may and may not modify

Then look at the existing codebase structure (especially the `vqasynth/`
package and `tests/` directory) to understand the project's conventions
and what actually exists to integrate against.

# Step 1 — decide: PR or Issue

After reading the brief AND inspecting the relevant existing code, decide
whether you can produce a *concrete*, *non-vacuous* scaffold. Open as
ISSUE (not PR) if any of these is true:

  - The paper's contribution requires infrastructure the codebase lacks
    (e.g. a trainer when the repo is inference-only, a dataset format
    the repo never touches).
  - The integration point is too vague to pick a real module / API —
    you'd be inventing the integration rather than slotting into one.
  - The paper requires specific external checkpoints, datasets, or
    services the team clearly doesn't have access to, AND the scaffold
    couldn't usefully exist without them.
  - You searched the codebase for the relevant entry points and found
    nothing reasonable to extend or call into.

If ANY of the above hold, DO NOT WRITE CODE. Instead, write a file at
`.remyx-recommendation/OPEN_AS_ISSUE.md` with this exact shape (Markdown):

```
# Title: short, action-oriented (becomes the Issue title)
Optional one-line subtitle.

## Why this paper is interesting for the team

(2-3 sentences from the spec + your own reading)

## What blocks a clean implementation

(Specifics: missing infra, vague integration point, required external
artifacts, etc. Be concrete about what would need to exist for a real
integration to be drafted.)

## What we'd need to know / decide first

(1-3 questions or decisions the team should resolve before this becomes
implementable.)
```

The orchestrator detects this file and opens a GitHub Issue instead of a
draft PR. No code is committed, no PR is opened, no time is wasted on
scaffolding that would mislead a reviewer.

# Step 2 — only if you DIDN'T write the issue file: implement

Implement the MINIMAL-VIABLE-SCAFFOLDING version of the spec:

- Create one new module under `vqasynth/` (likely `vqasynth/<paper_slug>_integration.py`)
  with:
    * A config dataclass (e.g. `<Paper>Config`) holding the paper's reported
      hyperparameters as defaults
    * A class scaffold for the integration entry point. Keep heavy lifting
      (external checkpoint loading, etc.) as documented TODOs so this PR
      doesn't pretend to do work that requires external dependencies.
    * Any utility functions described in the spec (pixel conversions,
      data adapters, etc.) — implement these concretely.

- Create `tests/test_<paper_slug>_integration.py` with passing tests for
  every utility function you implemented concretely. Stub-test the class
  scaffold (smoke test of the no-checkpoint path returning sensible defaults).

- Append a brief "(Paper Title) Integration (experimental) 🧪" section
  to README.md, attributing the work to Remyx Recommendation via the
  canonical attribution URL above (https://github.com/remyxai/mhpd-dpo-training/tree/main/agent). Do not invent
  a different URL.

Run pytest before declaring done. If tests fail, fix them or scope your
implementation down until they pass. Do not modify files outside the
guardrails allowlist.

When complete, output a one-paragraph SUMMARY of what you actually built.
Be honest about what you stubbed vs implemented.
