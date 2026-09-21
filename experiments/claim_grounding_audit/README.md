# Claim Grounding Audit — do the objects in generated CoT actually localize?

**Status:** experimental. Runnable wiring of `vqasynth.claim_grounding`: take the
reasoning traces the `r1_reasoning` stage writes, lift out the object claims each
trace makes, and verify every claim by **semantic-spatial agreement** — paraphrased
pointing queries about the object should keep finding *something* (semantic
support) and keep localizing it to the *same* image region (QIRV persistence /
overlap / dominance). The two branches are fused with a geometric mean, so a claim
scores high only when both hold; claims below threshold are the hallucination
suspects a curation step would flag or drop before the trace becomes training
data.

Adapted from *Semantic-Spatial Agreement Verification for Mitigating Object
Hallucination in Multimodal Large Language Models*
([arXiv:2609.17269](https://arxiv.org/abs/2609.17269)). The pure verification
logic (claim extraction, paraphrased query construction, the QIRV components,
geometric-mean fusion, the report) lives in `vqasynth.claim_grounding` and is
unit-tested in `tests/test_claim_grounding.py`; this package only owns I/O
(claims/query/prediction JSONL read, report printing). Pointing responses are
parsed with the existing `vqasynth.localize.extract_points_and_descriptions`
contract (Molmo `<point>` tags). No changes to the `vqasynth/` core.

## What this is (and isn't)

This is an **adapted port** of the paper's core mechanism, not a reproduction of
its experiments:

- **Kept at full fidelity** — the two evidence branches: semantic support
  estimated over paraphrased queries, and Query-Induced Regional Verification
  (cross-query region persistence, spatial overlap between query pairs, relative
  candidate dominance against the runner-up region), fused by geometric mean so
  either branch lacking support sinks the score.
- **Substituted / out of scope** — the paper's MLLM-inference and
  prompt-aggregation harness is replaced by the pointing-query JSONL split (the
  same maintainer-run split as `experiments/visual_credit_audit`): you run the
  pointing VLM once per emitted query and feed the responses back in. Claim
  extraction is a parameter-free article-phrase heuristic matched against the
  localize stage's `captions` inventory rather than a tagging model. The paper's
  POPE / CHAIR mitigation benchmarks are intentionally not ported — this is the
  filtering signal, not an evaluation framework.

## Prerequisites

- **Python 3.10+**
- `vqasynth` installed (`pip install -e .` from the repo root) — provides
  `vqasynth.claim_grounding` (and `vqasynth.localize` for response parsing when
  its heavy deps are installed)

## 1. Emit pointing queries for the claims in your traces

Traces JSONL — one record per reasoning trace (`output` is the column the
`r1_reasoning` stage writes; `captions` is the localize stage's object inventory
for the image, optional but recommended):

```json
{"id": "row-12",
 "output": "... the man in the red hat is 2-3 feet from the pallet of boxes ...",
 "captions": ["man wearing a red hat", "stack of wooden pallets with boxes"]}
```

```bash
python -m experiments.claim_grounding_audit.run queries \
    --claims traces.jsonl --output pointing_queries.jsonl
```

This extracts the object claims (mentions that do **not** match `captions` carry
`in_inventory: false` — the prime hallucination suspects) and writes one record
per paraphrased pointing query, with a ready-to-use `prompt` in the same format
`vqasynth.localize.MolmoCaptionLocalizer` uses.

## 2. Collect pointing responses (maintainer-run)

Run your pointing VLM (Molmo in this pipeline) once per `prompt` and collect the
raw `<point ...>` responses:

```json
{"id": "row-12", "claim": "red hat", "query_index": 0,
 "response": "<point x=\"30\" y=\"40\" alt=\"red hat\">",
 "image_w": 640, "image_h": 480}
```

Records may instead carry pre-parsed normalized `points` (`[[x, y], ...]` in
`[0, 1]` frame coordinates) when `vqasynth.localize`'s heavy deps are not
installed where the audit runs.

## 3. Audit

```bash
python -m experiments.claim_grounding_audit.run audit \
    --predictions predictions.jsonl --output grounded.jsonl
```

Prints the grounded-claim rate, mean semantic support / spatial agreement, and
the list of ungrounded claims; `--output` writes one JSON record per claim with
the full component scores and the `grounded` flag (`--threshold` moves the
flagging cutoff, default 0.5).

### Reading the numbers

- **Semantic support** is the share of paraphrases that found *anything* — an
  object that only appears under one wording (an isolated response) stays low.
- **Spatial agreement** is the QIRV branch: persistence (queries agreeing on one
  region), overlap (query pairs landing within one region), dominance (the modal
  region vs the runner-up). Dispersed localizations score near zero.
- **Score** is the geometric mean of the two, so either weak branch sinks it —
  the paper's fusion property.

## Testing

Structural tests for the verification logic run without CUDA or model weights
(CPU-only, Python 3.10):

```bash
pytest tests/test_claim_grounding.py
```
