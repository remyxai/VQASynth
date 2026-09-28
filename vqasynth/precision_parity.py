"""
Precision Parity Audit — paired cross-precision comparison for spatial VQA.

Adapted from "GHOST-Q: Towards Studying Grounding Hallucinations Overlooked
Under Same-score TradeOffs in Quantized VLMS" (arXiv:2609.29999). Post-training
quantization of a VLM is usually judged by aggregate accuracy and memory
savings; GHOST-Q shows a quantized variant can hold the headline score while
redistributing per-item grounding successes and failures, so two independent
accuracy numbers hide exactly the behavior that changed. The fix is a *paired,
item-by-item* comparison against the full-precision reference.

Given the same benchmark items answered twice — once by the baseline
(full-precision) model and once by a quantized variant (INT8 / NF4, as
produced by ``vqasynth.inference.run_inference_on_benchmark(precision=...)``)
— this module decomposes each pair into:

  * outcome flips       — correct under the baseline only (a grounding success
                          lost to quantization; the paper's overlooked
                          hallucination) or under the quantized model only (a
                          rescue). Net accuracy can round to zero while both
                          counts are large: the *same-score tradeoff*.
  * decision changes    — the stated answer changed even when the outcome did
                          not; a label-free parity signal.
  * flip asymmetry      — an exact two-sided sign test on the discordant
                          pairs, with Benjamini-Hochberg FDR correction when
                          the audit is broken down per category (the paper's
                          paired-effect protocol, parameter-free).

This is an ADAPTED PORT (Mode 2). The paper's same-device A100 memory/latency
profiling and its open-ended AMBER generation-budget audit are intentionally
out of scope — they need a hardware bench and an open-ended-generation
benchmark the repo does not host. Gold alignment and forced-choice decision
extraction reuse :mod:`vqasynth.visual_credit` (which in turn reuses
:mod:`vqasynth.evaluation`), so the audit scores predictions exactly the way
the multi-benchmark evaluation stage already does; ``BenchmarkRunner.
score_parity`` in :mod:`vqasynth.benchmarks` instead plugs the benchmark's
native scorer in as the correctness function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from vqasynth.visual_credit import decisions_differ, gold_aligned

# Aggregate-accuracy window (±2 percentage points) within which GHOST-Q
# considers a quantized variant's headline score "preserved" — precisely the
# regime where per-item flips are the only visible signal.
SAME_SCORE_TOLERANCE = 0.02

# Significance threshold for the paired flip-asymmetry sign test and its
# Benjamini-Hochberg FDR correction.
SIGNIFICANCE_ALPHA = 0.05

# Per-item outcome labels.
BOTH_CORRECT = "both_correct"
BOTH_WRONG = "both_wrong"
BASELINE_ONLY = "baseline_only"      # correct under baseline, wrong quantized
QUANTIZED_ONLY = "quantized_only"    # wrong under baseline, correct quantized
UNPAIRED = "unpaired"                # missing prediction or unscorable answer


# ---------------------------------------------------------------------------
# Parameter-free paired statistics (stdlib only)
# ---------------------------------------------------------------------------

def _binom_cdf(k, n, p=0.5):
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def sign_test_p(baseline_only, quantized_only):
    """
    Two-sided exact binomial sign test on the discordant pairs.

    Under the null (quantization is as likely to fix an item as to break it),
    the direction of the flips is a fair coin. Returns 1.0 when there are no
    discordant pairs.
    """
    n = baseline_only + quantized_only
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * _binom_cdf(min(baseline_only, quantized_only), n))


def benjamini_hochberg(pvalues):
    """
    Benjamini-Hochberg step-up FDR adjustment.

    Maps raw p-values to monotone q-values controlling the false discovery
    rate, the correction GHOST-Q applies across its grid of paired effects.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    q_values = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, pvalues[idx] * m / rank)
        q_values[idx] = running
    return [min(1.0, value) for value in q_values]


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class ParityItemResult:
    """Paired outcome for one item across the two precision conditions."""

    id: object
    baseline_correct: bool | None
    quantized_correct: bool | None
    decision_changed: bool
    outcome: str
    category: str = ""
    question_type: str = ""


@dataclass
class PrecisionParityReport:
    paired: int                              # items scored under both conditions
    total: int                               # items offered to the audit
    baseline_accuracy: float
    quantized_accuracy: float
    net_delta: float                         # quantized - baseline accuracy
    agreement: float                         # same outcome on both conditions
    decision_agreement: float                # same stated decision
    baseline_only_correct: int               # grounding successes lost
    quantized_only_correct: int              # failures rescued
    flip_rate: float                         # discordant / paired
    same_score_tradeoff: bool                # |net_delta| <= tol yet flips > 0
    symmetry_p: float                        # sign test on discordant pairs
    significant_asymmetry: bool              # symmetry_p < SIGNIFICANCE_ALPHA
    per_item: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _normalize_predictions(predictions, items):
    """Accept an id -> prediction dict, or a list aligned with ``items``."""
    if isinstance(predictions, dict):
        return predictions
    return {
        item["id"]: predictions[i]
        for i, item in enumerate(items)
        if i < len(predictions)
    }


def _gold_aligned_item(item, prediction):
    """Default correctness: :func:`vqasynth.visual_credit.gold_aligned`."""
    return gold_aligned(item.get("question", ""), item.get("answer", ""), prediction)


def audit(baseline_predictions, quantized_predictions, items,
          correctness=None, tolerance=SAME_SCORE_TOLERANCE):
    """
    Run the paired cross-precision audit.

    Args:
        baseline_predictions: id -> prediction text from the reference
            (full-precision) model, or a list aligned with ``items``.
        quantized_predictions: same shape, from the quantized variant.
        items: normalized benchmark items with ``id``, ``question``, and
            ``answer`` keys (``category`` / ``question_type`` optional).
        correctness: optional ``(item, prediction) -> bool | None`` callable
            scoring one prediction against its item. ``None`` marks the item
            unscorable (excluded from the pair). Defaults to
            :func:`vqasynth.visual_credit.gold_aligned` over the item's
            question and answer; ``BenchmarkRunner.score_parity`` plugs the
            benchmark's native scorer in here instead.
        tolerance: aggregate-accuracy window for the same-score-tradeoff
            flag (GHOST-Q's ±2 percentage points).

    Returns a :class:`PrecisionParityReport`.
    """
    if correctness is None:
        correctness = _gold_aligned_item

    baseline_map = _normalize_predictions(baseline_predictions, items)
    quantized_map = _normalize_predictions(quantized_predictions, items)

    per_item = []
    paired = both = neither = lost = rescued = changed = 0

    for item in items:
        baseline_pred = baseline_map.get(item["id"])
        quantized_pred = quantized_map.get(item["id"])

        baseline_correct = quantized_correct = None
        decision_changed = False
        outcome = UNPAIRED

        if baseline_pred is not None and quantized_pred is not None:
            baseline_correct = correctness(item, baseline_pred)
            quantized_correct = correctness(item, quantized_pred)

        if baseline_correct is not None and quantized_correct is not None:
            baseline_correct = bool(baseline_correct)
            quantized_correct = bool(quantized_correct)
            decision_changed = decisions_differ(baseline_pred, quantized_pred)
            paired += 1
            changed += int(decision_changed)
            if baseline_correct and quantized_correct:
                outcome = BOTH_CORRECT
                both += 1
            elif baseline_correct:
                outcome = BASELINE_ONLY
                lost += 1
            elif quantized_correct:
                outcome = QUANTIZED_ONLY
                rescued += 1
            else:
                outcome = BOTH_WRONG
                neither += 1

        per_item.append(
            ParityItemResult(
                id=item["id"],
                baseline_correct=baseline_correct,
                quantized_correct=quantized_correct,
                decision_changed=decision_changed,
                outcome=outcome,
                category=item.get("category", ""),
                question_type=item.get("question_type", ""),
            )
        )

    def _rate(num, den):
        return num / den if den else 0.0

    baseline_accuracy = _rate(both + lost, paired)
    quantized_accuracy = _rate(both + rescued, paired)
    net_delta = quantized_accuracy - baseline_accuracy
    discordant = lost + rescued
    symmetry_p = sign_test_p(lost, rescued)

    return PrecisionParityReport(
        paired=paired,
        total=len(per_item),
        baseline_accuracy=baseline_accuracy,
        quantized_accuracy=quantized_accuracy,
        net_delta=net_delta,
        agreement=_rate(both + neither, paired),
        decision_agreement=_rate(paired - changed, paired),
        baseline_only_correct=lost,
        quantized_only_correct=rescued,
        flip_rate=_rate(discordant, paired),
        same_score_tradeoff=(paired > 0 and discordant > 0
                             and abs(net_delta) <= tolerance),
        symmetry_p=symmetry_p,
        significant_asymmetry=symmetry_p < SIGNIFICANCE_ALPHA,
        per_item=per_item,
    )


def breakdown_by(items, report, key_fn):
    """
    Group parity metrics by ``key_fn(item)`` (e.g. by category or question
    type), with the per-group sign test FDR-corrected across the breakdown.

    Returns a mapping ``{key: {count, baseline_accuracy, quantized_accuracy,
    net_delta, baseline_only_correct, quantized_only_correct, flip_rate,
    decision_change_rate, p, q, significant}}``. Intended to be composed with
    :func:`vqasynth.evaluation.classify_question` or the benchmark-native
    ``category`` field, mirroring the paired-effect grid GHOST-Q reports.
    """
    groups = {}
    for item, result in zip(items, report.per_item):
        if result.outcome == UNPAIRED:
            continue
        bucket = groups.setdefault(
            key_fn(item),
            {"n": 0, "baseline": 0, "quantized": 0, "lost": 0, "rescued": 0,
             "changed": 0},
        )
        bucket["n"] += 1
        bucket["baseline"] += int(result.baseline_correct)
        bucket["quantized"] += int(result.quantized_correct)
        bucket["lost"] += int(result.outcome == BASELINE_ONLY)
        bucket["rescued"] += int(result.outcome == QUANTIZED_ONLY)
        bucket["changed"] += int(result.decision_changed)

    keys = sorted(groups, key=str)
    raw_p = [sign_test_p(groups[k]["lost"], groups[k]["rescued"]) for k in keys]
    adjusted_q = benjamini_hochberg(raw_p)

    summary = {}
    for key, p_value, q_value in zip(keys, raw_p, adjusted_q):
        bucket = groups[key]
        count = bucket["n"]
        summary[key] = {
            "count": count,
            "baseline_accuracy": bucket["baseline"] / count,
            "quantized_accuracy": bucket["quantized"] / count,
            "net_delta": (bucket["quantized"] - bucket["baseline"]) / count,
            "baseline_only_correct": bucket["lost"],
            "quantized_only_correct": bucket["rescued"],
            "flip_rate": (bucket["lost"] + bucket["rescued"]) / count,
            "decision_change_rate": bucket["changed"] / count,
            "p": p_value,
            "q": q_value,
            "significant": q_value < SIGNIFICANCE_ALPHA,
        }
    return summary


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_parity_report(report, breakdown=None):
    """
    Format a :class:`PrecisionParityReport` as a human-readable string.

    Mirrors the layout of :func:`vqasynth.benchmarks.format_benchmark_report`.
    ``breakdown`` is an optional mapping from :func:`breakdown_by`.
    """
    lines = [
        "",
        "=" * 70,
        "PRECISION PARITY AUDIT",
        "=" * 70,
        "",
        f"  ({report.paired} paired of {report.total} items)",
        "",
        f"  {'Metric':<38} {'Value':>12}",
        f"  {'-' * 52}",
        f"  {'Baseline accuracy':<38} {report.baseline_accuracy:>11.1%}",
        f"  {'Quantized accuracy':<38} {report.quantized_accuracy:>11.1%}",
        f"  {'Net accuracy delta':<38} {report.net_delta:>+11.1%}",
        f"  {'Outcome agreement':<38} {report.agreement:>11.1%}",
        f"  {'Decision agreement':<38} {report.decision_agreement:>11.1%}",
        f"  {'Correct under baseline only':<38} {report.baseline_only_correct:>12}",
        f"  {'Correct under quantized only':<38} {report.quantized_only_correct:>12}",
        f"  {'Flip rate':<38} {report.flip_rate:>11.1%}",
        f"  {'Flip asymmetry p (sign test)':<38} {report.symmetry_p:>12.4f}",
    ]

    if report.paired:
        lines.append("")
        if report.same_score_tradeoff:
            lines.append(
                f"  SAME-SCORE TRADEOFF: aggregate accuracy held within "
                f"±{SAME_SCORE_TOLERANCE:.0%}"
            )
            lines.append(
                f"  while {report.baseline_only_correct + report.quantized_only_correct}"
                " paired outcomes flipped — the headline"
            )
            lines.append(
                "  score hides the redistribution GHOST-Q measures."
            )
        else:
            lines.append("  No same-score tradeoff detected at the configured tolerance.")
        if report.significant_asymmetry:
            lines.append(
                f"  Flip direction is significant (p = {report.symmetry_p:.4f} "
                f"< {SIGNIFICANCE_ALPHA}): quantization "
                + ("breaks" if report.net_delta < 0 else "rescues")
                + " more items than it "
                + ("rescues" if report.net_delta < 0 else "breaks") + "."
            )

    if breakdown:
        lines.append("")
        lines.append(
            f"  {'Breakdown':<24} {'Base':>7} {'Quant':>7} {'Flips':>6} "
            f"{'p':>8} {'q(FDR)':>8} {'N':>5}"
        )
        lines.append(f"  {'-' * 70}")
        for key in sorted(breakdown, key=str):
            bucket = breakdown[key]
            lines.append(
                f"  {str(key):<24} {bucket['baseline_accuracy']:>6.1%} "
                f"{bucket['quantized_accuracy']:>6.1%} "
                f"{bucket['baseline_only_correct'] + bucket['quantized_only_correct']:>6} "
                f"{bucket['p']:>8.4f} {bucket['q']:>8.4f} {bucket['count']:>5}"
            )

    lines.extend(["", "=" * 70, ""])
    return "\n".join(lines)
