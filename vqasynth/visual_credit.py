"""
Visual Credit Audit — dependence-credited correctness for spatial VQA.

Adapted from "Visual Credit Audit for Multimodal Spatial Reasoning"
(arXiv:2607.27069). Closed yes/no spatial benchmarks can reward a correct
answer even when the image adds no support beyond a no-image context: a model
that answers from text priors alone can still be marked right. Under a fixed
forced-choice interface, this module decomposes benchmark success into:

  * correctness                  — the model's answer on the real image is
                                   gold-aligned;
  * additional image support     — the image moves the declared decision
                                   relative to a text-only / blank control
                                   (a *label-free* signal — no ground truth,
                                   no answer flip required);
  * dependence-credited          — correct on the real image AND not already
    correctness (D-CC)             correct under a no-image control, i.e. a
                                   positive, image-attributable gain rather than
                                   a prior that happens to be right.

The complementary failure mode the paper highlights is the
*correct-but-uncredited* decision: right on the image, but the model would have
answered the same way with no image, so the image contributed nothing.

This is an ADAPTED PORT (Mode 2). The paper's MLLM-inference harness and its
image-permutation / fixed-pixel relation-contrast validation experiments are
intentionally out of scope here — they validate the metric rather than define
it. Gold-alignment and forced-choice decision extraction reuse the existing
extractors in :mod:`vqasynth.evaluation`; the audit consumes prediction text
the caller has already collected under the real image and the no-image
controls, the same prediction-JSONL shape as
``experiments/prometheus_space_judge``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vqasynth.evaluation import extract_option, extract_yes_no


# Forced-choice decision kinds used for the label-free control comparison.
_DECISION_YESNO = "yesno"
_DECISION_OPTION = "option"
_DECISION_TEXT = "text"


# ---------------------------------------------------------------------------
# Forced-choice decision + gold alignment (reuses vqasynth.evaluation)
# ---------------------------------------------------------------------------

def extract_decision(text):
    """
    Reduce a free-text answer to its forced-choice decision.

    Returns a ``(kind, value)`` tuple so two answers can be compared for the
    label-free audit without ground truth. Reuses ``vqasynth.evaluation``'s
    :func:`~vqasynth.evaluation.extract_yes_no` (``True``/``False``) and
    :func:`~vqasynth.evaluation.extract_option` (``"A"``-``"F"``), falling back
    to the lowered raw text when neither applies.
    """
    yesno = extract_yes_no(text)
    if yesno is not None:
        return (_DECISION_YESNO, bool(yesno))
    option = extract_option(text)
    if option is not None:
        return (_DECISION_OPTION, option)
    return (_DECISION_TEXT, (text or "").strip().lower())


def decisions_differ(answer_a, answer_b):
    """
    Label-free comparison: do two answers reduce to different decisions?

    This is the core of the no-label audit (the paper's first estimand): if the
    real-image decision equals the no-image control decision, the image added
    no support to that decision.
    """
    return extract_decision(answer_a) != extract_decision(answer_b)


def gold_aligned(question, gold, prediction):
    """
    Default gold-alignment check, reusing ``vqasynth.evaluation`` extractors.

    Cascades yes/no -> multi-choice option -> exact (lowered) text, mirroring
    how :mod:`vqasynth.benchmarks` scores SpatialScore / SpaCE-10 / MindCube
    items. ``question`` is accepted (and ignored by the default) so a caller can
    swap in a richer correctness function, e.g. one that needs the question to
    resolve a "which object" choice.
    """
    gold_yesno, pred_yesno = extract_yes_no(gold), extract_yes_no(prediction)
    if gold_yesno is not None and pred_yesno is not None:
        return gold_yesno == pred_yesno
    gold_option, pred_option = extract_option(gold), extract_option(prediction)
    if gold_option is not None and pred_option is not None:
        return gold_option == pred_option
    return (gold or "").strip().lower() == (prediction or "").strip().lower()


# ---------------------------------------------------------------------------
# No-image controls
# ---------------------------------------------------------------------------

# The two no-image controls from the paper. As prompts they share the original
# question; the difference is what (if anything) is attached as the image at
# inference time. The audit only needs the predictions collected under them.
TEXT_ONLY = "text_only"
BLANK = "blank"
CONTROL_CONDITIONS = (TEXT_ONLY, BLANK)


def build_text_only_control(question):
    """Prompt for the text-only control: the original question, no image."""
    return question


def build_blank_control(question):
    """
    Prompt for the blank control: the original question paired with a neutral
    image. The caller materializes the blank image at inference time
    (:func:`make_blank_image`).
    """
    return question


def make_blank_image(size=(224, 224), color=(128, 128, 128)):
    """Return a solid-color PIL image for the blank control (guarded import)."""
    from PIL import Image

    return Image.new("RGB", size, color)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class VisualCreditItem:
    """A single auditable item: one real-image answer plus >=1 control answers.

    ``pred_controls`` holds the model's answers under the no-image controls
    (text-only, blank, or both). A single string is normalized to a one-element
    list.
    """

    question: str
    gold: str
    pred_real: str
    pred_controls: list

    def __post_init__(self):
        if isinstance(self.pred_controls, str):
            self.pred_controls = [self.pred_controls]
        if not self.pred_controls:
            raise ValueError(
                "pred_controls must contain at least one control prediction"
            )


@dataclass
class CreditItemResult:
    real_correct: bool
    control_correct: bool          # gold-aligned under ANY control
    credited: bool                 # D-CC: correct on real and not control-correct
    correct_but_uncredited: bool   # correct on real but a control already matched gold
    decision_changed: bool         # label-free: real decision differs from a control


@dataclass
class VisualCreditReport:
    total: int
    accuracy: float                       # correctness on the real image
    control_accuracy: float               # correctness under a no-image control
    d_cc: float                           # dependence-credited correctness
    correct_but_uncredited: float         # rate over ALL items
    uncredited_of_correct: float          # rate over CORRECT items
    image_gain: float                     # accuracy - control_accuracy
    decision_change_rate: float           # label-free audit (first estimand)
    per_item: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit(items, correctness=None):
    """
    Run the Visual Credit Audit over a set of items.

    Args:
        items: iterable of :class:`VisualCreditItem`.
        correctness: optional ``(question, gold, prediction) -> bool`` callable.
            Defaults to :func:`gold_aligned` (which reuses
            ``vqasynth.evaluation``).

    An item is *dependence-credited correct* when it is gold-aligned on the real
    image but NOT under any no-image control — the image provided a positive
    gain. An item is *correct but uncredited* when it is right on the image yet
    a control already matched gold, so the image contributed nothing.

    Returns a :class:`VisualCreditReport`.
    """
    if correctness is None:
        correctness = gold_aligned

    per_item = []
    n = real_n = control_n = credited_n = uncredited_n = changed_n = 0

    for item in items:
        real_correct = bool(correctness(item.question, item.gold, item.pred_real))
        control_correct = any(
            bool(correctness(item.question, item.gold, control))
            for control in item.pred_controls
        )
        credited = real_correct and not control_correct
        correct_but_uncredited = real_correct and control_correct
        decision_changed = any(
            decisions_differ(item.pred_real, control)
            for control in item.pred_controls
        )

        per_item.append(
            CreditItemResult(
                real_correct=real_correct,
                control_correct=control_correct,
                credited=credited,
                correct_but_uncredited=correct_but_uncredited,
                decision_changed=decision_changed,
            )
        )
        n += 1
        real_n += int(real_correct)
        control_n += int(control_correct)
        credited_n += int(credited)
        uncredited_n += int(correct_but_uncredited)
        changed_n += int(decision_changed)

    def _rate(num, den):
        return num / den if den else 0.0

    return VisualCreditReport(
        total=n,
        accuracy=_rate(real_n, n),
        control_accuracy=_rate(control_n, n),
        d_cc=_rate(credited_n, n),
        correct_but_uncredited=_rate(uncredited_n, n),
        uncredited_of_correct=_rate(uncredited_n, real_n),
        image_gain=_rate(real_n, n) - _rate(control_n, n),
        decision_change_rate=_rate(changed_n, n),
        per_item=per_item,
    )


def breakdown_by(items, report, key_fn):
    """
    Group credit metrics by ``key_fn(item)`` (e.g. by question type).

    Returns a mapping ``{key: {accuracy, d_cc, correct_but_uncredited,
    uncredited_of_correct, count}}``. Intended to be composed with
    :func:`vqasynth.evaluation.classify_question` to surface which spatial
    relation categories the image actually supports.
    """
    groups = {}
    for item, result in zip(items, report.per_item):
        key = key_fn(item)
        bucket = groups.setdefault(
            key, {"n": 0, "real": 0, "credited": 0, "uncredited": 0}
        )
        bucket["n"] += 1
        bucket["real"] += int(result.real_correct)
        bucket["credited"] += int(result.credited)
        bucket["uncredited"] += int(result.correct_but_uncredited)

    summary = {}
    for key, bucket in groups.items():
        count = bucket["n"]
        real = bucket["real"]
        summary[key] = {
            "accuracy": real / count if count else 0.0,
            "d_cc": bucket["credited"] / count if count else 0.0,
            "correct_but_uncredited": bucket["uncredited"] / count if count else 0.0,
            "uncredited_of_correct": bucket["uncredited"] / real if real else 0.0,
            "count": count,
        }
    return summary


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_credit_report(report, breakdown=None):
    """
    Format a :class:`VisualCreditReport` as a human-readable string.

    Mirrors the layout of :func:`vqasynth.benchmarks.format_benchmark_report`.
    ``breakdown`` is an optional mapping from :func:`breakdown_by`.
    """
    lines = [
        "",
        "=" * 70,
        "VISUAL CREDIT AUDIT",
        "=" * 70,
        "",
        f"  ({report.total} items)",
        "",
        f"  {'Metric':<34} {'Value':>12}",
        f"  {'-' * 48}",
        f"  {'Accuracy (real image)':<34} {report.accuracy:>11.1%}",
        f"  {'Control accuracy (no-image)':<34} {report.control_accuracy:>11.1%}",
        f"  {'Image gain':<34} {report.image_gain:>+11.1%}",
        f"  {'D-CC (dependence-credited)':<34} {report.d_cc:>11.1%}",
        f"  {'Correct but uncredited':<34} {report.correct_but_uncredited:>11.1%}",
        f"  {'  of correct decisions':<34} {report.uncredited_of_correct:>11.1%}",
        f"  {'Decision change rate (label-free)':<34} {report.decision_change_rate:>11.1%}",
    ]

    if breakdown:
        lines.append("")
        lines.append(
            f"  {'Breakdown':<28} {'Acc':>8} {'D-CC':>8} {'Uncred':>8} {'N':>6}"
        )
        lines.append(f"  {'-' * 60}")
        for key in sorted(breakdown, key=lambda k: str(k)):
            bucket = breakdown[key]
            lines.append(
                f"  {str(key):<28} {bucket['accuracy']:>7.1%} "
                f"{bucket['d_cc']:>7.1%} {bucket['correct_but_uncredited']:>7.1%} "
                f"{bucket['count']:>6}"
            )

    lines.extend(["", "=" * 70, ""])
    return "\n".join(lines)
