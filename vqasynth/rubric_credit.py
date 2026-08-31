"""
Rubric credit — per-proposition partial credit for spatial VQA answers.

Adapted from "V-Rubrics: Visual Faithfulness via Rubric-Based Reinforcement
Learning" (arXiv:2608.25580). A scalar score — or the all-or-nothing scorers in
:mod:`vqasynth.evaluation` — says whether an answer is acceptable; it cannot say
*which* grounded fact was wrong, *which* reasoning step contradicted the answer,
or *which* instruction constraint was missed. This module decomposes a reference
answer into atomic propositions and scores the model's response against each
one, along the paper's three components:

  * VF — Visual Faithfulness. The response asserts the same grounded facts as
    the reference: the object relation, the measured distance, the yes/no
    polarity, the chosen object. Every check routes through the existing
    ``vqasynth.evaluation`` extractors and the benchmarks' ratio tolerance.
  * RC — Reasoning Consistency. The *final* answer does not contradict the
    propositions asserted earlier in the same response (the chain of thought),
    e.g. reasoning "to the left" and concluding "to the right". Vacuous — not
    penalized — when the response carries no separate reasoning prefix.
  * IF — Instruction Following. The response answers in the format its question
    type demands (unit-bearing number / yes-no decision / choice option), keyed
    off ``vqasynth.evaluation.classify_question``.

Every scored proposition keeps the character span of the evidence found in the
response, so a failure localizes to the exact clause that is unsupported rather
than to the whole answer.

This is an ADAPTED PORT (Mode 2). The paper's Gemini-3-Pro rubric annotator is
replaced by this deterministic, parameter-free decomposition + scorer built on
the repo's own extractors; the paper's GRPO training loop, V-Rubrics-50K
dataset and Qwen3-VL SFT checkpoint need post-training infrastructure VQASynth
does not host and are intentionally out of scope. The rubric is delivered as an
*evaluation* signal over the same prediction records as
``experiments/visual_credit_audit`` — the same split that package makes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vqasynth.evaluation import (
    classify_question,
    extract_numeric_with_unit,
    extract_option,
    extract_yes_no,
    normalize_to_cm,
    score_choice,
    score_distance_mra,
    score_yes_no,
)

# Component names, as in the paper.
VF = "VF"
RC = "RC"
IF = "IF"
COMPONENTS = (VF, RC, IF)

# Proposition kinds.
RELATION = "relation"
MEASUREMENT = "measurement"
POLARITY = "polarity"
CHOICE = "choice"
FORMAT = "format"

# Question types whose answer must carry a number with a unit.
_NUMERIC_TYPES = ("distance", "vertical_distance", "horizontal_distance", "measurement")


# ---------------------------------------------------------------------------
# Relation propositions (the predicate / comparative answer templates)
# ---------------------------------------------------------------------------

# Antonym pairs over vqasynth.prompt_templates' predicate + comparative
# vocabulary. A response asserting the antonym of a reference relation is a
# localized Visual-Faithfulness failure, not just a wrong answer.
_RELATION_ANTONYMS = {
    "left": "right", "right": "left",
    "above": "below", "below": "above",
    "front": "behind", "behind": "front",
    "taller": "shorter", "shorter": "taller",
    "larger": "smaller", "smaller": "larger",
    "higher": "lower", "lower": "higher",
    "closer": "farther", "farther": "closer",
}
_RELATION_WORDS = "|".join(_RELATION_ANTONYMS)

# Connectives that may sit between "the <object>" and the relation word in the
# answer templates ("the cup is to the left of the book"). Each repetition
# consumes at least one separator character, so the group cannot match empty.
_CONNECTIVE = (
    r"(?:[\s,\-]+(?:is|are|appears?|seems?|looks?|stands?|sits?|positioned|"
    r"located|found|to|on|at|in|directly|just|way|far|slightly|either|also|"
    r"still|then|the))*"
)

_RELATION_RE = re.compile(
    r"\bthe\s+(?P<a>[A-Za-z][A-Za-z0-9 '\-]*?)"
    + _CONNECTIVE
    + r"\s*(?P<rel>" + _RELATION_WORDS + r")\b"
    + r"(?:\s+(?:of|than)\s+(?:the\s+)?"
    r"(?P<b>[A-Za-z][A-Za-z0-9 '\-]*?)(?=[\s.,!?;]|$))?"
    r"(?:\s+(?:the\s+)?(?P<b2>[A-Za-z][A-Za-z0-9 '\-]*?)(?=[\s.,!?;]|$))?"
    r"[\s.,!?;]*",
    re.IGNORECASE,
)

_LEADING_NOISE = re.compile(r"^(?:or|and|than|of|a|an|the|other)\b\s*", re.IGNORECASE)


def _normalize_object(name):
    """
    Reduce an object mention to a comparable key.

    Handles the template's hedged phrasings — "either to the right of or
    directly aligned with the table" yields ``"table"`` — and strips leading
    connectives left over from a lazy regex match. Applied identically to the
    reference and the response, so both sides land on the same key.
    """
    cleaned = (name or "").strip().strip(".,;:!?").lower()
    for _ in range(3):
        stripped = _LEADING_NOISE.sub("", cleaned).strip()
        if stripped == cleaned:
            break
        cleaned = stripped
    if " the " in f" {cleaned} ":
        cleaned = f" {cleaned} ".rsplit(" the ", 1)[1].strip()
    return cleaned


def _antonym(relation):
    return _RELATION_ANTONYMS.get(relation)


def _comparable_pair(prop_a, prop_b):
    """Do two relation propositions talk about (at least) one shared object?"""
    names_a = [o for o in prop_a.objects if o]
    names_b = [o for o in prop_b.objects if o]
    return any(x == y or x in y or y in x for x in names_a for y in names_b)


# ---------------------------------------------------------------------------
# Atomic propositions
# ---------------------------------------------------------------------------

@dataclass
class Proposition:
    """One atomic claim lifted out of a reference or a model response."""

    kind: str                    # relation | measurement | polarity | choice | format
    text: str                    # canonical wording, for reports
    span: tuple = None           # (start, end) char offsets into the source text
    relation: str = None
    objects: tuple = ()
    value: float = None
    unit: str = None
    polarity: bool = None
    option: str = None


def _measurement_span(text, value, unit):
    """Locate the ``value unit`` occurrence the extractor found, for reports."""
    pattern = re.compile(rf"{re.escape(f'{value:g}')}\s*{re.escape(unit)}", re.IGNORECASE)
    match = pattern.search(text or "")
    return (match.start(), match.end()) if match else None


def decompose(text):
    """
    Decompose free text into its atomic spatial propositions.

    Deterministic and parameter-free: relation clauses come from the predicate
    and comparative templates in :mod:`vqasynth.prompt_templates`; measurements,
    yes/no polarity and multi-choice options come from the existing
    ``vqasynth.evaluation`` extractors. Each relation and measurement keeps the
    character span it was found at, which is what makes a later failure
    localizable.
    """
    text = text or ""
    propositions = []

    for match in _RELATION_RE.finditer(text):
        propositions.append(
            Proposition(
                kind=RELATION,
                text=match.group(0).strip(" ,.;!?"),
                span=(match.start(), match.end()),
                relation=match.group("rel").lower(),
                objects=(
                    _normalize_object(match.group("a")),
                    _normalize_object(match.group("b") or match.group("b2")),
                ),
            )
        )

    measurement = extract_numeric_with_unit(text)
    if measurement is not None:
        value, unit = measurement
        propositions.append(
            Proposition(
                kind=MEASUREMENT,
                text=f"{value:g} {unit}",
                span=_measurement_span(text, value, unit),
                value=value,
                unit=unit,
            )
        )

    polarity = extract_yes_no(text)
    if polarity is not None:
        propositions.append(
            Proposition(kind=POLARITY, text="yes" if polarity else "no", polarity=polarity)
        )

    option = extract_option(text)
    if option is not None:
        propositions.append(Proposition(kind=CHOICE, text=option, option=option))

    return propositions


# The candidate-object parse mirrors vqasynth.evaluation.score_choice, so a
# choice rubric agrees with the repo's existing choice scorer.
_CANDIDATES_RE = re.compile(r"the\s+(.+?)\s+or\s+the\s+(.+?)[?.]")


def _gold_pick(question, gold):
    """The gold-chosen object name for a "which of the X or the Y" question."""
    match = _CANDIDATES_RE.search((question or "").lower())
    if not match:
        return ""
    gold_lower = (gold or "").lower()
    for name in map(_normalize_object, match.groups()):
        if name and name in gold_lower:
            return name
    return ""


def build_rubric(question, gold):
    """
    Decompose the reference (gold) answer into the rubric items to score.

    Falls back to a single choice-by-object-name proposition when the gold
    answer yields no proposition but the question names two candidates — the
    shape of the comparative templates ("Positioned to the left is the table.").
    """
    propositions = decompose(gold)
    if not propositions:
        pick = _gold_pick(question, gold)
        if pick:
            propositions.append(Proposition(kind=CHOICE, text=pick, objects=(pick,)))
    return propositions


# ---------------------------------------------------------------------------
# Reasoning / final-answer split
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CONCLUSION_TAG = re.compile(r"<conclusion>(.*?)</conclusion>", re.DOTALL | re.IGNORECASE)


def split_reasoning(text):
    """
    Split a response into ``(reasoning, final_answer, offset)``.

    ``offset`` is the character offset of ``final_answer`` inside ``text`` (or
    ``None`` if it cannot be located), so proposition spans lifted from the
    final answer can be reported against the full response. Recognizes the
    markers this repo already produces and consumes — ``<think>`` /
    ``<answer>`` (``vqasynth.r1_reasoning``), ``<conclusion>``
    (``vqasynth.evaluation``) and a trailing ``Answer:`` — and falls back to
    treating the last sentence as the final answer.
    """
    text = (text or "").strip()
    if not text:
        return "", "", None

    if "</think>" in text:
        head, _, tail = text.partition("</think>")
        final = _ANSWER_TAG.sub(r"\1", tail).strip()
    else:
        tag = _ANSWER_TAG.search(text) or _CONCLUSION_TAG.search(text)
        if tag:
            head, final = text[: tag.start()], tag.group(1).strip()
        elif "Answer: " in text:
            head, _, final = text.rpartition("Answer: ")
            final = final.strip()
        else:
            sentences = _SENTENCE_SPLIT.split(text)
            head, final = " ".join(sentences[:-1]), sentences[-1]
            offset = len(text) - len(final.strip())
            return head.strip(), final.strip(), offset

    offset = text.find(final)
    return head.strip(), final, None if offset < 0 else offset


# ---------------------------------------------------------------------------
# Component checks
# ---------------------------------------------------------------------------

@dataclass
class RubricItemResult:
    """The credit one proposition earned, on one component."""

    proposition: Proposition
    component: str               # VF | RC | IF
    credit: float                # in [0, 1]; partial credit is the point
    evidence_span: tuple = None  # where in the RESPONSE the evidence was found
    detail: str = ""

    @property
    def satisfied(self):
        return self.credit >= 1.0


def _relation_result(prop, response, component):
    """
    Score one reference relation against the relations the response asserts.

    Reuses :func:`decompose` on the response: asserting the same relation for a
    shared object pair earns full credit, asserting the *antonym* earns none and
    localizes to that clause's span, and asserting nothing comparable earns none
    with no span (the claim is simply unsupported).
    """
    candidates = [p for p in decompose(response) if p.kind == RELATION]
    shared = [p for p in candidates if _comparable_pair(prop, p)]

    for other in shared:
        if other.relation == prop.relation:
            # "not to the left" states the reference relation but denies it —
            # the relation is named, the polarity is carried by the yes/no
            # proposition, so this item stays satisfied and the denial is
            # scored there instead.
            return RubricItemResult(
                prop, component, 1.0, other.span,
                f"asserts '{other.relation}' for the same object pair",
            )
    for other in shared:
        if other.relation == _antonym(prop.relation):
            return RubricItemResult(
                prop, component, 0.0, other.span,
                f"asserts the antonym '{other.relation}' for the same object pair",
            )
    return RubricItemResult(prop, component, 0.0, None, "no comparable relation asserted")


def _choice_span(response, pick):
    index = (response or "").lower().find(pick)
    return None if index < 0 else (index, index + len(pick))


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


# Question type -> the extractor whose presence the answer format requires.
_FORMAT_CHECKS = {
    "distance": extract_numeric_with_unit,
    "vertical_distance": extract_numeric_with_unit,
    "horizontal_distance": extract_numeric_with_unit,
    "measurement": extract_numeric_with_unit,
    "comparison_yn": extract_yes_no,
    "comparison_choice": extract_option,
}
_FORMAT_EXPECTATION = {
    extract_numeric_with_unit: "a number with a unit",
    extract_yes_no: "a yes/no decision",
    extract_option: "a choice option",
}


def _check_instruction_following(response, question_type):
    check = _FORMAT_CHECKS.get(question_type)
    if check is None:
        credit, expectation = float(bool((response or "").strip())), "a non-empty answer"
    else:
        credit = float(check(response or "") is not None)
        expectation = _FORMAT_EXPECTATION[check]
    return RubricItemResult(
        Proposition(kind=FORMAT, text=expectation), IF, credit, None,
        f"{question_type} expects {expectation}",
    )


def _check_reasoning(head, tolerance, props):
    """
    RC: does each final-answer proposition contradict the reasoning prefix?

    ``head`` holds the propositions lifted from the reasoning prefix (their
    spans are already absolute in the full response, since ``split_reasoning``
    returns a prefix of it); ``props`` are the propositions from the final
    answer. Returns one result per final proposition that has a comparable
    partner in the reasoning. A contradiction wins over an agreement so the
    failure is surfaced, and a final proposition with no partner contributes
    nothing — RC measures the validity of the steps actually taken.
    """
    results = []

    for prop in props:
        if prop.kind == RELATION:
            partners = [p for p in head if p.kind == RELATION and _comparable_pair(prop, p)]
            contradict = next(
                (p for p in partners if p.relation == _antonym(prop.relation)), None
            )
            agree = next((p for p in partners if p.relation == prop.relation), None)
            if contradict is not None:
                results.append(RubricItemResult(
                    prop, RC, 0.0, contradict.span,
                    f"reasoning asserts the antonym '{contradict.relation}'",
                ))
            elif agree is not None:
                results.append(RubricItemResult(
                    prop, RC, 1.0, agree.span,
                    f"reasoning asserts '{agree.relation}'",
                ))

        elif prop.kind == MEASUREMENT:
            for other in (p for p in head if p.kind == MEASUREMENT):
                results.append(RubricItemResult(
                    prop, RC,
                    float(_distances_agree(prop, other, tolerance)),
                    other.span,
                    f"reasoning states {other.value:g} {other.unit}",
                ))
                break

        elif prop.kind == POLARITY:
            flipped = next(
                (p for p in head if p.kind == POLARITY and p.polarity != prop.polarity), None
            )
            if flipped is not None:
                results.append(RubricItemResult(
                    prop, RC, 0.0, None,
                    f"reasoning concludes {'yes' if flipped.polarity else 'no'}",
                ))

    return results


def _distances_agree(prop_a, prop_b, tolerance):
    """Ratio-tolerance agreement between two measurements (SpatialScore rule)."""
    a_cm = normalize_to_cm(prop_a.value, prop_a.unit)
    b_cm = normalize_to_cm(prop_b.value, prop_b.unit)
    if a_cm and b_cm:
        return max(a_cm / b_cm, b_cm / a_cm) < tolerance
    return abs(prop_a.value - prop_b.value) < 1e-9


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class RubricItemScore:
    """Rubric credit for one question / gold / response triple."""

    question: str
    gold: str
    response: str
    question_type: str
    items: list = field(default_factory=list)
    scalar: float = None       # the existing all-or-nothing score, for contrast

    def component(self, name):
        return _mean(r.credit for r in self.items if r.component == name)

    @property
    def vf(self):
        return self.component(VF)

    @property
    def rc(self):
        return self.component(RC)

    @property
    def instruction_following(self):
        return self.component(IF)

    @property
    def rubric_score(self):
        return _mean(r.credit for r in self.items)

    def failures(self):
        """Unsatisfied rubric items, with their localized evidence spans."""
        return [r for r in self.items if r.credit < 1.0]


def scalar_score(question, gold, response, tolerance=2.0):
    """
    The repo's existing all-or-nothing score for the same triple.

    Dispatches on ``classify_question`` to the scorers the multi-benchmark
    evaluation stage already ships. Held alongside the rubric so a report can
    show what partial credit buys over the binary signal.
    """
    question_type = classify_question(question)
    if question_type == "comparison_yn":
        return score_yes_no(response, gold)
    if question_type == "comparison_choice":
        return score_choice(response, gold, question)
    if question_type in _NUMERIC_TYPES:
        return score_distance_mra(response, gold)
    return None


def score_item(question, gold, response, tolerance=2.0):
    """
    Score one response against the reference answer's atomic propositions.

    Returns a :class:`RubricItemScore` carrying a :class:`RubricItemResult` per
    (proposition, component) pair. VF items come from the reference
    decomposition, IF from the question type, and RC from comparing the final
    answer against the response's own reasoning prefix.
    """
    question_type = classify_question(question)
    results = []

    for prop in build_rubric(question, gold):
        if prop.kind == RELATION:
            results.append(_relation_result(prop, response, VF))
        elif prop.kind == MEASUREMENT:
            credit = score_distance_mra(response, gold)
            span = next(
                (p.span for p in decompose(response) if p.kind == MEASUREMENT), None
            )
            results.append(RubricItemResult(
                prop, VF, 0.0 if credit is None else credit, span,
                "measured distance vs reference (mean relative accuracy)",
            ))
        elif prop.kind == POLARITY:
            credit = score_yes_no(response, gold)
            results.append(RubricItemResult(
                prop, VF, 0.0 if credit is None else credit, None,
                "yes/no polarity vs reference",
            ))
        elif prop.kind == CHOICE:
            credit = score_choice(response, gold, question)
            results.append(RubricItemResult(
                prop, VF, float(credit or 0.0), _choice_span(response, prop.text),
                "chosen object vs reference",
            ))

    results.append(_check_instruction_following(response, question_type))

    reasoning, final, _offset = split_reasoning(response)
    if reasoning:
        results.extend(
            _check_reasoning(decompose(reasoning), tolerance, decompose(final))
        )

    return RubricItemScore(
        question=question,
        gold=gold,
        response=response,
        question_type=question_type,
        items=results,
        scalar=scalar_score(question, gold, response, tolerance),
    )


@dataclass
class RubricReport:
    """Aggregate rubric credit over a set of scored items."""

    total: int
    vf: float                       # mean over items with >=1 VF proposition
    reasoning_consistency: float    # mean over items with >=1 RC comparison
    instruction_following: float
    rubric_score: float             # mean per-item rubric score
    scalar_accuracy: float          # mean of the existing all-or-nothing score
    propositions_per_item: float
    failure_rate: float             # share of rubric items not fully satisfied
    per_item: list = field(default_factory=list)


def score_items(triples, tolerance=2.0):
    """
    Score an iterable of ``(question, gold, response)`` triples.

    Component means are taken over the items where that component produced at
    least one rubric item, so an item with no chain of thought does not drag
    down reasoning consistency.
    """
    scores = [
        score_item(question, gold, response, tolerance)
        for question, gold, response in triples
    ]

    def _component_mean(name):
        return _mean(
            [score.component(name) for score in scores if score.component(name) is not None]
        )

    all_items = [result for score in scores for result in score.items]
    scalars = [score.scalar for score in scores if score.scalar is not None]

    return RubricReport(
        total=len(scores),
        vf=_component_mean(VF) or 0.0,
        reasoning_consistency=_component_mean(RC) or 0.0,
        instruction_following=_component_mean(IF) or 0.0,
        rubric_score=_mean(score.rubric_score for score in scores) or 0.0,
        scalar_accuracy=_mean(scalars) or 0.0,
        propositions_per_item=_mean(len(score.items) for score in scores) or 0.0,
        failure_rate=1.0 - (_mean(r.credit for r in all_items) or 0.0),
        per_item=scores,
    )


def breakdown_by(scores, key_fn):
    """
    Group rubric credit by ``key_fn(score)`` (e.g. by question type).

    Returns ``{key: {vf, reasoning_consistency, instruction_following,
    rubric_score, scalar_accuracy, count}}``. Intended to be composed with
    :func:`vqasynth.evaluation.classify_question`, the same grouping the
    multi-benchmark evaluation stage reports.
    """
    groups = {}
    for score in scores:
        groups.setdefault(key_fn(score), []).append(score)

    summary = {}
    for key, bucket in groups.items():
        summary[key] = {
            "vf": _mean([s.vf for s in bucket if s.vf is not None]) or 0.0,
            "reasoning_consistency": (
                _mean([s.rc for s in bucket if s.rc is not None]) or 0.0
            ),
            "instruction_following": (
                _mean(
                    [s.instruction_following for s in bucket
                     if s.instruction_following is not None]
                )
                or 0.0
            ),
            "rubric_score": _mean([s.rubric_score for s in bucket]) or 0.0,
            "scalar_accuracy": _mean([s.scalar for s in bucket if s.scalar is not None]) or 0.0,
            "count": len(bucket),
        }
    return summary


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_rubric_report(report, breakdown=None):
    """
    Format a :class:`RubricReport` as a human-readable string.

    Mirrors the layout of :func:`vqasynth.visual_credit.format_credit_report`.
    """
    lines = [
        "",
        "=" * 70,
        "RUBRIC CREDIT (per-proposition partial credit)",
        "=" * 70,
        "",
        f"  ({report.total} items, {report.propositions_per_item:.2f} rubric items each)",
        "",
        f"  {'Metric':<34} {'Value':>12}",
        f"  {'-' * 48}",
        f"  {'VF — visual faithfulness':<34} {report.vf:>11.1%}",
        f"  {'RC — reasoning consistency':<34} {report.reasoning_consistency:>11.1%}",
        f"  {'IF — instruction following':<34} {report.instruction_following:>11.1%}",
        f"  {'-' * 48}",
        f"  {'Rubric score (partial credit)':<34} {report.rubric_score:>11.1%}",
        f"  {'Scalar score (all-or-nothing)':<34} {report.scalar_accuracy:>11.1%}",
        f"  {'Rubric items not satisfied':<34} {report.failure_rate:>11.1%}",
    ]

    if breakdown:
        lines.append("")
        lines.append(
            f"  {'Breakdown':<26} {'VF':>7} {'RC':>7} {'IF':>7} {'Rubric':>8} {'N':>6}"
        )
        lines.append(f"  {'-' * 64}")
        for key in sorted(breakdown, key=str):
            bucket = breakdown[key]
            lines.append(
                f"  {str(key):<26} {bucket['vf']:>6.1%} "
                f"{bucket['reasoning_consistency']:>6.1%} "
                f"{bucket['instruction_following']:>6.1%} "
                f"{bucket['rubric_score']:>7.1%} {bucket['count']:>6}"
            )

    lines.extend(["", "=" * 70, ""])
    return "\n".join(lines)


def format_localizations(score, limit=5):
    """
    Format one item's failures as ``component: claim -> evidence (detail)``.

    This is the error-localization half of the rubric: which proposition
    failed, on which component, and where in the response the offending clause
    sits.
    """
    lines = []
    for result in score.failures()[:limit]:
        where = ""
        if result.evidence_span is not None:
            start, end = result.evidence_span
            clause = score.response[start:end].strip()
            where = f" @[{start}:{end}] \"{clause}\""
        lines.append(
            f"  {result.component}  {result.proposition.kind}: "
            f"\"{result.proposition.text}\"{where} — {result.detail}"
        )
    return "\n".join(lines)
