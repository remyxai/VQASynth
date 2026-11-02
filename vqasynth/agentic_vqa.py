"""
Agentic visual-understanding benchmark support for the eval stage.

Registers AgentVQA as a fifth benchmark in the multi-benchmark evaluation
stage (``vqasynth.benchmarks``): 1,000 human-verified multiple-choice
questions curated from 23 source datasets and grouped into five agentic
domains — Web Agents, Robotics, Egocentric Videos, Games, and Spatial
Planning. Adapted from *AgentVQA: A Unified Benchmark for Agentic Visual
Understanding* (https://arxiv.org/abs/2610.31494).

The paper's cross-domain analysis attributes most VLM errors to grounding
failures — localizing UI elements, reading motion, tracking state changes,
inferring 3D spatial relations — rather than to task misinterpretation, so the
per-domain breakdown is the number that matters here: the Spatial Planning and
Robotics categories are the failure surface VQASynth's synthetic spatial data
and CoT templates target.

Items normalize to the common schema in ``vqasynth.benchmarks`` and score
through the existing ``vqasynth.evaluation.extract_option`` multi-choice path,
so ``BenchmarkRunner`` / ``load_benchmark`` / the docker eval stage pick the
benchmark up by name once ``register_agentvqa()`` has run.
"""

from vqasynth.evaluation import extract_option

AGENTVQA_BENCHMARK_NAME = "agentvqa"
AGENTVQA_REPO_ID = "AgentVQA/AgentVQA"

AGENTVQA_DOMAINS = [
    "Web Agents",
    "Robotics",
    "Egocentric Videos",
    "Games",
    "Spatial Planning",
]

# Source dataset -> agentic domain, as reported in the AgentVQA paper (23 total).
AGENTVQA_DOMAIN_BY_SUBSET = {
    "AitW": "Web Agents",
    "MONDAY": "Web Agents",
    "Screenspot": "Web Agents",
    "Screenspot-Pro": "Web Agents",
    "ERQA": "Robotics",
    "ManipBench": "Robotics",
    "Robo2VLM": "Robotics",
    "Roborefit": "Robotics",
    "Share-Robot": "Robotics",
    "EgoPlan-Bench": "Egocentric Videos",
    "OpenEQA": "Egocentric Videos",
    "Perception-Test": "Egocentric Videos",
    "VidEgoThink": "Egocentric Videos",
    "VSI-Bench": "Egocentric Videos",
    "Atari": "Games",
    "GameQA": "Games",
    "MarioQA": "Games",
    "VideoGameBunny": "Games",
    "3DSRBench": "Spatial Planning",
    "EmbSpatial-Bench": "Spatial Planning",
    "OmniSpatial": "Spatial Planning",
    "SpaCE-10": "Spatial Planning",
    "Spatial-MM": "Spatial Planning",
}

AGENTVQA_SUBSETS = sorted(AGENTVQA_DOMAIN_BY_SUBSET)


def subsets_for_domains(domains):
    """Map agentic domain name(s) onto the subsets that compose them."""
    wanted = {d.strip() for d in domains}
    unknown = sorted(wanted - set(AGENTVQA_DOMAINS))
    if unknown:
        valid = ", ".join(AGENTVQA_DOMAINS)
        raise ValueError(f"Unknown AgentVQA domain(s): {unknown}. Valid: {valid}")
    return [s for s in AGENTVQA_SUBSETS if AGENTVQA_DOMAIN_BY_SUBSET[s] in wanted]


def _format_option(option):
    """
    Render an answer option as prompt text.

    Pointing options arrive as ``[x, y]`` pixel coordinates; render them as
    ``(x, y)`` so the ``(A)``/``(B)`` option formatting in
    ``vqasynth.inference.run_inference_on_benchmark`` stays letter-answerable.
    """
    if isinstance(option, (list, tuple)):
        return "(" + ", ".join(_format_coord(c) for c in option) + ")"
    return str(option).strip()


def _format_coord(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _entry_images(entry):
    """Embedded still images for a row, or [] when the visual input is a video file."""
    for key in ("Image", "image"):
        value = entry.get(key)
        if value is None:
            continue
        return value if isinstance(value, list) else [value]
    return []


def load_agentvqa(subsets=None, split="train", streaming=False, require_image=True):
    """
    Load the AgentVQA benchmark from HuggingFace (AgentVQA/AgentVQA).

    Each subset is a config with a single split; rows carry an embedded
    ``Image``, a ``Question``, 2-4 ``Options`` and a letter ``Answer``. Rows
    belonging to video subsets reference a separate ``media/`` file instead of
    an embedded still — those are dropped unless ``require_image`` is False,
    since single-image VLM inference cannot consume them.

    Args:
        subsets: Subset (source dataset) names to load, e.g. ["3DSRBench"].
            Defaults to all 23. Two of them reappear in this repo's own
            benchmark stage (OmniSpatial, SpaCE-10) as small re-curated cuts.
        split: Split name (the published benchmark is evaluation-only).
        streaming: If True, iterate items lazily.
        require_image: Drop rows without an embedded still image.

    Returns list of normalized items (see the schema in vqasynth.benchmarks).
    """
    if subsets is None:
        subsets = list(AGENTVQA_SUBSETS)

    unknown = [s for s in subsets if s not in AGENTVQA_DOMAIN_BY_SUBSET]
    if unknown:
        valid = ", ".join(AGENTVQA_SUBSETS)
        raise ValueError(f"Unknown AgentVQA subset(s): {unknown}. Valid: {valid}")

    from datasets import load_dataset  # heavy dep — imported after input validation

    items = []
    skipped_no_image = 0
    for subset in subsets:
        try:
            ds = load_dataset(
                AGENTVQA_REPO_ID, subset, split=split, streaming=streaming
            )
        except Exception as e:
            print(f"Warning: Could not load AgentVQA subset '{subset}': {e}")
            continue

        for index, entry in enumerate(ds):
            images = _entry_images(entry)
            if not images:
                skipped_no_image += 1
                if require_image:
                    continue

            items.append({
                "id": f"{subset}_{index}",
                "question": entry.get("Question") or entry.get("question") or "",
                "answer": str(entry.get("Answer") or entry.get("answer") or "").strip(),
                "question_type": "multi-choice",
                "category": AGENTVQA_DOMAIN_BY_SUBSET[subset],
                "subcategory": subset,
                "options": [
                    _format_option(o)
                    for o in (entry.get("Options") or entry.get("options") or [])
                ],
                "images": images,
                "source": "AgentVQA",
            })

    if skipped_no_image:
        print(
            f"Warning: {skipped_no_image} AgentVQA items have no embedded still image "
            f"(video-file input); {'dropped' if require_image else 'kept with empty images'}."
        )

    return items


def score_agentvqa_item(item, prediction):
    """
    Score an AgentVQA item: exact letter match on the extracted option, the
    same rule the benchmark's own harness applies. Falls back to matching the
    gold option's text (pointing answers are often spelled out as coordinates).
    """
    gt = str(item["answer"]).strip().upper()
    pred = extract_option(prediction)

    if pred:
        return 1.0 if pred == gt else 0.0

    options = item.get("options", [])
    idx = ord(gt) - ord("A")
    if 0 <= idx < len(options):
        return 1.0 if options[idx].lower() in prediction.lower() else 0.0

    return 0.0


def register_agentvqa():
    """
    Register the AgentVQA loader and scorer with ``vqasynth.benchmarks`` so the
    existing evaluation stage can address it by name. Idempotent.

    Returns the registry key ("agentvqa").
    """
    from vqasynth import benchmarks

    benchmarks.BENCHMARK_LOADERS[AGENTVQA_BENCHMARK_NAME] = load_agentvqa
    benchmarks.BENCHMARK_SCORERS[AGENTVQA_BENCHMARK_NAME] = score_agentvqa_item
    return AGENTVQA_BENCHMARK_NAME
