# Prometheus-vision Judge — score SpaceLLaVA outputs and build a judge dataset

**Status:** experimental. Runnable wiring of `vqasynth.judge_dataset`: reformat
an OpenSpaces-style spatial-VQA dataset into the record shape
[prometheus-eval/prometheus-vision](https://github.com/prometheus-eval/prometheus-vision)
expects (consumed via `llava.eval.model_vqa`), then parse the `[N]` scores out of
the judge feedback into a score distribution and a score-matched dataset ready
for the Hugging Face Hub.

Design context: [VQASynth Issue #28](https://github.com/remyxai/VQASynth/issues/28).
The pure-stdlib reformat + score-parse logic lives in `vqasynth.judge_dataset`
and is unit-tested in `tests/test_judge_dataset.py`; this package only owns I/O
(image materialization, JSONL read/write, the matplotlib histogram, the Hub push
via `vqasynth.datasets`). No changes to the `vqasynth/` core.

## Prerequisites

- **Python 3.10+**
- `vqasynth` installed (`pip install -e .` from the repo root) — provides
  `vqasynth.judge_dataset`, `vqasynth.datasets.Dataloader`, and
  `vqasynth.utils.filter_null`
- The HuggingFace stack + plotting libs: `datasets`, `Pillow`, `matplotlib`
  (already project deps)

The external judge step additionally needs `prometheus-eval/prometheus-vision`
+ `flash-attn` and the `remyxai/SpaceLLaVA` weights (see "External eval").

## Install

```bash
pip install -e .                       # VQASynth core (incl. vqasynth.judge_dataset)
pip install datasets pillow matplotlib
```

## Build the judge-input JSONL

Reformat the dataset and materialize its images, producing the `--question-file`
you hand to the external judge:

```bash
python -m experiments.prometheus_space_judge.run build \
    --dataset remyxai/OpenSpaces --limit 1000 \
    --image-dir openspaces --output openspaces/sample_eval_data.jsonl
```

`--image-dir` is where each row's first image is written as
`{image_dir}/{index}.{image_ext}`; the JSONL `image` field references those same
paths so `llava.eval.model_vqa` can resolve them.

## External eval (maintainer-run, NOT in this package)

Install `prometheus-eval/prometheus-vision` + `flash-attn`, fetch
`remyxai/SpaceLLaVA`, then run the llava eval to produce the answers JSONL:

```bash
python3 -m llava.eval.model_vqa \
    --model-path /path/to/SpaceLLaVA \
    --question-file ./openspaces/sample_eval_data.jsonl \
    --answers-file ./evaluation_results.jsonl \
    --temperature 1.0 --top_p 0.9 --conv-mode vicuna_v1
```

Feed the resulting `--results` file to the `score` subcommand below.

## Parse scores + build the scored dataset

```bash
python -m experiments.prometheus_space_judge.run score \
    --eval openspaces/sample_eval_data.jsonl \
    --results evaluation_results.jsonl \
    --histogram score_histogram.png \
    --push-to-hub <user>/SpaceJudgeDataset
```

This matches the judge inputs against the answers positionally, parses the
`[N]` scores, plots the score distribution (`--histogram`), and writes or pushes
the score-matched OpenSpaces-format dataset (`--output-dataset` for a local
JSONL, or `--push-to-hub` to push to the Hub). `--skip-images` stores image
paths instead of opening PIL images.

## Testing

Structural tests for the reformat + score-parse logic run without CUDA or model
weights (CPU-only, Python 3.10):

```bash
pytest tests/test_judge_dataset.py
```

## Licensing caveat

`prometheus-eval/prometheus-vision` cites **EXPERT**, a non-permissive source.
Treat any dataset built with this judge as carrying that licensing restriction,
and check EXPERT's terms before redistributing.
