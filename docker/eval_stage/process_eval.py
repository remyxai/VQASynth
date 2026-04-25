import os
import json
import argparse
from collections import defaultdict
from vqasynth.datasets import Dataloader
from vqasynth.utils import filter_null
from vqasynth.evaluation import Evaluator, ALL_BENCHMARK_NAMES
from vqasynth.benchmarks import (
    BenchmarkRunner,
    format_benchmark_report,
    BENCHMARK_LOADERS,
)


def save_and_push_datasets(dataset, output_dir, target_repo_name, dataloader):
    """
    Save the dataset with evaluation results and push to the hub.
    """
    dataloader.save_to_disk(dataset)
    dataloader.push_to_hub(dataset, f"{target_repo_name}_eval")


def aggregate_report(dataset, benchmarks):
    """
    Aggregate per-example eval results into a dataset-level report.

    For single-benchmark mode, aggregates eval_scores/eval_types.
    For multi-benchmark mode, merges eval_report JSON across examples.
    """
    is_multi = len(benchmarks) > 1

    if is_multi:
        merged = {"benchmarks": defaultdict(lambda: {"scores": [], "categories": defaultdict(list)})}
        total_pairs = 0

        for split in dataset:
            for example in dataset[split]:
                report_str = example.get("eval_report")
                if not report_str:
                    continue
                report = json.loads(report_str)
                total_pairs += report.get("total_qa_pairs", 0)

                for bench_name, bench_data in report.get("benchmarks", {}).items():
                    merged["benchmarks"][bench_name]["scores"].extend(
                        [bench_data["overall_accuracy"]] * bench_data["total_qa_pairs"]
                    )
                    for cat, cat_data in bench_data.get("by_category", {}).items():
                        merged["benchmarks"][bench_name]["categories"][cat].extend(
                            [cat_data["accuracy"]] * cat_data["count"]
                        )

        summary = {"benchmarks": {}, "summary": {}, "total_qa_pairs": total_pairs}
        for bench_name, bdata in merged["benchmarks"].items():
            scores = bdata["scores"]
            overall = sum(scores) / len(scores) if scores else 0.0
            cats = {}
            for cat, cat_scores in bdata["categories"].items():
                cats[cat] = {
                    "accuracy": sum(cat_scores) / len(cat_scores) if cat_scores else 0.0,
                    "count": len(cat_scores),
                }
            summary["benchmarks"][bench_name] = {
                "overall_accuracy": overall,
                "by_category": cats,
            }
            summary["summary"][bench_name] = overall

        return summary

    else:
        all_scores = []
        type_scores = defaultdict(list)
        for split in dataset:
            for example in dataset[split]:
                for score, qa_type in zip(example["eval_scores"], example["eval_types"]):
                    all_scores.append(score)
                    type_scores[qa_type].append(score)

        return {
            "overall_accuracy": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "total_qa_pairs": len(all_scores),
            "by_type": {
                t: {"accuracy": sum(s) / len(s), "count": len(s)}
                for t, s in type_scores.items()
            },
        }


def print_report(summary, benchmarks):
    """Pretty-print the evaluation report for VQASynth internal evaluation."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    if len(benchmarks) > 1 and "benchmarks" in summary:
        print(f"\nTotal QA pairs: {summary['total_qa_pairs']}")
        print(f"\n{'Benchmark':<20} {'Overall Accuracy':>18}")
        print("-" * 40)
        for bench_name, acc in summary.get("summary", {}).items():
            print(f"{bench_name:<20} {acc:>17.1%}")

        for bench_name, bench_data in summary.get("benchmarks", {}).items():
            print(f"\n--- {bench_name} ---")
            print(f"  Overall: {bench_data['overall_accuracy']:.1%}")
            if bench_data.get("by_category"):
                for cat, cat_data in bench_data["by_category"].items():
                    print(f"  {cat:<35} {cat_data['accuracy']:>7.1%}  (n={cat_data['count']})")
    else:
        print(f"\nOverall accuracy: {summary['overall_accuracy']:.1%}")
        print(f"Total QA pairs: {summary['total_qa_pairs']}")
        if summary.get("by_type"):
            print(f"\n{'Question Type':<25} {'Accuracy':>10} {'Count':>8}")
            print("-" * 45)
            for t, data in summary["by_type"].items():
                print(f"{t:<25} {data['accuracy']:>9.1%} {data['count']:>8}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Internal evaluation: score VQASynth predictions against VQASynth GT
# ---------------------------------------------------------------------------

def run_internal_eval(args, benchmark):
    """Evaluate model predictions stored in a VQASynth dataset."""
    dataloader = Dataloader(args.output_dir)

    evaluator = Evaluator(
        prediction_column=args.prediction_column,
        ground_truth_column=args.ground_truth_column,
        use_llm_judge=args.use_llm_judge and bool(args.api_key),
        llm_api_key=args.api_key,
        llm_model=args.model,
        distance_tolerance=args.distance_tolerance,
        benchmark=benchmark,
    )

    dataset = dataloader.load_dataset(args.source_repo_id)

    for col in ["eval_scores", "eval_types", "eval_mean_score", "eval_report"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)

    dataset = dataset.map(evaluator.apply_transform, batched=False)
    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    summary = aggregate_report(dataset, evaluator.benchmarks)
    print_report(summary, evaluator.benchmarks)

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation summary saved to {summary_path}")

    save_and_push_datasets(dataset, args.output_dir, args.target_repo_name, dataloader)
    print("Evaluation step complete")


# ---------------------------------------------------------------------------
# Benchmark evaluation: score predictions against external benchmarks
# ---------------------------------------------------------------------------

def run_benchmark_eval(args, benchmark_names):
    """
    Evaluate predictions against external benchmark datasets.

    Predictions are loaded from a JSON file mapping benchmark name to
    either a dict of {item_id: prediction} or a list of predictions.
    """
    from openai import OpenAI

    llm_client = None
    if args.use_llm_judge and args.api_key:
        llm_client = OpenAI(api_key=args.api_key)

    runner = BenchmarkRunner(
        benchmarks=benchmark_names,
        llm_client=llm_client,
        llm_model=args.model,
    )

    # Load predictions from JSON file
    with open(args.predictions_file, "r") as f:
        predictions_by_benchmark = json.load(f)

    # Build loader kwargs for benchmarks that need paths
    load_kwargs = {}
    if args.spatialscore_path:
        load_kwargs["spatialscore"] = {"data_path": args.spatialscore_path}
    if args.mindcube_path:
        load_kwargs["mindcube"] = {"data_path": args.mindcube_path}

    report = runner.run(predictions_by_benchmark, load_kwargs=load_kwargs)

    # Print and save
    print(format_benchmark_report(report))

    summary_path = os.path.join(args.output_dir, "benchmark_report.json")
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Benchmark report saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate spatial reasoning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate VQASynth predictions using SpatialScore metrics
  python process_eval.py --output_dir ./cache --source_repo_id user/dataset \\
      --target_repo_name user/results --benchmark spatialscore

  # Full report across all metric frameworks
  python process_eval.py --output_dir ./cache --source_repo_id user/dataset \\
      --target_repo_name user/results --benchmark all

  # Evaluate against external benchmark datasets
  python process_eval.py --output_dir ./cache --mode benchmark \\
      --benchmark-dataset omnispatial,space10 \\
      --predictions-file predictions.json

  # Evaluate against all benchmarks with LLM judge
  python process_eval.py --output_dir ./cache --mode benchmark \\
      --benchmark-dataset all --predictions-file predictions.json \\
      --use_llm_judge --api_key sk-...
        """,
    )

    parser.add_argument("--output_dir", type=str, required=True, help="Path to output/cache directory")
    parser.add_argument("--mode", type=str, default="internal", choices=["internal", "benchmark"],
                        help="'internal': score VQASynth predictions; 'benchmark': score against external benchmarks")

    # Internal mode args
    parser.add_argument("--source_repo_id", type=str, default="", help="Source HuggingFace dataset repo id")
    parser.add_argument("--target_repo_name", type=str, default="", help="Target HuggingFace dataset repo id")
    parser.add_argument("--prediction_column", type=str, default="predictions", help="Column with model predictions")
    parser.add_argument("--ground_truth_column", type=str, default="messages", help="Column with ground truth messages")
    parser.add_argument("--benchmark", type=str, default="all",
                        help=f"Scoring framework: {', '.join(ALL_BENCHMARK_NAMES)}, or 'all'")

    # Benchmark mode args
    parser.add_argument("--benchmark-dataset", type=str, default="all",
                        help=f"External benchmarks to evaluate against: {', '.join(BENCHMARK_LOADERS.keys())}, or 'all'")
    parser.add_argument("--predictions-file", type=str, default="",
                        help="JSON file with predictions keyed by benchmark name")
    parser.add_argument("--spatialscore-path", type=str, default="",
                        help="Path to SpatialScore.json (required for SpatialScore benchmark)")
    parser.add_argument("--mindcube-path", type=str, default="",
                        help="Path to MindCube.jsonl (optional, otherwise loads from HuggingFace)")

    # Shared args
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API key for LLM judge")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM judge model")
    parser.add_argument("--use_llm_judge", action="store_true", help="Enable LLM judge fallback")
    parser.add_argument("--distance_tolerance", type=float, default=2.0, help="Ratio tolerance for distance scoring")

    args = parser.parse_args()

    if args.mode == "benchmark":
        # Parse benchmark dataset selection
        if args.benchmark_dataset.lower() == "all":
            benchmark_names = list(BENCHMARK_LOADERS.keys())
        elif "," in args.benchmark_dataset:
            benchmark_names = [b.strip() for b in args.benchmark_dataset.split(",")]
        else:
            benchmark_names = [args.benchmark_dataset]

        if not args.predictions_file:
            parser.error("--predictions-file is required in benchmark mode")

        run_benchmark_eval(args, benchmark_names)

    else:
        # Parse benchmark scoring framework
        if args.benchmark.lower() == "all":
            benchmark = "all"
        elif "," in args.benchmark:
            benchmark = [b.strip() for b in args.benchmark.split(",")]
        else:
            benchmark = args.benchmark

        if not args.source_repo_id:
            parser.error("--source_repo_id is required in internal mode")
        if not args.target_repo_name:
            parser.error("--target_repo_name is required in internal mode")

        run_internal_eval(args, benchmark)
