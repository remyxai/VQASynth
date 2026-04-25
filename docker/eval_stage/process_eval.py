import os
import json
import argparse
from collections import defaultdict
from vqasynth.datasets import Dataloader
from vqasynth.utils import filter_null
from vqasynth.evaluation import Evaluator, ALL_BENCHMARK_NAMES


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
        # Merge per-example reports
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

        # Build final summary
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
        # Single benchmark: aggregate from flat eval_scores/eval_types
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
    """Pretty-print the evaluation report."""
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


def main(
    output_dir,
    source_repo_id,
    target_repo_name,
    prediction_column,
    ground_truth_column,
    api_key,
    model,
    use_llm_judge,
    distance_tolerance,
    benchmark,
):
    dataloader = Dataloader(output_dir)

    evaluator = Evaluator(
        prediction_column=prediction_column,
        ground_truth_column=ground_truth_column,
        use_llm_judge=use_llm_judge and bool(api_key),
        llm_api_key=api_key,
        llm_model=model,
        distance_tolerance=distance_tolerance,
        benchmark=benchmark,
    )

    dataset = dataloader.load_dataset(source_repo_id)

    # Remove existing eval columns if re-running
    for col in ["eval_scores", "eval_types", "eval_mean_score", "eval_report"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)

    dataset = dataset.map(
        evaluator.apply_transform,
        batched=False,
    )

    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    # Aggregate and print report
    summary = aggregate_report(dataset, evaluator.benchmarks)
    print_report(summary, evaluator.benchmarks)

    # Save summary to file
    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation summary saved to {summary_path}")

    save_and_push_datasets(dataset, output_dir, target_repo_name, dataloader)
    print("Evaluation step complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to local dataset cache")
    parser.add_argument("--source_repo_id", type=str, required=True, help="Source HuggingFace dataset repo id")
    parser.add_argument("--target_repo_name", type=str, required=True, help="Target HuggingFace dataset repo id")
    parser.add_argument("--prediction_column", type=str, default="predictions", help="Column with model predictions")
    parser.add_argument("--ground_truth_column", type=str, default="messages", help="Column with ground truth messages")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API key for LLM judge")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM judge model")
    parser.add_argument("--use_llm_judge", action="store_true", help="Enable LLM judge fallback")
    parser.add_argument("--distance_tolerance", type=float, default=2.0, help="Ratio tolerance for distance scoring")
    parser.add_argument(
        "--benchmark", type=str, default="all",
        help=f"Benchmark to run: {', '.join(ALL_BENCHMARK_NAMES)}, or 'all' for full report"
    )
    args = parser.parse_args()

    # Parse benchmark: "all", single name, or comma-separated list
    if args.benchmark.lower() == "all":
        benchmark = "all"
    elif "," in args.benchmark:
        benchmark = [b.strip() for b in args.benchmark.split(",")]
    else:
        benchmark = args.benchmark

    main(
        args.output_dir,
        args.source_repo_id,
        args.target_repo_name,
        args.prediction_column,
        args.ground_truth_column,
        args.api_key,
        args.model,
        args.use_llm_judge,
        args.distance_tolerance,
        benchmark,
    )
