import os
import json
import argparse

from openai import OpenAI

from vqasynth.benchmarks import (
    BenchmarkRunner,
    format_benchmark_report,
    BENCHMARK_LOADERS,
)
from vqasynth.inference import run_inference_on_benchmark


def run_eval(args, benchmark_names):
    """
    Run inference with a HuggingFace VLM and score against external benchmarks.
    """
    llm_client = None
    if args.use_llm_judge and args.api_key:
        llm_client = OpenAI(api_key=args.api_key)

    runner = BenchmarkRunner(
        benchmarks=benchmark_names,
        llm_client=llm_client,
        llm_model=args.judge_model,
    )

    report = {"summary": {}, "benchmarks": {}}
    for bname in benchmark_names:
        try:
            items = runner.load(bname)
        except Exception as e:
            print(f"Warning: could not load benchmark '{bname}': {e}")
            continue

        if args.max_items > 0:
            items = items[: args.max_items]

        print(f"Running inference on {bname} ({len(items)} items) with {args.model}...")
        preds = run_inference_on_benchmark(
            model_name=args.model,
            benchmark_items=items,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"  {bname}: {len(preds)} predictions generated")

        # Score against the items we actually ran inference on. We deliberately
        # do not call runner.run() here because it reloads the full benchmark
        # and would score thousands of un-predicted items as zero.
        result = runner.score(bname, items, preds)
        report["benchmarks"][result["benchmark"]] = {
            "overall_accuracy": result["overall_accuracy"],
            "total": result["total"],
            "by_category": result["by_category"],
            "by_subcategory": result["by_subcategory"],
            "per_item": result["per_item"],
        }
        report["summary"][result["benchmark"]] = result["overall_accuracy"]

    print(format_benchmark_report(report))

    summary_path = os.path.join(args.output_dir, "benchmark_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Benchmark report saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a HuggingFace VLM against spatial reasoning benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a model against all four benchmarks
  python process_eval.py --output_dir ./cache \\
      --model remyxai/SpaceThinker-Qwen2.5VL-3B

  # Run against a single benchmark
  python process_eval.py --output_dir ./cache \\
      --model Qwen/Qwen2.5-VL-7B-Instruct --benchmark omnispatial

  # Run against multiple benchmarks (comma-separated)
  python process_eval.py --output_dir ./cache \\
      --model Qwen/Qwen2.5-VL-7B-Instruct --benchmark omnispatial,space10
        """,
    )

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write benchmark_report.json into")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace VLM model slug to evaluate (e.g., remyxai/SpaceThinker-Qwen2.5VL-3B)")
    parser.add_argument("--benchmark", type=str, default="all",
                        help=f"Benchmark(s) to evaluate against: {', '.join(BENCHMARK_LOADERS.keys())}, comma-separated, or 'all'")

    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per response")
    parser.add_argument("--max_items", type=int, default=0,
                        help="Cap items per benchmark (0 = no cap). Useful for smoke tests.")

    parser.add_argument("--api_key", type=str, default="",
                        help="OpenAI API key for LLM judge (required for OmniSpatial scoring)")
    parser.add_argument("--judge_model", type=str, default="gpt-4o",
                        help="LLM judge model")
    parser.add_argument("--use_llm_judge", action="store_true",
                        help="Enable LLM judge fallback for ambiguous outputs")

    args = parser.parse_args()

    if args.benchmark.lower() == "all":
        benchmark_names = list(BENCHMARK_LOADERS.keys())
    elif "," in args.benchmark:
        benchmark_names = [b.strip() for b in args.benchmark.split(",")]
    else:
        benchmark_names = [args.benchmark]

    run_eval(args, benchmark_names)
