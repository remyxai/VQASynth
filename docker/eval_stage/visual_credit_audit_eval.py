"""
Visual Credit Audit eval entrypoint.

Sister script to ``process_eval.py``: instead of (only) scoring correctness, it
runs ``vqasynth.visual_credit_audit`` over the ``comparison_yn`` subset of a
benchmark to report dependence-credited correctness (D-CC) and image-credit
metrics — i.e. whether the model's yes/no spatial decisions genuinely rely on
the image versus textual shortcuts. Reuses the benchmark loaders and the loaded
HuggingFace VLM from the existing eval stage; nothing new is loaded.
"""

import argparse
import json
import os

from vqasynth.benchmarks import BENCHMARK_LOADERS
from vqasynth.inference import VLMInference
from vqasynth.visual_credit_audit import (
    format_vca_report,
    run_visual_credit_audit,
    select_comparison_yn,
)


def run_vca_eval(args):
    """Load a benchmark, filter to comparison_yn, run the VCA, write a report."""
    if args.benchmark not in BENCHMARK_LOADERS:
        valid = ", ".join(BENCHMARK_LOADERS.keys())
        raise ValueError(f"Unknown benchmark '{args.benchmark}'. Valid: {valid}")

    items = BENCHMARK_LOADERS[args.benchmark]()
    items = select_comparison_yn(items)
    if args.max_items > 0:
        items = items[: args.max_items]

    print(
        f"VCA on {args.benchmark}: {len(items)} comparison_yn items with "
        f"{args.model} (control={args.control})"
    )

    vlm = VLMInference(args.model, max_new_tokens=args.max_new_tokens)
    report = run_visual_credit_audit(vlm, items, control=args.control)

    print(format_vca_report(report))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "visual_credit_audit_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"VCA report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual Credit Audit on spatial yes/no benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Audit SpaceThinker on SpatialScore's yes/no subset (text-only control)
  python visual_credit_audit_eval.py --output_dir ./cache \\
      --model remyxai/SpaceThinker-Qwen2.5VL-3B --benchmark spatialscore

  # Use a blank uniform image as the null control instead
  python visual_credit_audit_eval.py --output_dir ./cache \\
      --model Qwen/Qwen2.5-VL-7B-Instruct --benchmark spatialscore \\
      --control blank --max_items 200
        """,
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write visual_credit_audit_report.json into")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace VLM model slug to audit")
    parser.add_argument("--benchmark", type=str, default="spatialscore",
                        help=f"One of: {', '.join(BENCHMARK_LOADERS.keys())}")
    parser.add_argument("--max_items", type=int, default=0,
                        help="Cap comparison_yn items (0 = no cap). Useful for smoke tests.")
    parser.add_argument("--control", choices=["text", "blank"], default="text",
                        help="Null control: text-only (default) or blank uniform image")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Reserved for VLMInference compatibility (audit reads logits, not generations)")

    args = parser.parse_args()
    run_vca_eval(args)
