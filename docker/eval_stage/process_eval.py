import os
import json
import argparse
from vqasynth.datasets import Dataloader
from vqasynth.utils import filter_null
from vqasynth.evaluation import Evaluator


def save_and_push_datasets(dataset, output_dir, target_repo_name, dataloader):
    """
    Save the dataset with evaluation results and push to the hub.
    """
    dataloader.save_to_disk(dataset)
    dataloader.push_to_hub(dataset, f"{target_repo_name}_eval")


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
    use_mra,
):
    dataloader = Dataloader(output_dir)

    evaluator = Evaluator(
        prediction_column=prediction_column,
        ground_truth_column=ground_truth_column,
        use_llm_judge=use_llm_judge and bool(api_key),
        llm_api_key=api_key,
        llm_model=model,
        distance_tolerance=distance_tolerance,
        use_mra=use_mra,
    )

    dataset = dataloader.load_dataset(source_repo_id)

    # Remove existing eval columns if re-running
    for col in ["eval_scores", "eval_types", "eval_mean_score"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)

    dataset = dataset.map(
        evaluator.apply_transform,
        batched=False,
    )

    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    # Compute and print aggregate metrics
    all_scores = []
    type_scores = {}
    for split in dataset:
        for example in dataset[split]:
            for score, qa_type in zip(example["eval_scores"], example["eval_types"]):
                all_scores.append(score)
                type_scores.setdefault(qa_type, []).append(score)

    summary = {
        "overall_accuracy": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "total_qa_pairs": len(all_scores),
        "by_type": {
            t: {"accuracy": sum(s) / len(s), "count": len(s)}
            for t, s in type_scores.items()
        },
    }

    print(json.dumps(summary, indent=2))

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
    parser.add_argument("--use_mra", action="store_true", help="Use Mean Relative Accuracy for distance scoring")
    args = parser.parse_args()

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
        args.use_mra,
    )
