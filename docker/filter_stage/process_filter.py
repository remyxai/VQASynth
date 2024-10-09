import os
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.datasets.embeddings import TagFilter

def main(output_dir, include_tags, exclude_tags, confidence_threshold=0.7):
    tag_filter = TagFilter()
    include_tags = include_tags.strip().split(",")
    exclude_tags = exclude_tags.strip().split(",")

    for filename in os.listdir(output_dir):
        if filename.endswith(".pkl"):
            pkl_path = os.path.join(output_dir, filename)
            df = pd.read_pickle(pkl_path)

            df['tag'] = df.apply(
                lambda row: tag_filter.get_best_matching_tag(
                    row['embedding'], include_tags + exclude_tags 
                ),
                axis=1
            )

            df['keep'] = df.apply(
                lambda row: tag_filter.filter_by_tag(
                    row['tag'], include_tags, exclude_tags
                ),
                axis=1
            )

            df_filtered = df[df['keep']]
            df_filtered = df_filtered.drop(columns=['keep'])
            df_filtered.to_pickle(pkl_path)

            print(f"Processed and updated {filename}, filtered out {len(df) - len(df_filtered)} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Depth extraction", add_help=True)
    parser.add_argument("--output_dir", type=str, required=True, help="path to output dataset directory")
    parser.add_argument("--include_tags", type=str, required=False, default=None, help="Comma-separated list of tags to include (optional)")
    parser.add_argument("--exclude_tags", type=str, required=False, default=None, help="Comma-separated list of tags to exclude (optional)")
    args = parser.parse_args()
    main(args.output_dir, args.include_tags, args.exclude_tags)
