import os
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from vqasynth.localize import Localizer

localizer = Localizer()

def main(output_dir):
    for filename in os.listdir(output_dir):
        if filename.endswith(".pkl"):
            pkl_path = os.path.join(output_dir, filename)
            df = pd.read_pickle(pkl_path)
            df[["masks", "bboxes", "captions"]] = pd.DataFrame(
                df.apply(lambda row: localizer.run(row["image"]), axis=1).tolist(),
                index=df.index,
            )
            df.to_pickle(pkl_path)
            print(f"Processed and updated {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images from .pkl files", add_help=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to directory containing .pkl files",
    )
    args = parser.parse_args()

    main(args.output_dir)
