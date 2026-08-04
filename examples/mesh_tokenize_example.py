"""Example: tokenize a 3D mesh into LLaMA-Mesh text tokens for VLM fine-tuning.

Loads a Wavefront OBJ, runs the VQASynth mesh-tokenization pipeline (filter ->
quantize -> rotate -> sort -> emit ``v x y z`` / ``f a b c``), and writes the
result as a ``.txt`` file suitable for text-to-3D instruction tuning.

Requires numpy + scipy (already in VQASynth's requirements). The Objaverse mode
additionally needs ``pip install objaverse``.

Usage:
    # One mesh:
    python examples/mesh_tokenize_example.py --input cow.obj --output cow.txt

    # Every .obj in a directory (per-mesh .txt outputs):
    python examples/mesh_tokenize_example.py --input meshes/ --output tokens/

    # Download and tokenize a sample of Objaverse XL meshes:
    python examples/mesh_tokenize_example.py --objaverse --output tokens/ --sample 10
"""
from __future__ import annotations

import argparse
import os
import random
import sys

# Make `vqasynth` importable when run as a standalone script from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        help="Path to a .obj file, or a directory of .obj files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .txt path (single file) or output directory (dir/objaverse mode).",
    )
    parser.add_argument("--max-faces", type=int, default=500, help="Face budget (default 500).")
    parser.add_argument("--bins", type=int, default=64, help="Quantization bins per axis (default 64).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the rotation augmentation.")
    parser.add_argument(
        "--objaverse",
        action="store_true",
        help="Download a sample of Objaverse XL meshes and tokenize each.",
    )
    parser.add_argument(
        "--sample", type=int, default=10, help="Number of Objaverse meshes to sample (with --objaverse).",
    )
    args = parser.parse_args()

    # Lazy import so --help works without numpy/scipy/objaverse installed.
    from vqasynth.mesh_tokenize import (
        download_and_process_objaverse,
        mesh_to_text,
        process_directory,
        process_mesh_file,
    )

    rng = random.Random(args.seed)

    if args.objaverse:
        download_and_process_objaverse(
            args.output,
            objects_to_sample=args.sample,
            max_faces=args.max_faces,
            bins=args.bins,
            rng=rng,
        )
        print(f"Wrote tokenized Objaverse meshes to {args.output}/")
        return

    if not args.input:
        parser.error("--input is required unless --objaverse is set.")

    if os.path.isdir(args.input):
        records = process_directory(
            args.input, args.output,
            max_faces=args.max_faces, bins=args.bins, rng=rng,
        )
        print(f"Tokenized {len(records)} mesh(es) into {args.output}/")
        for record in records:
            print(f"  {record['id']}: {record['n_vertices']} v, {record['n_faces']} f -> {record['output']}")
        return

    # Single file.
    vertices, faces = process_mesh_file(
        args.input, max_faces=args.max_faces, bins=args.bins, rng=rng,
    )
    text = mesh_to_text(vertices, faces)
    with open(args.output, "w") as f:
        f.write(text)
    print(f"Wrote {len(vertices)} vertices / {len(faces)} faces to {args.output}")


if __name__ == "__main__":
    main()
