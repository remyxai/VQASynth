#!/bin/bash

python3 process_depth.py "$@"

# Signal completion (e.g., by creating a file)
touch depth_done.txt
