#!/usr/bin/bash

# 1. Check if number of arguments is 6
# If not, print usage and exit
# Usage:  bash forest_fire.sh <absolute_path_to_graph> <absolute_path_to_seed_set> <absolute_output_file_path> <k> <n_random_instances> <hops>

if [ "$#" -ne 6 ]; then
    echo "Usage: bash forest_fire.sh <absolute_path_to_graph> <absolute_path_to_seed_set> <absolute_output_file_path> <k> <n_random_instances> <hops>"
    exit 1
fi

# 2. Assign arguments to variables
GRAPH_PATH="$1"
SEED_SET_PATH="$2"
OUTPUT_PATH="$3"
K="$4"
N_RANDOM_INSTANCES="$5"
HOPS="$6"

# 3. Run the forest fire algorithm
# Assuming main.py is in the same directory as this script
# usage: main.py [-h] --dataset_path DATASET_PATH --seed_path SEED_PATH --k K --num_sims NUM_SIMS --hops HOPS --output_path OUTPUT_PATH
python3 main.py --dataset_path "$GRAPH_PATH" --seed_path "$SEED_SET_PATH" --k "$K" --num_sims "$N_RANDOM_INSTANCES" --hops "$HOPS" --output_path "$OUTPUT_PATH"

