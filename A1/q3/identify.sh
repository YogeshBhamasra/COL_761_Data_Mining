#!/bin/bash

# Usage: bash identify.sh <path_graph_dataset> <path_discriminative_subgraphs>

if [ "$#" -ne 2 ]; then
    echo "Usage: bash identify.sh <path_graph_dataset> <path_discriminative_subgraphs>"
    exit 1
fi

GRAPH_DATASET=$1
OUTPUT_SUBGRAPHS=$2

# # Get script directory and activate virtual environment if it exists
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# VENV_DIR="${SCRIPT_DIR}/venv"

# if [ -d "$VENV_DIR" ]; then
#     source "$VENV_DIR/bin/activate"
# fi

python3 identify.py "$GRAPH_DATASET" "$OUTPUT_SUBGRAPHS"