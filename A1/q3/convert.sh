#!/bin/bash

# Usage: bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>

if [ "$#" -ne 3 ]; then
    echo "Usage: bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>"
    exit 1
fi

GRAPHS=$1
SUBGRAPHS=$2
OUTPUT_FEATURES=$3

python3 convert.py "$GRAPHS" "$SUBGRAPHS" "$OUTPUT_FEATURES"