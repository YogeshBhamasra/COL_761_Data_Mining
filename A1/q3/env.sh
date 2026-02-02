#!/bin/bash

echo "=========================================="
echo "Environment Setup for Q3"
echo "=========================================="

# Install required packages
echo "Installing required Python packages..."
pip3 install --user numpy matplotlib networkx 

# echo ""
# echo "=========================================="
# echo "Checking for Gaston executable..."
# echo "=========================================="

# # Check if Gaston is available
# if [ -f "gaston" ]; then
#     echo "✓ Gaston found"
# else
#     echo "✗ Gaston not found!"
#     echo "Please compile Gaston and place it in this directory"
# fi

# echo ""
echo "Setup complete!"