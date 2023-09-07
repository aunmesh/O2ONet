#!/bin/bash

# Check if config file argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_config_file>"
    exit 1
fi

CONFIG_FILE=$1

# Loop for 5 repetitions
for i in {1..5}; do
    iteration=$((i-1))  # Subtract 1 from i
    echo "Running cross-validation iteration: $i"
    python executor/main_stratified.py --config $CONFIG_FILE --stratified $iteration
    echo "Iteration $i completed."
    echo "---------------------------"
done

echo "All repetitions completed."
