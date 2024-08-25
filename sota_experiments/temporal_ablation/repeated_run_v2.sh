#!/bin/bash

max_attempts=6

# Check if at least the config file argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_config_file> [initial_stratified_value]"
    exit 1
fi

CONFIG_FILE="$1"
START_STRATIFIED_VALUE="${2:-0}"  # Default to 0 if not provided

# Loop from the given or default stratified value to 4 (both inclusive)
for stratified_value in $(seq $START_STRATIFIED_VALUE 4); do
    echo "Running with --stratified value: $stratified_value"
    counter=0
    while [ $counter -lt $max_attempts ]; do
        python executor/main_strat_custom_log.py --config "$CONFIG_FILE" --stratified "$stratified_value"
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "Script for --stratified value $stratified_value finished successfully!"
            break
        else
            counter=$((counter+1))
            echo "Script crashed with exit code $exit_code for --stratified value $stratified_value. Attempt $counter of $max_attempts." >&2
            if [ $counter -eq $max_attempts ]; then
                echo "Max attempts reached for --stratified value $stratified_value. Not restarting."
                break
            fi
            sleep 1
        fi
    done
    echo "------------------------------------------"
done

echo "All stratified values completed."
