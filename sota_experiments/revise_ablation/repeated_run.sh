#!/bin/bash

counter=0
max_attempts=6

while [ $counter -lt $max_attempts ]; do
    python executor/main_strat_custom_log.py --config "$1" --stratified "$2"
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Script finished successfully!"
        exit 0
    else
        counter=$((counter+1))
        echo "Script crashed with exit code $exit_code. Attempt $counter of $max_attempts." >&2
        if [ $counter -eq $max_attempts ]; then
            echo "Max attempts reached. Not restarting."
            exit 1
        fi
        sleep 1
    fi
done
