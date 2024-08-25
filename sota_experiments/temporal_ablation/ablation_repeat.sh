#!/bin/bash

# Iterate through values from 1 to 4 and run the python script
for i in {1..4}
do
    echo "Running for stratified value: $i"
    python base_cmd_v4.py --stratified $i
    echo "Completed for stratified value: $i"
done
