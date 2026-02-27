#!/bin/bash
# Loop over all JSON files in the ../configs directory
for cfg in ../configs/*.json; do
    echo "Starting run with config: $cfg"
    # Run the command without backgrounding here, so they execute sequentially
    python3 main.py --cfg_path "$cfg" > "output_$(basename "$cfg" .json).log" 2>&1
done