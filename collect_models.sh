#!/bin/bash

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --runs)
            RUNS_DIR="$2"
            shift # past argument
            shift # past value
            ;;
        --target)
            TARGET_DIR="$2"
            shift
            shift
            ;;
        --prefix)
            PREFIX="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Verify required arguments are provided
if [ -z "$RUNS_DIR" ] || [ -z "$TARGET_DIR" ] || [ -z "$PREFIX" ]; then
    echo "Usage: $0 --runs <runs_directory> --target <target_directory> --prefix <prefix>"
    exit 1
fi

# Loop over subdirectories in RUNS_DIR that start with the given prefix
for dir in "$RUNS_DIR"/"$PREFIX"*; do
    if [ -d "$dir" ] && [ -f "$dir/final_results.json" ]; then
        # Extract the directory basename to use as a prefix for the output file
        base=$(basename "$dir")
        cp "$dir/best.ckpt" "$TARGET_DIR/${base}_best.ckpt"
        echo "Copied $dir/best.ckpt to $TARGET_DIR/${base}_best.ckpt"
    fi
done

zip -r "$TARGET_DIR.zip" "$TARGET_DIR"
echo "Created zip archive: $TARGET_DIR.zip"