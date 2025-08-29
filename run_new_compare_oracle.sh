#!/bin/bash

# --- Configuration ---
# Directory containing the ORACLE result files
ORACLE_DIR="new_results_ds_r1_7b_full_baseline_pass3"
# Directory containing YOUR METHOD's result files
# !!! IMPORTANT: Please update this path to point to your method's results directory !!!
METHOD_DIR="new_results_ds_r1_v4.75" # <--- PLEASE CHANGE THIS

# File suffixes
METHOD_SUFFIX="_evaluation_results.jsonl"
ORACLE_SUFFIX="_oracle_results.jsonl"
# Path to the comparison script
SCRIPT_PATH="new_code/compare_to_oracle.py"

# --- Script Body ---

# Check if the python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: The comparison script at $SCRIPT_PATH does not exist."
    exit 1
fi

# Check if the directories exist
if [ ! -d "$ORACLE_DIR" ]; then
    echo "Error: The oracle results directory $ORACLE_DIR does not exist."
    exit 1
fi
if [ ! -d "$METHOD_DIR" ]; then
    echo "Error: The method results directory $METHOD_DIR does not exist."
    echo "Please update the METHOD_DIR variable in the script."
    exit 1
fi

echo "Starting comparison:"
echo "  - Method dir: $METHOD_DIR"
echo "  - Oracle dir: $ORACLE_DIR"
echo "--------------------------------------------------------------------------------"

# Loop through all oracle result files in the directory
for oracle_file in "$ORACLE_DIR"/*"$ORACLE_SUFFIX"; do
    # Check if the oracle file exists to avoid errors with no-match globs
    if [ -e "$oracle_file" ]; then
        # Extract the base filename from the oracle file path (e.g., "model_a_temp_0.5")
        oracle_filename=$(basename "$oracle_file")
        base_name="${oracle_filename%$ORACLE_SUFFIX}"
        
        # Construct the full path to the corresponding method file in the other directory
        method_file="${METHOD_DIR}/${base_name}${METHOD_SUFFIX}"

        # Check if the corresponding method file exists
        if [ ! -f "$method_file" ]; then
            echo "Info: Oracle file '$oracle_file' found, but no matching method file in '$METHOD_DIR'."
            echo "      (Was looking for: '$method_file')"
            echo "--------------------------------------------------------------------------------"
            continue
        fi
        
        echo "Comparing files:"
        echo "  - Method File: $method_file"
        echo "  - Oracle File: $oracle_file"
        echo ""
        
        # Run the python comparison script
        python "$SCRIPT_PATH" --method_file "$method_file" --oracle_file "$oracle_file"
        
        echo "--------------------------------------------------------------------------------"
    fi
done

echo "All comparison pairs have been processed."