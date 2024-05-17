#!/bin/bash

# Total number of data points in the CSV file
total_datapoints=$(wc -l < data/phrases_v1.csv)  # ensure the correct path to your CSV file
datapoints_per_batch=50

# Define the script to run
script_name="run_pipeline.py"

for (( start=1; start<=total_datapoints; start+=datapoints_per_batch ))
do
    end=$((start + datapoints_per_batch - 1))
    if [ $end -gt $total_datapoints ]; then
        end=$total_datapoints
    fi

    echo "Processing datapoints from $start to $end"
    # Run your Python script with the appropriate command-line arguments
    python $script_name --start $start --end $end

    # Sleep for a brief period if needed to ensure everything is cleaned up before the next batch starts
    sleep 5
done

echo "All batches processed."
