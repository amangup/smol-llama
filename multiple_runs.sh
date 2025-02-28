#!/bin/bash

# Arrays of values to test
n_layers_list=(16 24 32)
d_model_list=(576 960)
n_epochs_list=(4)

log_dir="think_attempt2_logs_exp1"
mkdir -p "$log_dir"

# Loop through all combinations
for n_layers in "${n_layers_list[@]}"; do
    for d_model in "${d_model_list[@]}"; do
        for n_epochs in "${n_epochs_list[@]}"; do
            # Create a unique identifier for this run
            run_id="layers${n_layers}_dim${d_model}_epochs${n_epochs}"
            log_file="${log_dir}/${run_id}.log"
            
            # Check if this configuration has already been run
            if [ -f "$log_file" ]; then
                echo "⏩ Skipping existing run: $run_id"
                continue
            fi
            
            echo "Starting run: $run_id"
            
            torchrun --standalone --nproc_per_node=4 run_ddp_train-wiki.py \
                --n_layers "$n_layers" \
                --d_model "$d_model" \
                --n_epochs "$n_epochs" \
                > "${log_dir}/${run_id}.log" 2>&1
            
            # Check if the run was successful
            if [ $? -eq 0 ]; then
                echo "✓ Completed: $run_id"
            else
                echo "✗ Failed: $run_id"
            fi
        done
    done
done

echo "All runs completed. Logs saved in: $log_dir"
