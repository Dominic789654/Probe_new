export CUDA_VISIBLE_DEVICES=3

# Define parameter arrays for looping
muon_lrs=(0.02 )
epochs=(100)
loss_types=("ce")

                # --data_path ./new_data/oracle_selected_results_fixed.jsonl \
# 
# Loop through all combinations
for muon_lr in "${muon_lrs[@]}"; do
    for epoch in "${epochs[@]}"; do
        for loss_type in "${loss_types[@]}"; do
            # Create unique save path for each combination
            save_path="./new_ckpt_test/ds_qwen3_4b_${loss_type}_muon_lr_${muon_lr}_epochs_${epoch}_new_data_only_think_muon"
            mkdir -p $save_path
            
            # Create unique run name
            run_name="ds_qwen3_4b_${loss_type}_muon_lr_${muon_lr}_epochs_${epoch}_old_data"
            
            echo "Starting training with muon_lr=${muon_lr}, epochs=${epoch}, loss_type=${loss_type}"
            
            python new_code/train_budget_probe.py \
                --llm_model /home/zhtang/hf_models/Qwen/Qwen3-4B \
                --save_path $save_path \
                --batch_size 32 \
                --epochs $epoch \
                --val_every_n_epochs 10 \
                --learning_rate 1e-3 \
                --probe_type mlp \
                --loss_type $loss_type \
                --optimizer "muon" \
                --muon_lr $muon_lr \
                --muon_aux_lr 3e-4 \
                --project_name new-probe-training \
                --run_name $run_name \
                --data_path ./analysis/0809_deepmath_sampled_by_original_difficulty_entropy_n10_300_samples_final_evaluation_results.csv \
                --val_split 0.2 \
                --random_seed 42
            
            echo "Completed training with muon_lr=${muon_lr}, epochs=${epoch}, loss_type=${loss_type}"
            echo "----------------------------------------"
        done
    done
done


# Define parameter arrays for looping
# learning_rates=(1e-3 5e-3 1e-4)
# epochs=(300)
# loss_types=("weighted_ce" "ce" "focal")

# # Loop through all combinations
# for lr in "${learning_rates[@]}"; do
#     for epoch in "${epochs[@]}"; do
#         for loss_type in "${loss_types[@]}"; do
#             # Create unique save path for each combination
#             save_path="./new_ckpt_test/ds_qwen3_4b_${loss_type}_lr_${lr}_epochs_${epoch}_new_data_only_think_adamw"
#             mkdir -p $save_path
            
#             # Create unique run name
#             run_name="ds_qwen3_4b_${loss_type}_lr_${lr}_epochs_${epoch}_new_data_only_think_adamw"
            
#             echo "Starting training with lr=${lr}, epochs=${epoch}, loss_type=${loss_type}"
            
#             python new_code/train_budget_probe.py \
#                 --llm_model /home/zhtang/hf_models/Qwen/Qwen3-4B \
#                 --save_path $save_path \
#                 --batch_size 32 \
#                 --epochs $epoch \
#                 --val_every_n_epochs 10 \
#                 --learning_rate $lr \
#                 --probe_type mlp \
#                 --loss_type $loss_type \
#                 --optimizer "adamw" \
#                 --project_name new-probe-training \
#                 --run_name $run_name \
#                 --data_path ./new_data/oracle_selected_results_fixed.jsonl \
#                 --val_split 0.2 \
#                 --random_seed 42
            
#             echo "Completed training with lr=${lr}, epochs=${epoch}, loss_type=${loss_type}"
#             echo "----------------------------------------"
#         done
#     done
# done

# echo "All training runs completed!"

