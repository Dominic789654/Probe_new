python new_code/compare_metrics.py \
    --csv_files \
        analysis/0820_Qwen2.5-Math-1.5B_entropy_n10_temp0.6_final_evaluation_results.csv \
        analysis/0820_DeepSeek-R1-Distill-Qwen-1.5B_entropy_n10_temp0.6_final_evaluation_results.csv \
        analysis/0820_Nemotron-Research-Reasoning-Qwen-1.5B_entropy_n10_temp0.6_final_evaluation_results.csv \
    --labels "Pre-training" "Distillation" "RL-tuning" \
    --output_csv analysis/model_comparison.csv \
    --output_plot analysis_plots/model_comparison.pdf