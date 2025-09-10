export CUDA_VISIBLE_DEVICES=5



# LLM_MODEL="/home/zhtang/hf_models/DeepSeek-R1-Distill-Qwen-7B"
LLM_MODEL="/home/zhtang/hf_models/Qwen/Qwen3-4B"
MODEL_NAME="qwen3_4b"
# PROBE_MODEL_PATH="./new_ckpt_test/ds_r1_7b_focal_muon_lr_0.04/budget_control_probe_best.pt"
PROBE_MODEL_PATH="./new_ckpt_test/ds_qwen3_4b_ce_muon_lr_0.02_new_data/budget_control_probe_best.pt" # new data
# PROBE_MODEL_PATH="./new_ckpt_test/ds_qwen3_4b_ce_muon_lr_0.02_epochs_100_old_data/budget_control_probe_best.pt" # old data
OUTPUT_DIR="./new_results_ds_qwen3_4b_baseline"
BATCH_SIZE=10
EXTRACT_LAYER=-1
# DATASETS=("math" "gsm8k" "aime" "minerva" "gpqa" "aime25" "amc" "mmlu-pro" "olympiadbench")
# SPLITS=("test" "test" "test2024" "test" "test" "test" "test" "test" "test")
# MAX_SAMPLES=(500 1000 500 500 500 500 500 500 700)

# Dataset configurations - each array element corresponds to the same index dataset
DATASETS=("minerva" "gpqa" "aime25" "amc" "mmlu-pro" "olympiadbench")
SPLITS=("test" "test" "test" "test" "test" "test" )
MAX_SAMPLES=( 500 500 500 500 500 700)
# Individual max_length for each dataset (context window)
MAX_LENGTHS=(32768 32768 32768 32768 32768 32768)
# Individual max_new_tokens for each dataset (generation limit)  
MAX_NEW_TOKENS_ARRAY=(32768 32768 32768 32768 32768 32768)

# DATASETS=("aime25")
# SPLITS=("test")
# MAX_SAMPLES=(500)
OUTPUT_DIR="./new_results_qwen3_4b_probe_new_rest_bench_0910"
LOG_DIR="./logs"

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Create a timestamp for the log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${MODEL_NAME}_qwen3_4b_probe_new_rest_bench_${TIMESTAMP}.log"

echo "Logging output to: $LOG_FILE"
echo "Starting inference run at $(date)" | tee "$LOG_FILE"

for i in "${!DATASETS[@]}"; do
    DATASET_NAME=${DATASETS[$i]}
    DATASET_SPLIT=${SPLITS[$i]}
    MAX_SAMPLES_NUM=${MAX_SAMPLES[$i]}
    DATASET_MAX_LENGTH=${MAX_LENGTHS[$i]}
    DATASET_MAX_NEW_TOKENS=${MAX_NEW_TOKENS_ARRAY[$i]}
    OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}_${DATASET_NAME}_probe_${DATASET_MAX_LENGTH}_baseline.json"

    echo "Running inference for dataset: $DATASET_NAME" | tee -a "$LOG_FILE"
    echo "Dataset: $DATASET_NAME, Split: $DATASET_SPLIT, Max samples: $MAX_SAMPLES_NUM" | tee -a "$LOG_FILE"
    echo "Max length: $DATASET_MAX_LENGTH, Max new tokens: $DATASET_MAX_NEW_TOKENS" | tee -a "$LOG_FILE"
    echo "Output path: $OUTPUT_PATH" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"

    python new_code/inference_budget_probe.py \
        --llm_model "$LLM_MODEL" \
        --probe_path "$PROBE_MODEL_PATH" \
        --probe_type mlp \
        --num_classes 3 \
        --dataset_name "$DATASET_NAME" \
        --dataset_split "$DATASET_SPLIT" \
        --output_path "$OUTPUT_PATH" \
        --batch_size $BATCH_SIZE \
        --max_length $DATASET_MAX_LENGTH \
        --extract_layer $EXTRACT_LAYER \
        --max_samples "$MAX_SAMPLES_NUM" \
        --use_vllm \
        --seed 42 \
        --max_new_tokens $DATASET_MAX_NEW_TOKENS \
        --stage2_n_runs 3 \
        --prompt_normal_idx 1 \
        --prompt_too_hard_idx 2 \
        --prompt_too_easy_idx 0 \
        2>&1 | tee -a "$LOG_FILE"

        # # --only_probe_prediction

    echo "Completed dataset $DATASET_NAME at: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

done

echo "All inference runs completed at $(date)" | tee -a "$LOG_FILE"

echo "----------------------------------------------------" | tee -a "$LOG_FILE"
echo "Starting evaluation for all datasets at $(date)" | tee -a "$LOG_FILE"
echo "----------------------------------------------------" | tee -a "$LOG_FILE"

# Run evaluation for each dataset
for DATASET_NAME in "${DATASETS[@]}"; do
    INPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_${DATASET_NAME}_probe_32768_baseline.json"
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_${DATASET_NAME}_evaluation_results.jsonl"
    
    echo "Running evaluation for dataset: $DATASET_NAME" | tee -a "$LOG_FILE"
    echo "Input file: $INPUT_FILE" | tee -a "$LOG_FILE"
    echo "Output file: $OUTPUT_FILE" | tee -a "$LOG_FILE"
    
    python new_code/run_bench.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --pass_k "1,3" \
        --model_path "/home/zhtang/hf_models/IAAR-Shanghai/xVerify-0.5B-I" \
        2>&1 | tee -a "$LOG_FILE"
        
    echo "Completed evaluation for dataset $DATASET_NAME at: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "All evaluation runs completed at $(date)" | tee -a "$LOG_FILE"
