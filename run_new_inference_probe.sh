export CUDA_VISIBLE_DEVICES=7



# LLM_MODEL="/home/zhtang/hf_models/DeepSeek-R1-Distill-Qwen-7B"
LLM_MODEL="/home/zhtang/hf_models/Qwen/Qwen3-4B"
MODEL_NAME="qwen3_4b"
# PROBE_MODEL_PATH="./new_ckpt_test/ds_r1_7b_focal_muon_lr_0.04/budget_control_probe_best.pt"
PROBE_MODEL_PATH="./new_ckpt_test/ds_qwen3_4b_ce_muon_lr_0.02_new_data/budget_control_probe_best.pt"
OUTPUT_DIR="./new_results_ds_qwen3_4b_baseline"
BATCH_SIZE=10
MAX_LENGTH=32768
MAX_NEW_TOKENS=32768
EXTRACT_LAYER=-1
# DATASETS=("math" "gsm8k" "aime" "minerva" "gpqa" "aime25" "amc" "mmlu-pro" "olympiadbench")
# SPLITS=("test" "test" "test2024" "test" "test" "test" "test" "test" "test")
# MAX_SAMPLES=(500 1000 500 500 500 500 500 500 700)

DATASETS=("math" "gsm8k" "aime"  "aime25" )
SPLITS=("test" "test" "test2024" "test" )
MAX_SAMPLES=(500 1000 500 500 )

# DATASETS=("aime25")
# SPLITS=("test")
# MAX_SAMPLES=(500)
OUTPUT_DIR="./new_results_ds_qwen3_4b_probe_test_recommended_params"


mkdir -p $OUTPUT_DIR

for i in "${!DATASETS[@]}"; do
    DATASET_NAME=${DATASETS[$i]}
    DATASET_SPLIT=${SPLITS[$i]}
    MAX_SAMPLES_NUM=${MAX_SAMPLES[$i]}
    OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}_${DATASET_NAME}_probe_32k_baseline.json"

    echo "Running inference for dataset: $DATASET_NAME"

    python new_code/inference_budget_probe.py \
        --llm_model "$LLM_MODEL" \
        --probe_path "$PROBE_MODEL_PATH" \
        --probe_type mlp \
        --num_classes 3 \
        --dataset_name "$DATASET_NAME" \
        --dataset_split "$DATASET_SPLIT" \
        --output_path "$OUTPUT_PATH" \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        --extract_layer $EXTRACT_LAYER \
        --max_samples "$MAX_SAMPLES_NUM" \
        --use_vllm \
        --seed 42 \
        --max_length $MAX_LENGTH \
        --max_new_tokens $MAX_NEW_TOKENS \
        --stage2_n_runs 3 \
        --too_easy_bonus 0.0 \
        --normal_penalty 0.0 \
        --too_hard_penalty 0.0 \
        # --only_probe_prediction

done
