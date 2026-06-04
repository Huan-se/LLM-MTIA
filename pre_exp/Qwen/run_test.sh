python -u eval_test.py \
    --mode base \
    --base_model_path "./models/Qwen2.5-Coder-1.5B" \
    --output_json "./outputs/eval_results/pure_base.json" \
    --gpu_id "6"

python eval_test.py \
    --mode spliced \
    --base_model_path "./models/Qwen2.5-Coder-1.5B" \
    --suffix_model_path "./outputs/Oracle_Model_Merged" \
    --output_json "./outputs/eval_results/spliced_oracle.json" \
    --gpu_id "6"