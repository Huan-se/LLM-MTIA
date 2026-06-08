#!/bin/bash

case "$1" in
    1)
        python eval_codebleu.py \
            --model_path "./outputs/Oracle_Model_Merged" \
            --output_json "./outputs/eval_results/oracle.json"
        ;;
    2)
        python eval_codebleu.py \
            --model_path "./outputs/Phase2_Baseline_Merged" \
            --output_json "./outputs/eval_results/baseline.json"
        ;;
    3)
        python eval_codebleu.py \
            --model_path "./outputs/Phase3_Proposed_Merged" \
            --output_json "./outputs/eval_results/proposed.json"
        ;;
    0)
        echo "Running all evaluations..."
        for model in oracle baseline proposed; do
            case $model in
                oracle)
                    python eval_codebleu.py \
                        --model_path "./outputs/Oracle_Model_Merged" \
                        --output_json "./outputs/eval_results/oracle.json"
                    ;;
                baseline)
                    python eval_codebleu.py \
                        --model_path "./outputs/Phase2_Baseline_Merged" \
                        --output_json "./outputs/eval_results/baseline.json"
                    ;;
                proposed)
                    python eval_codebleu.py \
                        --model_path "./outputs/Phase3_Proposed_Merged" \
                        --output_json "./outputs/eval_results/proposed.json"
                    ;;
            esac
        done
        ;;
    *)
        echo "Usage: $0 {1|2|3|0}"
        echo "  1 : Oracle"
        echo "  2 : Baseline"
        echo "  3 : Proposed"
        echo "  0 : All"
        exit 1
        ;;
esac