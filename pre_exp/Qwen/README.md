1. 运行三个阶段的微调 run_finetuning.sh 1\2\3
2. 运行三个阶段的Codebleu测试 run_codebleu.sh 1\2\3
CUDA_VISIBLE_DEVICES=7 nohup python -u eval_codebleu.py > log_eval.txt  2>&1 &