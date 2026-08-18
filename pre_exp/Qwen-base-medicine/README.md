1. 运行三个阶段的微调 run_finetuning.sh 1\2\3 [gpu_id]
2. 运行三个阶段的Codebleu测试 run_codebleu.sh 1\2\3 [gpu_id]
3.  运行三个阶段的Codebleu测试 run_feature_eval.sh 1\2\3 [gpu_id]

nohup ./run_finetuning.sh 2 3 > log_phase2.txt 2>&1 &