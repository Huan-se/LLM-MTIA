1. 运行单个试验直接./run_[tbs]_[dataset].sh
2. 运行参数搜索实验使用batch_run.sh脚本  使用nohup ./batch_run_search.sh >log_search.txt 2>&1 &
3. 使用eval_resultx.py来整理输出结果