import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # ==========================================
    # 第一阶段：数据提取与统计汇总
    # ==========================================
    print("=== 开始提取并解析实验数据 ===")
    
    # 动态定位结果存放目录（适应在 white-box-inversion 下执行的场景）
    results_dir = "../resultsx"
    if not os.path.exists(results_dir):
        print(f"目录 {results_dir} 不存在！尝试降级至当前目录 ./resultsx")
        results_dir = "./resultsx"
        if not os.path.exists(results_dir):
            print("未找到结果目录，请检查执行路径是否正确。")
            return

    extracted_data = []

    # 遍历 resultsx 下的所有实验目录
    for exp_folder in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue

        # 解析文件夹名称，提取代理模型名称和学习率
        try:
            parts = exp_folder.split('_vs_')
            if len(parts) < 2: 
                continue
            
            remainder = parts[1]
            layer_idx = remainder.find('_layer')
            surrogate_model = remainder[:layer_idx] if layer_idx != -1 else remainder
            
            lr_str = remainder.split('_lr')[-1]
            lr = float(lr_str)
        except Exception as e:
            print(f"跳过无法解析的文件夹名称: {exp_folder}")
            continue

        # 进入子目录寻找权重指标文件
        for sub_folder in os.listdir(exp_path):
            sub_path = os.path.join(exp_path, sub_folder)
            if not os.path.isdir(sub_path):
                continue
                
            pt_file = os.path.join(sub_path, "invert-best.pt")
            if os.path.exists(pt_file):
                try:
                    data = torch.load(pt_file, map_location='cpu')
                    metrics = data.get('evaluation_metrics', {})
                    
                    row = {
                        "Surrogate Model": surrogate_model,
                        "LR": lr,
                        "BLEU Score": metrics.get("bleu_score", 0.0),
                        "ROUGE-L": metrics.get("rougeL_score", 0.0),
                        "Exact Match": metrics.get("exact_match", 0.0),
                        "Token F1": metrics.get("token_set_f1", 0.0),
                        "Final Loss": data['L'][-1].item() if 'L' in data and len(data['L']) > 0 else None
                    }
                    extracted_data.append(row)
                except Exception as e:
                    print(f"读取指标文件失败 {pt_file}: {e}")

    if not extracted_data:
        print("未提取到任何有效数据，请检查是否有成功运行完 5000 步的反转实验。")
        return

    # 转化为 DataFrame 并格式化
    df = pd.DataFrame(extracted_data)
    df = df.sort_values(by=["Surrogate Model", "LR"]).reset_index(drop=True)
    
    # 彻底解决 float_format 报错：采用提前保留四位小数的强截断方案
    df_rounded = df.round(4)

    print("\n=== MTIA 攻击评估结果汇总 (Markdown) ===\n")
    try:
        # 去掉 kwargs，使用兼容性最强的打印方式
        print(df_rounded.to_markdown(index=False))
    except ImportError:
        print("[警告] 环境中未安装 tabulate，使用原生 Pandas 格式打印：")
        print(df_rounded)

    csv_path = os.path.join(results_dir, "mtia_results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 数据已成功提取并保存至: {os.path.abspath(csv_path)}")

    # ==========================================
    # 第二阶段：数据重塑与 Seaborn 可视化
    # ==========================================
    print("\n=== 开始渲染可视化图表 ===")
    
    # 将宽表转化为适配绘图的长表格式
    df_melt = pd.melt(
        df, 
        id_vars=['Surrogate Model', 'LR'],
        value_vars=['BLEU Score', 'ROUGE-L', 'Exact Match', 'Token F1'],
        var_name='Metric', 
        value_name='Value'
    )

    # 映射坐标轴与图例名称
    metric_map = {
        'BLEU Score': 'BLEU', 
        'ROUGE-L': 'Rouge-L', 
        'Exact Match': 'EM', 
        'Token F1': 'F1'
    }
    df_melt['Metric'] = df_melt['Metric'].map(metric_map)

    # 百分比指标放大 100 倍以便于横向可比性
    df_melt.loc[df_melt['Metric'] == 'EM', 'Value'] *= 100
    df_melt.loc[df_melt['Metric'] == 'F1', 'Value'] *= 100

    sns.set_theme(style="darkgrid")
    
    g = sns.catplot(
        x='LR', 
        y='Value', 
        data=df_melt, 
        col='Metric', 
        hue='Surrogate Model', 
        kind='bar', 
        sharey=False, 
        height=4.0, 
        aspect=1.2
    )
    
    g.set_axis_labels("Learning Rate", "Metric Value")
    
    # 填充子图标题
    titles = ['BLEU Score', 'ROUGE-L Score', 'Exact Match (%)', 'Token F1 (%)']
    for ax, title in zip(g.axes[0], titles):
        ax.set_title(title)

    # 添加科研风的网格线阴影映射
    hatches = ['o', 'x', '\\\\', '*', '+', '-'] 
    for i, ax in enumerate(g.axes[0]):
        num_lrs = max(len(df['LR'].unique()), 1)
        for j, thisbar in enumerate(ax.patches):
            num_models = len(df['Surrogate Model'].unique())
            if num_models > 0:
                hatch_idx = (j // num_lrs) % len(hatches)
                thisbar.set_hatch(hatches[hatch_idx])

    # 统一调整刻度格式
    for ax in g.axes[0]:
        plt.setp(ax.get_yticklabels(), fontsize=10)
        ax.tick_params(pad=-4, rotation=0)

    plt.subplots_adjust(hspace=0., wspace=0.25)
    
    # 导出高清图表
    plot_path = os.path.join(results_dir, "mtia_comparison_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 绘图完成！科研对比图表已保存为: {os.path.abspath(plot_path)}")
    
    # 提取第一条成功结果进行反转质量的视觉校验
    print("\n=== 最佳反转文本抽样视觉校验 ===")
    for sub_folder in os.listdir(os.path.join(results_dir, os.listdir(results_dir)[0])):
        sample_pt = os.path.join(results_dir, os.listdir(results_dir)[0], sub_folder, "invert-best.pt")
        if os.path.exists(sample_pt):
             sample_data = torch.load(sample_pt, map_location='cpu')
             if 'invert_text' in sample_data and len(sample_data['invert_text']) > 0:
                 print(f"抽样模型目录: {os.listdir(results_dir)[0]}")
                 print(f"反转片段: {sample_data['invert_text'][0][:250]}...\n")
             break

import os
import json
import torch
from transformers import AutoTokenizer
import utils

def main1():
    print("=== 开始生成输入与逆向输入视觉对比报告 ===")
    
    # 动态定位结果存放目录
    results_dir = "../resultsx"
    if not os.path.exists(results_dir):
        results_dir = "./resultsx"
        if not os.path.exists(results_dir):
            print("找不到 resultsx 目录，请确保在 white-box-inversion 目录下运行本脚本。")
            return

    dataset_cache = {}
    tokenizer_cache = {}
    report_lines = ["# MTIA 输入反转对比报告\n\n"]

    # 遍历所有的实验结果文件夹
    for exp_folder in sorted(os.listdir(results_dir)):
        exp_path = os.path.join(results_dir, exp_folder)
        if not os.path.isdir(exp_path): continue

        for sub_folder in os.listdir(exp_path):
            sub_path = os.path.join(exp_path, sub_folder)
            if not os.path.isdir(sub_path): continue

            args_file = os.path.join(sub_path, "args.json")
            pt_file = os.path.join(sub_path, "invert-best.pt")

            if os.path.exists(args_file) and os.path.exists(pt_file):
                with open(args_file, 'r') as f:
                    args = json.load(f)

                dataset_name = args.get('dataset')
                target_model = args.get('target_model')
                data_range = args.get('range')

                if not dataset_name or not target_model: continue

                # 加载分词器 (缓存以防重复加载)
                if target_model not in tokenizer_cache:
                    print(f"正在加载 Tokenizer: {target_model} ...")
                    llm_path = utils.LLM_PATH[target_model]
                    tokenizer_cache[target_model] = AutoTokenizer.from_pretrained(llm_path)
                tokenizer = tokenizer_cache[target_model]
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # 读取并按 Token 长度排序原始数据集 (还原实验时的真实数据分布)
                if dataset_name not in dataset_cache:
                    print(f"正在解析原始数据集: {dataset_name} (依数据集大小可能需要十秒钟) ...")
                    raw_texts = utils.get_list_invert_text(dataset_name)
                    sorted_texts = sorted(raw_texts, key=lambda x: len(tokenizer.encode(x, add_special_tokens=False)), reverse=True)
                    dataset_cache[dataset_name] = sorted_texts

                # 截取范围内的原始文本，并执行与 attack_batch.py 相同的 128 Token 截断对齐
                start, end = map(int, data_range.split(':'))
                original_texts = dataset_cache[dataset_name][start:end]

                inputdict = tokenizer(original_texts, padding=True, truncation=True, max_length=128, add_special_tokens=False, return_tensors="pt")
                
                true_references = []
                for i in range(len(original_texts)):
                    ref_ids_list = inputdict['input_ids'][i].tolist()
                    if tokenizer.pad_token_id in ref_ids_list:
                        ref_ids_list = ref_ids_list[:ref_ids_list.index(tokenizer.pad_token_id)]
                    true_references.append(tokenizer.decode(ref_ids_list, skip_special_tokens=True))

                # 读取反转结果和指标
                data = torch.load(pt_file, map_location='cpu')
                invert_texts = data.get('invert_text', [])
                metrics = data.get('evaluation_metrics', {})

                # 写入报告内容
                report_lines.append(f"## 实验组: {exp_folder}")
                report_lines.append(f"> **测试范围**: {data_range} | **EM**: {metrics.get('exact_match', 0)*100:.2f}% | **F1**: {metrics.get('token_set_f1', 0)*100:.2f}% | **BLEU**: {metrics.get('bleu_score', 0):.4f}\n")

                # 上下拼装原始文本和逆向文本
                for i in range(len(true_references)):
                    orig = true_references[i]
                    inv = invert_texts[i] if i < len(invert_texts) else "N/A"

                    report_lines.append(f"### 样本 {start + i}")
                    report_lines.append("**[Ground Truth - 原始真实输入 (限制在模型可见的128 Token)]**")
                    report_lines.append("```text")
                    report_lines.append(orig)
                    report_lines.append("```")
                    report_lines.append("**[Inverted Text - 逆向恢复输入]**")
                    report_lines.append("```text")
                    report_lines.append(inv)
                    report_lines.append("```")
                    report_lines.append("---\n")

    # 导出文件
    report_path = os.path.join(results_dir, "text_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ 文本对比报告生成完毕！已保存至: {os.path.abspath(report_path)}")
    print("💡 建议直接在 VS Code 中点击打开该 .md 文件，并点击右上角的【预览 (Open Preview)】按钮以获得极佳的对比阅读体验。")


if __name__ == "__main__":
    main()
    print("="*50)
    main1()