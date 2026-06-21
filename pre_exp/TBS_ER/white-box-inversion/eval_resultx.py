import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import utils

def main_extraction_and_plot():
    # ==========================================
    # 第一阶段：数据提取与统计汇总 (适配全新超参数搜索命名)
    # ==========================================
    print("=== 开始提取并解析实验数据 ===")
    
    results_dir = "../resultsx"
    if not os.path.exists(results_dir):
        results_dir = "./resultsx"
        if not os.path.exists(results_dir):
            print("未找到结果目录，请检查执行路径是否正确。")
            return

    extracted_data = []

    for exp_folder in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_folder)
        if not os.path.isdir(exp_path): continue

        setting_label = "Unknown"
        lr = 0.0
        
        if exp_folder.startswith("search_Oracle_"):
            # 解析格式: search_Oracle_l4_tbs_lr0.01_wdm20_wl10.001_ep5000_mse
            try:
                params_str = exp_folder.replace("search_Oracle_", "")
                parts = params_str.split('_')
                layer = parts[0].replace('l', '')
                method = parts[1].upper()
                lr_val = parts[2].replace('lr', '')
                w_dm = parts[3].replace('wdm', '')
                w_l1 = parts[4].replace('wl1', '')
                steps = parts[5].replace('ep', '')
                loss_type = parts[6] if len(parts) > 6 else "mse"
                # 处理可能不存在的旧文件夹以防报错
                changevar = parts[7].replace('cv', '') if len(parts) > 7 else "atan-5"
                # 注意：xav_uni 中间本身有一个下划线，如果是拆分，在 parts 里占据两格
                if len(parts) > 9:
                    init_type = parts[8].replace('in', '') + "_" + parts[9]
                else:
                    init_type = parts[8].replace('in', '') if len(parts) > 8 else "ones"
                
                lr = float(lr_val)
                # 全量拼接标签
                setting_label = f"lr{lr_val}_L{layer}_{method}_wdm{w_dm}_wl1{w_l1}_ep{steps}_{loss_type}_cv{changevar}_in{init_type}"
            except Exception as e:
                print(f"跳过无法解析的搜参文件夹: {exp_folder}")
                continue
        # ---------------------------
        elif "_vs_" in exp_folder:
            # 兼容老格式
            try:
                parts = exp_folder.split('_vs_')
                remainder = parts[1]
                layer_idx = remainder.find('_layer')
                setting_label = remainder[:layer_idx] if layer_idx != -1 else remainder
                lr = float(remainder.split('_lr')[-1])
            except:
                continue
        else:
            continue

        # 进入子目录寻找指标文件
        for sub_folder in os.listdir(exp_path):
            sub_path = os.path.join(exp_path, sub_folder)
            if not os.path.isdir(sub_path): continue
                
            pt_file = os.path.join(sub_path, "invert-best.pt")
            if os.path.exists(pt_file):
                try:
                    data = torch.load(pt_file, map_location='cpu')
                    metrics = data.get('evaluation_metrics', {})
                    
                    row = {
                        "Setting (Layer-Method-Reg)": setting_label,
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
        print("未提取到任何有效数据，请确保至少有一个任务成功完成了所有迭代。")
        return

    # 转化为 DataFrame 并格式化
    df = pd.DataFrame(extracted_data)
    df = df.sort_values(by=["Setting (Layer-Method-Reg)", "LR"]).reset_index(drop=True)
    df_rounded = df.round(4)

    print("\n=== MTIA 攻击超参数搜索结果汇总 (Markdown) ===\n")
    try:
        print(df_rounded.to_markdown(index=False))
    except ImportError:
        print("[警告] 环境中未安装 tabulate，使用原生 Pandas 格式打印：")
        print(df_rounded)

    csv_path = os.path.join(results_dir, "mtia_search_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 数据已提取并保存至: {os.path.abspath(csv_path)}")

    # ==========================================
    # 第二阶段：数据重塑与 Seaborn 搜参可视化
    # ==========================================
    print("\n=== 开始渲染可视化图表 ===")
    
    df_melt = pd.melt(
        df, 
        id_vars=['Setting (Layer-Method-Reg)', 'LR'],
        value_vars=['BLEU Score', 'ROUGE-L', 'Exact Match', 'Token F1'],
        var_name='Metric', 
        value_name='Value'
    )

    metric_map = {'BLEU Score': 'BLEU', 'ROUGE-L': 'Rouge-L', 'Exact Match': 'EM', 'Token F1': 'F1'}
    df_melt['Metric'] = df_melt['Metric'].map(metric_map)

    df_melt.loc[df_melt['Metric'] == 'EM', 'Value'] *= 100
    df_melt.loc[df_melt['Metric'] == 'F1', 'Value'] *= 100

    sns.set_theme(style="darkgrid")
    
    g = sns.catplot(
        x='LR', 
        y='Value', 
        data=df_melt, 
        col='Metric', 
        hue='Setting (Layer-Method-Reg)', # 使用解析出的超参数组合进行图例区分
        kind='bar', 
        sharey=False, 
        height=4.0, 
        aspect=1.2
    )
    
    g.set_axis_labels("Learning Rate", "Metric Value")
    
    titles = ['BLEU Score', 'ROUGE-L Score', 'Exact Match (%)', 'Token F1 (%)']
    for ax, title in zip(g.axes[0], titles):
        ax.set_title(title)

    hatches = ['o', 'x', '\\\\', '*', '+', '-'] 
    for i, ax in enumerate(g.axes[0]):
        num_lrs = max(len(df['LR'].unique()), 1)
        for j, thisbar in enumerate(ax.patches):
            num_settings = len(df['Setting (Layer-Method-Reg)'].unique())
            if num_settings > 0:
                hatch_idx = (j // num_lrs) % len(hatches)
                thisbar.set_hatch(hatches[hatch_idx])

    for ax in g.axes[0]:
        plt.setp(ax.get_yticklabels(), fontsize=10)
        ax.tick_params(pad=-4, rotation=0)

    plt.subplots_adjust(hspace=0., wspace=0.25)
    
    plot_path = os.path.join(results_dir, "mtia_search_comparison_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 绘图完成！科研对比图表已保存为: {os.path.abspath(plot_path)}")


def main_text_report():
    print("\n=== 开始生成输入与逆向输入视觉对比报告 ===")
    
    results_dir = "../resultsx"
    if not os.path.exists(results_dir):
        results_dir = "./resultsx"
        if not os.path.exists(results_dir): return

    dataset_cache = {}
    tokenizer_cache = {}
    report_lines = ["# MTIA 输入反转参数搜索对比报告\n\n"]

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

                # --- 核心更新：修复 Mistral Tokenizer 兼容性 ---
                if target_model not in tokenizer_cache:
                    print(f"正在加载 Tokenizer: {target_model} ...")
                    llm_path = utils.LLM_PATH[target_model]
                    try:
                        tokenizer_cache[target_model] = AutoTokenizer.from_pretrained(llm_path, fix_mistral_regex=True)
                    except TypeError:
                        tokenizer_cache[target_model] = AutoTokenizer.from_pretrained(llm_path)
                        
                tokenizer = tokenizer_cache[target_model]
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # 数据集读取 (utils 已经更新，自动兼容 Parquet 目录)
                if dataset_name not in dataset_cache:
                    print(f"正在解析原始数据集: {dataset_name} ...")
                    raw_texts = utils.get_list_invert_text(dataset_name)
                    sorted_texts = sorted(raw_texts, key=lambda x: len(tokenizer.encode(x, add_special_tokens=False)), reverse=True)
                    dataset_cache[dataset_name] = sorted_texts

                try:
                    start, end = map(int, data_range.split(':'))
                    original_texts = dataset_cache[dataset_name][start:end]
                except Exception as e:
                    print(f"范围解析失败或数据量不足: {e}")
                    continue

                inputdict = tokenizer(original_texts, padding=True, truncation=True, max_length=128, add_special_tokens=False, return_tensors="pt")
                
                true_references = []
                for i in range(len(original_texts)):
                    ref_ids_list = inputdict['input_ids'][i].tolist()
                    if tokenizer.pad_token_id in ref_ids_list:
                        ref_ids_list = ref_ids_list[:ref_ids_list.index(tokenizer.pad_token_id)]
                    true_references.append(tokenizer.decode(ref_ids_list, skip_special_tokens=True))

                data = torch.load(pt_file, map_location='cpu')
                invert_texts = data.get('invert_text', [])
                metrics = data.get('evaluation_metrics', {})

                report_lines.append(f"## 实验配置组: `{exp_folder}`")
                report_lines.append(f"> **范围**: {data_range} | **EM**: {metrics.get('exact_match', 0)*100:.2f}% | **F1**: {metrics.get('token_set_f1', 0)*100:.2f}% | **BLEU**: {metrics.get('bleu_score', 0):.4f}\n")

                for i in range(len(true_references)):
                    orig = true_references[i]
                    inv = invert_texts[i] if i < len(invert_texts) else "N/A"

                    report_lines.append(f"### 样本 {start + i}")
                    report_lines.append("**[Ground Truth - 原始真实输入 (截断至128 Token)]**")
                    report_lines.append("```text")
                    report_lines.append(orig)
                    report_lines.append("```")
                    report_lines.append("**[Inverted Text - 逆向恢复输入]**")
                    report_lines.append("```text")
                    report_lines.append(inv)
                    report_lines.append("```")
                    report_lines.append("---\n")

    report_path = os.path.join(results_dir, "text_search_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ 文本对比报告生成完毕！已保存至: {os.path.abspath(report_path)}")
    print("💡 使用 VS Code 的 Markdown 预览功能即可直观对比不同超参数下的反转效果差异。")

if __name__ == "__main__":
    main_extraction_and_plot()
    print("="*50)
    main_text_report()