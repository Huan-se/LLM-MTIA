import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import AutoTokenizer
import utils


def main_extraction_and_plot():
    # ==========================================
    # 第一阶段：数据提取与统计汇总
    # ==========================================
    print("=== 开始提取并解析实验数据 ===")
    
    results_dir = "../resultsx"
    if not os.path.exists(results_dir):
        results_dir = "./resultsx"
        if not os.path.exists(results_dir):
            print("未找到结果目录，请检查执行路径是否正确。")
            return

    extracted_data = []
    hp_pattern = re.compile(r'_l(\d+)_([a-zA-Z]+)_lr([0-9\.]+)_wdm([0-9\.]+)(?:_wl1([0-9\.]+))?_ep([0-9]+)_([a-zA-Z]+)(?:_cv([a-zA-Z0-9\-]+))?_in([a-zA-Z0-9_]+)$')

    for exp_folder in os.listdir(results_dir):
        if exp_folder == "batch_logs":
            continue
            
        exp_path = os.path.join(results_dir, exp_folder)
        if not os.path.isdir(exp_path): continue

        setting_label = "Unknown"
        model_family = "[Unknown]"
        lr = 0.0
        attack_type = "Oracle"
        
        match = hp_pattern.search(exp_folder)
        if match:
            layer = match.group(1)
            method = match.group(2).upper()
            lr_val = match.group(3)
            w_dm = match.group(4)
            w_l1 = match.group(5) if match.group(5) else "0.0"
            steps = match.group(6)
            loss_type = match.group(7)
            changevar = match.group(8) if match.group(8) else "atan-5"
            init_type = match.group(9)
            
            lr = float(lr_val)
            
            # 提取所属 Model Family (大类) 与完整 Setting 标签
            if "Transfer_" in exp_folder or "eval_transfer_" in exp_folder:
                attack_type = "Transfer"
                surr_match = re.search(r'Transfer_(.*?)_l\d+_', exp_folder)
                proxy_name = surr_match.group(1) if surr_match else "Proxy"
                if "model" in proxy_name:
                    proxy_short = "model"
                else:
                    proxy_short = "_".join(proxy_name.split('_')[1:3]) if '_' in proxy_name else proxy_name[:10]
                model_family = f"[{attack_type}-{proxy_short}]"
                setting_label = f"{model_family} lr{lr_val}_wdm{w_dm}_ep{steps}_{init_type}"
            elif "Base" in exp_folder or "base" in exp_folder:
                attack_type = "Base-prefix"
                model_family = "[Base-prefix]"
                setting_label = f"{model_family} lr{lr_val}_wdm{w_dm}_ep{steps}_{init_type}"
            else:
                attack_type = "Oracle"
                model_family = "[Oracle]"
                setting_label = f"{model_family} lr{lr_val}_wdm{w_dm}_ep{steps}_{init_type}"
                
        elif "_vs_" in exp_folder:
            try:
                parts = exp_folder.split('_vs_')
                remainder = parts[1]
                layer_idx = remainder.find('_layer')
                model_family = "[Legacy]"
                setting_label = f"[Legacy] {remainder[:layer_idx] if layer_idx != -1 else remainder}"
                lr = float(remainder.split('_lr')[-1])
                lr_val = str(lr)
            except:
                continue
        else:
            print(f"跳过无法解析的文件夹: {exp_folder}")
            continue

        for sub_folder in os.listdir(exp_path):
            sub_path = os.path.join(exp_path, sub_folder)
            if not os.path.isdir(sub_path): continue
                
            pt_file = os.path.join(sub_path, "invert-best.pt")
            if os.path.exists(pt_file):
                try:
                    data = torch.load(pt_file, map_location='cpu')
                    metrics = data.get('evaluation_metrics', {})
                    
                    row = {
                        "Model Family": model_family,
                        "Attack Class": attack_type,
                        "Setting (Layer-Method-Reg)": setting_label,
                        "LR": lr,
                        "LR_str": lr_val,
                        "BLEU Score": metrics.get("bleu_score", 0.0),
                        "ROUGE-L": metrics.get("rougeL_score", 0.0),
                        "Token F1": metrics.get("token_set_f1", 0.0),
                        "Final Loss": data['L'][-1].item() if 'L' in data and len(data['L']) > 0 else None
                    }
                    extracted_data.append(row)
                except Exception as e:
                    print(f"读取指标文件失败 {pt_file}: {e}")

    if not extracted_data:
        print("未提取到任何有效数据。")
        return

    df = pd.DataFrame(extracted_data)
    
    # 仅针对同名配置的多次重复测试取均值，保留不同超参配置为独立行
    agg_funcs = {
        'BLEU Score': 'mean',
        'ROUGE-L': 'mean',
        'Token F1': 'mean',
        'Final Loss': 'mean'
    }
    df_aggregated = df.groupby(['Model Family', 'Attack Class', 'Setting (Layer-Method-Reg)', 'LR_str'], as_index=False).agg(agg_funcs)
    df_aggregated['LR'] = df_aggregated['LR_str'].astype(float)

    # 核心：按 Model Family 归类排序，使同大类的项紧密排列
    df_aggregated = df_aggregated.sort_values(by=["Model Family", "LR", "Setting (Layer-Method-Reg)"]).reset_index(drop=True)
    metric_cols = ['BLEU Score', 'ROUGE-L', 'Token F1', 'Final Loss']
    df_rounded = df_aggregated.copy()
    df_rounded[metric_cols] = df_aggregated[metric_cols].round(4)
    
    output_column_order = [
        "Model Family",
        "Setting (Layer-Method-Reg)",
        "LR",
        "BLEU Score",
        "ROUGE-L",
        "Token F1",
        "Final Loss"
    ]
    df_display = df_rounded[output_column_order].copy()

    print("\n=== MTIA 综合攻击结果汇总 (Markdown) ===\n")
    try:
        print(df_display.to_markdown(index=False))
    except ImportError:
        print(df_display)

    csv_path = os.path.join(results_dir, "mtia_search_summary.csv")
    df_display.to_csv(csv_path, index=False)
    print(f"\n✅ 数据已提取并保存至: {os.path.abspath(csv_path)}")

    # ====================================================================
    # 第二阶段：同族色系渐变映射 (Family Color Palette + Texture Generation)
    # ====================================================================
    print("\n=== 开始构建同族色系映射与渲染可视化图表 ===")
    
    # 1. 基础族系主色调库
    base_color_pool = [
        '#1f77b4',  # 蓝色系 (Transfer-1)
        '#ff7f0e',  # 橙色系 (Transfer-2)
        '#9467bd',  # 紫色系 (Transfer-3)
        '#d62728',  # 红色系 (Transfer-4)
        '#8c564b',  # 棕色系
        '#e377c2',  # 粉色系
        '#17becf',  # 青蓝系
    ]
    
    # 特别指定具有基准意义的模型主色
    special_family_hues = {
        '[Oracle]': '#2ca02c',       # 翡翠绿系 (上限参考)
        '[Base-prefix]': '#7f7f7f',  # 经典灰系 (基准参考)
    }

    unique_families = list(df_display['Model Family'].unique())
    palette_dict = {}
    hatch_dict = {}
    hatch_pool = ['', '///', '\\\\\\', 'xxx', '...', '+++', 'ooo', '***']

    color_idx = 0
    for family in unique_families:
        # 获取属于该 Family 的所有独立 Setting 配置
        family_settings = list(df_display[df_display['Model Family'] == family]['Setting (Layer-Method-Reg)'].unique())
        n_sub = len(family_settings)

        # 确定大族的主色
        if family in special_family_hues:
            base_hue = special_family_hues[family]
        else:
            base_hue = base_color_pool[color_idx % len(base_color_pool)]
            color_idx += 1

        # 自动生成同族深浅渐变色阶（避开过浅发白的颜色）
        if n_sub == 1:
            shades = [base_hue]
        else:
            cmap = sns.light_palette(base_hue, n_colors=n_sub + 2, input='hex')
            shades = [cmap[i] for i in range(2, n_sub + 2)]

        # 赋予组内每个设置：对应的渐变色 + 区分性纹理
        for idx, setting in enumerate(family_settings):
            palette_dict[setting] = shades[idx]
            hatch_dict[setting] = hatch_pool[idx % len(hatch_pool)]

    # 2. 数据重塑为长格式 (3栏指标)
    target_metrics = ['BLEU Score', 'ROUGE-L', 'Token F1']
    df_melt = pd.melt(
        df_display, 
        id_vars=['Model Family', 'Setting (Layer-Method-Reg)', 'LR'],
        value_vars=target_metrics,
        var_name='Metric', 
        value_name='Value'
    )

    metric_map = {'BLEU Score': 'BLEU', 'ROUGE-L': 'ROUGE-L', 'Token F1': 'F1'}
    df_melt['Metric'] = df_melt['Metric'].map(metric_map)
    df_melt.loc[df_melt['Metric'] == 'F1', 'Value'] *= 100

    # 3. 开始 Seaborn 画图
    sns.set_theme(style="whitegrid", font="sans-serif")
    all_ordered_settings = list(df_display['Setting (Layer-Method-Reg)'].unique())

    g = sns.catplot(
        x='LR', 
        y='Value', 
        data=df_melt, 
        col='Metric', 
        hue='Setting (Layer-Method-Reg)',
        hue_order=all_ordered_settings,  # 确保同组配置在柱状图中按顺序排列
        palette=palette_dict,            # 应用同族色系映射
        kind='bar', 
        sharey=False, 
        height=4.6, 
        aspect=1.2,
        legend_out=True
    )
    
    g.set_axis_labels("Learning Rate (LR)", "Metric Score")
    
    sub_titles = ['BLEU Score', 'ROUGE-L Score', 'Token F1 (%)']
    for ax, title in zip(g.axes.flat, sub_titles):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=10)

    # 4. 精准为各个容器添加对应的组内纹理（Hatch）
    for ax in g.axes.flat:
        for container, setting in zip(ax.containers, all_ordered_settings):
            hatch_pat = hatch_dict.get(setting, '')
            for bar in container:
                bar.set_hatch(hatch_pat)
                bar.set_edgecolor('#222222')
                bar.set_linewidth(0.5)

    # 5. 美化图例
    if g._legend:
        g._legend.set_title("Experimental Setting (Grouped by Model)")
        plt.setp(g._legend.get_title(), fontsize=10, fontweight='bold')
        plt.setp(g._legend.get_texts(), fontsize=8)

    plt.subplots_adjust(wspace=0.22)
    
    plot_path = os.path.join(results_dir, "mtia_search_comparison_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 绘图完成！同族分色渐变图表已保存为: {os.path.abspath(plot_path)}")


def main_text_report():
    print("\n=== 开始生成输入与逆向输入视觉对比报告 ===")
    results_dir = "../resultsx"
    if not os.path.exists(results_dir):
        results_dir = "./resultsx"
        if not os.path.exists(results_dir): return

    dataset_cache = {}
    tokenizer_cache = {}
    report_lines = ["# MTIA 综合攻击 (Oracle & Transfer) 对比报告\n\n"]

    for exp_folder in sorted(os.listdir(results_dir)):
        if exp_folder == "batch_logs":
            continue
            
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
                surrogate_model = args.get('surrogate_model', target_model)
                data_range = args.get('range')

                if not dataset_name or not target_model: continue

                attack_context = "🎯 单模型攻击 (Oracle)" if target_model == surrogate_model else f"⚔️ 跨模型窃取 (Proxy: {surrogate_model})"

                if target_model not in tokenizer_cache:
                    llm_path = utils.LLM_PATH[target_model]
                    try:
                        tokenizer_cache[target_model] = AutoTokenizer.from_pretrained(llm_path, fix_mistral_regex=True)
                    except TypeError:
                        tokenizer_cache[target_model] = AutoTokenizer.from_pretrained(llm_path)
                        
                tokenizer = tokenizer_cache[target_model]
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                if dataset_name not in dataset_cache:
                    raw_texts = utils.get_list_invert_text(dataset_name)
                    sorted_texts = sorted(raw_texts, key=lambda x: len(tokenizer.encode(x, add_special_tokens=False)), reverse=True)
                    dataset_cache[dataset_name] = sorted_texts

                try:
                    start, end = map(int, data_range.split(':'))
                    original_texts = dataset_cache[dataset_name][start:end]
                except Exception:
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
                report_lines.append(f"**攻击场景**: {attack_context}")
                report_lines.append(f"> **范围**: {data_range} | **F1**: {metrics.get('token_set_f1', 0)*100:.2f}% | **BLEU**: {metrics.get('bleu_score', 0):.4f}\n")

                for i in range(len(true_references)):
                    orig = true_references[i]
                    inv = invert_texts[i] if i < len(invert_texts) else "N/A"

                    report_lines.append(f"### 样本 {start + i}")
                    report_lines.append("**[Ground Truth - 原始真实输入]**\n```text\n" + orig + "\n```")
                    report_lines.append("**[Inverted Text - 逆向恢复输入]**\n```text\n" + inv + "\n```\n---\n")

    report_path = os.path.join(results_dir, "text_search_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ 文本对比报告生成完毕！已保存至: {os.path.abspath(report_path)}")


if __name__ == "__main__":
    main_extraction_and_plot()
    print("="*50)
    main_text_report()