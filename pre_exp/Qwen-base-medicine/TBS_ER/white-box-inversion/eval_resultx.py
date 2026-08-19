import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import AutoTokenizer

# 💡 [修改] 导入我们刚刚为医学数据适配的 attack_utils
import utils

def main_extraction_and_plot():
    # ==========================================
    # 第一阶段：数据提取与统计汇总
    # ==========================================
    print("=== 开始提取并解析实验数据 ===")
    
    # 💡 [修改] 结果目录指向刚才 batch_medical_attack.sh 生成的医学攻击结果目录
    results_dir = "./outputs/attack_results_med"
    if not os.path.exists(results_dir):
        print(f"未找到结果目录 {results_dir}，请检查执行路径是否正确。")
        return

    extracted_data = []
    
    # 💡 [修改] 适配 run_attack.py 和 batch_medical_attack.sh 生成的文件夹命名规范
    # 示例: eval_transfer_19695-19700_Transfer_Qwen2.5-1.5B-Base_l4_tbs_lr0.00005_wdm100_wl10.0_ep200000_cos_cvatan-5_inrandn
    hp_pattern = re.compile(r'Transfer_(.*?)_l(\d+)_([a-zA-Z]+)_lr([0-9\.]+)_wdm([0-9\.]+)_wl1([0-9\.]+)_ep([0-9]+)_([a-zA-Z]+)_cv([a-zA-Z0-9\-]+)_in([a-zA-Z0-9_]+)')

    for exp_folder in os.listdir(results_dir):
        if exp_folder == "batch_logs": continue
            
        exp_path = os.path.join(results_dir, exp_folder)
        if not os.path.isdir(exp_path): continue

        setting_label = "Unknown"
        model_family = "[Unknown]"
        lr = 0.0
        attack_type = "Oracle"
        
        match = hp_pattern.search(exp_folder)
        if match:
            surrogate_model, layer, method, lr_val, w_dm, w_l1, steps, loss_type, changevar, init_type = match.groups()
            lr = float(lr_val)
            
            # 为了图表简洁，简化代理模型的名称
            proxy_short = surrogate_model
            if "Oracle" in proxy_short: proxy_short = "Oracle"
            elif "Base" in proxy_short: proxy_short = "Base"
            elif "Align" in proxy_short: proxy_short = proxy_short.split('_')[1] # 提取 0-1-0-0

            attack_type = "Oracle" if proxy_short == "Oracle" else "Transfer"
            model_family = f"[{proxy_short}]"
            # Setting 标签中体现核心扰动参数
            setting_label = f"{model_family} lr{lr_val}_wdm{w_dm}_ep{steps}"
        else:
            continue

        # 进入内部子文件夹 (如 19695:19700-tbs-...)
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
                        "Med Entity F1": metrics.get("med_entity_f1", 0.0), # 💡 [修改] 提取医学实体 F1
                        "Final Loss": data['L'][-1].item() if 'L' in data and len(data['L']) > 0 else None
                    }
                    extracted_data.append(row)
                except Exception as e:
                    print(f"读取指标文件失败 {pt_file}: {e}")

    if not extracted_data:
        print("未提取到任何有效数据。")
        return

    df = pd.DataFrame(extracted_data)
    
    agg_funcs = {
        'BLEU Score': 'mean',
        'ROUGE-L': 'mean',
        'Token F1': 'mean',
        'Med Entity F1': 'mean', # 💡 [修改] 参与聚合
        'Final Loss': 'mean'
    }
    df_aggregated = df.groupby(['Model Family', 'Attack Class', 'Setting (Layer-Method-Reg)', 'LR_str'], as_index=False).agg(agg_funcs)
    df_aggregated['LR'] = df_aggregated['LR_str'].astype(float)

    df_aggregated = df_aggregated.sort_values(by=["Model Family", "LR", "Setting (Layer-Method-Reg)"]).reset_index(drop=True)
    metric_cols = ['BLEU Score', 'ROUGE-L', 'Token F1', 'Med Entity F1', 'Final Loss']
    df_rounded = df_aggregated.copy()
    df_rounded[metric_cols] = df_aggregated[metric_cols].round(4)
    
    # 💡 [修改] 加入 Medical Entity F1 的显示顺序
    output_column_order = [
        "Model Family",
        "Setting (Layer-Method-Reg)",
        "LR",
        "BLEU Score",
        "ROUGE-L",
        "Token F1",
        "Med Entity F1",
        "Final Loss"
    ]
    df_display = df_rounded[output_column_order].copy()

    print("\n=== MTIA 医学攻击结果汇总 (Markdown) ===\n")
    try:
        print(df_display.to_markdown(index=False))
    except ImportError:
        print(df_display)

    csv_path = os.path.join(results_dir, "mtia_search_summary.csv")
    df_display.to_csv(csv_path, index=False)
    print(f"\n✅ 数据已提取并保存至: {os.path.abspath(csv_path)}")

    # ====================================================================
    # 第二阶段：同族色系渐变映射 (可视化拓展为 4 栏)
    # ====================================================================
    print("\n=== 开始构建同族色系映射与渲染可视化图表 ===")
    
    base_color_pool = ['#1f77b4', '#ff7f0e', '#9467bd', '#d62728', '#8c564b', '#e377c2', '#17becf']
    special_family_hues = {'[Oracle]': '#2ca02c', '[Base]': '#7f7f7f'}

    unique_families = list(df_display['Model Family'].unique())
    palette_dict = {}
    hatch_dict = {}
    hatch_pool = ['', '///', '\\\\\\', 'xxx', '...', '+++', 'ooo', '***']

    color_idx = 0
    for family in unique_families:
        family_settings = list(df_display[df_display['Model Family'] == family]['Setting (Layer-Method-Reg)'].unique())
        n_sub = len(family_settings)

        if family in special_family_hues:
            base_hue = special_family_hues[family]
        else:
            base_hue = base_color_pool[color_idx % len(base_color_pool)]
            color_idx += 1

        if n_sub == 1: shades = [base_hue]
        else:
            cmap = sns.light_palette(base_hue, n_colors=n_sub + 2, input='hex')
            shades = [cmap[i] for i in range(2, n_sub + 2)]

        for idx, setting in enumerate(family_settings):
            palette_dict[setting] = shades[idx]
            hatch_dict[setting] = hatch_pool[idx % len(hatch_pool)]

    # 💡 [修改] 追加 Med Entity F1
    target_metrics = ['BLEU Score', 'ROUGE-L', 'Token F1', 'Med Entity F1']
    df_melt = pd.melt(
        df_display, 
        id_vars=['Model Family', 'Setting (Layer-Method-Reg)', 'LR'],
        value_vars=target_metrics,
        var_name='Metric', 
        value_name='Value'
    )

    metric_map = {'BLEU Score': 'BLEU', 'ROUGE-L': 'ROUGE-L', 'Token F1': 'F1', 'Med Entity F1': 'Med_F1'}
    df_melt['Metric'] = df_melt['Metric'].map(metric_map)
    # 将 F1 指标拉伸到百分制
    df_melt.loc[df_melt['Metric'].isin(['F1', 'Med_F1']), 'Value'] *= 100

    sns.set_theme(style="whitegrid", font="sans-serif")
    all_ordered_settings = list(df_display['Setting (Layer-Method-Reg)'].unique())

    g = sns.catplot(
        x='LR', 
        y='Value', 
        data=df_melt, 
        col='Metric', 
        hue='Setting (Layer-Method-Reg)',
        hue_order=all_ordered_settings,  
        palette=palette_dict,            
        kind='bar', 
        sharey=False, 
        height=4.6, 
        aspect=1.2, # 微调长宽比适应 4 栏
        legend_out=True
    )
    
    g.set_axis_labels("Learning Rate (LR)", "Metric Score")
    
    sub_titles = ['BLEU Score', 'ROUGE-L Score', 'Token F1 (%)', 'Medical Entity F1 (%)']
    for ax, title in zip(g.axes.flat, sub_titles):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=10)

    for ax in g.axes.flat:
        for container, setting in zip(ax.containers, all_ordered_settings):
            hatch_pat = hatch_dict.get(setting, '')
            for bar in container:
                bar.set_hatch(hatch_pat)
                bar.set_edgecolor('#222222')
                bar.set_linewidth(0.5)

    if g._legend:
        g._legend.set_title("Experimental Setting (Grouped by Model)")
        plt.setp(g._legend.get_title(), fontsize=10, fontweight='bold')
        plt.setp(g._legend.get_texts(), fontsize=8)

    plt.subplots_adjust(wspace=0.22)
    
    plot_path = os.path.join(results_dir, "mtia_search_comparison_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 绘图完成！同族分色渐变 4 栏图表已保存为: {os.path.abspath(plot_path)}")


def main_text_report():
    # ==========================================
    # 第三阶段：文本输入与逆向输入视觉对比报告
    # ==========================================
    print("\n=== 开始生成输入与逆向输入视觉对比报告 ===")
    results_dir = "./outputs/attack_results_med"
    if not os.path.exists(results_dir): return

    dataset_cache = {}
    tokenizer_cache = {}
    report_lines = ["# 医学黑盒反演攻击 (Medical MTIA) 视觉对比报告\n\n"]

    for exp_folder in sorted(os.listdir(results_dir)):
        if exp_folder == "batch_logs": continue
            
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
                if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

                if dataset_name not in dataset_cache:
                    raw_texts = utils.get_list_invert_text(dataset_name)
                    # 💡 安全处理：避免数据集越界错误
                    sorted_texts = raw_texts
                    dataset_cache[dataset_name] = sorted_texts

                try:
                    start, end = map(int, data_range.split(':'))
                    original_texts = dataset_cache[dataset_name][start:min(end, len(dataset_cache[dataset_name]))]
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
                # 💡 [修改] 在报告说明中加入 Med Entity F1
                report_lines.append(f"> **范围**: {data_range} | **Med Entity F1**: {metrics.get('med_entity_f1', 0)*100:.2f}% | **Token F1**: {metrics.get('token_set_f1', 0)*100:.2f}% | **BLEU**: {metrics.get('bleu_score', 0):.4f}\n")

                for i in range(len(true_references)):
                    # 💡 [修改] 在文本视觉展示时彻底清除医学模板带来的 <|endoftext|> 脏字符
                    orig = true_references[i].replace("<|endoftext|>", "").strip()
                    inv = invert_texts[i].replace("<|endoftext|>", "").strip() if i < len(invert_texts) else "N/A"

                    report_lines.append(f"### 样本 {start + i}")
                    report_lines.append("**[Ground Truth - 原始患者提问与诊断]**\n```text\n" + orig + "\n```")
                    report_lines.append("**[Inverted Text - 黑盒反演重构输入]**\n```text\n" + inv + "\n```\n---\n")

    report_path = os.path.join(results_dir, "text_search_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ 文本对比报告生成完毕！已保存至: {os.path.abspath(report_path)}")


if __name__ == "__main__":
    main_extraction_and_plot()
    print("="*50)
    main_text_report()