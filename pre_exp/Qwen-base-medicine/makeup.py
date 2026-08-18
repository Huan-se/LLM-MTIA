import os
import re
import glob

def parse_logs_to_md():
    # 💡 [修改] 适配最新的一键搜参综合评估日志路径
    log_dir = "./search_outputs/batch_logs_integrated"
    # 生成的 Markdown 文件保存路径
    output_md_path = "./search_outputs/summary_results_medical.md"
    
    if not os.path.exists(log_dir):
        print(f"❌ 找不到日志文件夹: {log_dir}")
        return

    # 💡 [修改] 精确提取全新医学评估指标的正则表达式
    regex_patterns = {
        # 1. 文本与医学实体生成质量指标
        "BLEU-4": re.compile(r"🔸\s*BLEU-4\s*:\s*([\d.]+)"),
        "ROUGE-1": re.compile(r"🔸\s*ROUGE-1\s*:\s*([\d.]+)"),
        "ROUGE-L": re.compile(r"🔸\s*ROUGE-L\s*:\s*([\d.]+)"),
        "Med_Entity_F1": re.compile(r"🔸\s*临床实体 F1\s*:\s*([\d.]+)"),
        
        # 2. Layer 4 特征映射保真度指标
        "MSE": re.compile(r"🔸\s*MSE\s*\(均方误差\)\s*:\s*([\d.]+)"),
        "Cosine": re.compile(r"🔸\s*Cosine\s*\(余弦相似度\)\s*:\s*([\d.]+)"),
        "CKA": re.compile(r"🔸\s*CKA\s*\(中心化核对齐\)\s*:\s*([\d.]+)"),
        "Sensitivity": re.compile(r"🔹\s*变化量敏感度\s*\(ΔHe/ΔHo\)\s*:\s*([\d.]+)")
    }
    
    # 提取文件名的正则保持不变，解析超参命名
    filename_pattern = re.compile(r"align_a([0-9.]+)_g([0-9.]+)_l([0-9.]+)_b([0-9.]+)_([a-zA-Z0-9]+)\.log")

    results = []
    
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        print("⚠️ 在文件夹中没有找到任何 .log 文件。")
        return

    print(f"📂 发现 {len(log_files)} 个日志文件，开始解析医学评估指标...\n")

    for filepath in log_files:
        filename = os.path.basename(filepath)
        
        # 解析方法名称 (Method)
        match_name = filename_pattern.search(filename)
        if match_name:
            a, g, l, b, mode = match_name.groups()
            # 格式化样式: "0.2 0.1 0 0 splicedbase"
            method_name = f"{float(a):g} {float(g):g} {float(l):g} {float(b):g} {mode}"
        else:
            method_name = filename.replace(".log", "")
            
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 提取各个指标，如果因报错没跑完或不存在则填 "N/A"
        row_data = {"Method": method_name}
        for key, pattern in regex_patterns.items():
            match = pattern.search(content)
            if match:
                row_data[key] = match.group(1)
            else:
                row_data[key] = "N/A"
                
        results.append(row_data)

    # 按照 Method 名称的字母顺序进行排序
    results = sorted(results, key=lambda x: x["Method"])

    # 构建 Markdown 表格
    md_lines = []
    # 💡 [修改] 重新组织表头
    md_lines.append("| Method (a/g/l/b/mode) | BLEU-4 ↑ | ROUGE-1 ↑ | ROUGE-L ↑ | Med Entity F1 ↑ | MSE ↓ | Cosine ↑ | CKA ↑ | 敏感度 (越近1.0越好) |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    
    for r in results:
        line = f"| {r['Method']} | {r['BLEU-4']} | {r['ROUGE-1']} | {r['ROUGE-L']} | {r['Med_Entity_F1']} | {r['MSE']} | {r['Cosine']} | {r['CKA']} | {r['Sensitivity']} |"
        md_lines.append(line)
        
    md_content = "\n".join(md_lines)
    
    # 打印到控制台
    print(md_content)
    print("\n" + "="*80)
    
    # 写入文件
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("# 医学问答对齐模型搜参评估汇总\n\n")
        f.write(md_content)
        f.write("\n\n> *注: `Med Entity F1` 为 scispacy 识别的专业疾病与药物实体召回情况。`敏感度` 越接近 1.0 表示模型对输入扰动的抗性与 Oracle 越相似。*\n")
    
    print(f"✅ 解析完成！Markdown 结果表格已保存至: {os.path.abspath(output_md_path)}")

if __name__ == "__main__":
    parse_logs_to_md()