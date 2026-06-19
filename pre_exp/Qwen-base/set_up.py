import os
import re
import glob

def parse_logs_to_md():
    # 日志文件夹路径
    log_dir = "./search_outputs/batch_logs"
    # 生成的 Markdown 文件保存路径
    output_md_path = "./search_outputs/summary_results.md"
    
    if not os.path.exists(log_dir):
        print(f"❌ 找不到日志文件夹: {log_dir}")
        return

    # 定义正则表达式来精确提取各项指标
    regex_patterns = {
        "CodeBLEU": re.compile(r"===\s*总分\s*\(CodeBLEU\):\s*([\d.]+)\s*==="),
        "N-gram": re.compile(r"-\s*N-gram 匹配:\s*([\d.]+)"),
        "Keyword": re.compile(r"-\s*关键词匹配\s*:\s*([\d.]+)"),
        "Syntax": re.compile(r"-\s*语法树匹配\s*:\s*([\d.]+)"),
        "Dataflow": re.compile(r"-\s*数据流匹配\s*:\s*([\d.]+)"),
        "MSE": re.compile(r"🔸\s*MSE\s*\(均方误差\)\s*:\s*([\d.]+)"),
        "Cosine": re.compile(r"🔸\s*Cosine\s*\(余弦相似度\)\s*:\s*([\d.]+)")
    }
    
    # 提取文件名的正则，例如: align_a0.2_g0.1_l0.0_b0.0_phase2.log
    filename_pattern = re.compile(r"align_a([0-9.]+)_g([0-9.]+)_l([0-9.]+)_b([0-9.]+)_([a-zA-Z0-9]+)\.log")

    results = []
    
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        print("⚠️ 在文件夹中没有找到任何 .log 文件。")
        return

    print(f"📂 发现 {len(log_files)} 个日志文件，开始解析...\n")

    for filepath in log_files:
        filename = os.path.basename(filepath)
        
        # 解析方法名称 (Method)
        match_name = filename_pattern.search(filename)
        if match_name:
            a, g, l, b, mode = match_name.groups()
            # 格式化为你需要的样式，例如 "0.2 0.1 0 0 base"
            method_name = f"{float(a):g} {float(g):g} {float(l):g} {float(b):g} {mode}"
        else:
            method_name = filename.replace(".log", "")
            
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 提取各个指标，如果因报错没跑完则填 "N/A"
        row_data = {"Method": method_name}
        for key, pattern in regex_patterns.items():
            match = pattern.search(content)
            if match:
                row_data[key] = match.group(1)
            else:
                row_data[key] = "N/A"
                
        results.append(row_data)

    # 按照 Method 名称的字母顺序进行排序，方便查看
    results = sorted(results, key=lambda x: x["Method"])

    # 构建 Markdown 表格
    md_lines = []
    md_lines.append("| Method | CodeBLEU ↑ | N-gram ↑ | Keyword ↑ | Syntax ↑ | Dataflow ↑ | MSE ↓ | Cosine ↑ |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    
    for r in results:
        line = f"| {r['Method']} | {r['CodeBLEU']} | {r['N-gram']} | {r['Keyword']} | {r['Syntax']} | {r['Dataflow']} | {r['MSE']} | {r['Cosine']} |"
        md_lines.append(line)
        
    md_content = "\n".join(md_lines)
    
    # 打印到控制台
    print(md_content)
    print("\n" + "="*50)
    
    # 写入文件
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"✅ 解析完成！Markdown 表格已保存至: {output_md_path}")

if __name__ == "__main__":
    parse_logs_to_md()