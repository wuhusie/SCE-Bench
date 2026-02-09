
import json
import re
from pathlib import Path

# Mapping of common Chinese terms to academic English terms
TRANSLATIONS = {
    r"读取所有 json": "Load all JSON files",
    r"缩写规则": "Model name abbreviation rules",
    r"模型名缩写": "Model name abbreviation",
    r"拆分": "Split",
    r"合并": "Merge",
    r"指标方向与归一化": "Metric orientation and normalization",
    r"对数放大区分度": "Log transformation to enhance distinguishability",
    r"方向统一": "Uniform direction",
    r"越大越好": "Higher is better",
    r"越小越好": "Lower is better",
    r"颜色映射": "Color mapping",
    r"雷达图函数": "Radar plot function",
    r"雷达图": "Radar plot",
    r"棒棒糖图": "Lollipop chart",
    r"均值引导线": "Mean reference line",
    r"超过阈值标红": "Highlighting targets exceeding benchmarks in red",
    r"参考模型值": "Benchmark model values",
    r"0 参考线": "Zero reference line",
    r"保存到": "Save to",
    r"期望路径": "Expected path",
    r"选择需要的列": "Select required columns",
    r"每个 domain 画一个雷达图": "Plot a radar chart for each domain",
    r"拼接 N1 \+ N50 的 5 个指标": "Combine 5 metrics from N1 and N50",
    r"超过 \(GPT5m/GEM3f\) 任意一个时标红": "Highlight in red if exceeding any GPT5m/GEM3f benchmark",
    r"超过 \(GPT5m/GEM3f\) 任意一个时标蓝": "Highlight in blue if exceeding any GPT5m/GEM3f benchmark",
    r"在 rmse 子图里临时打印": "Temporary print in RMSE subplot",
    r"函数定义": "Function Definitions",
    r"指标方向": "Metric orientation",
    r"归一化": "Normalization",
    r"放大区分度": "Enhance distinguishability",
    r"确保非负": "Ensure non-negative",
    r"同一类模型颜色相同": "Assign same color to models of the same class",
    r"区分度更大": "Improve distinguishability",
    r"直接按数量采样": "Sample directly by quantity",
    r"在 \[0,1\] 上均匀取点": "Uniform sampling in [0,1]",
    r"避免集中在某一段颜色": "Avoid color concentration",
    r"用 tab20 并拉开颜色间隔": "Use tab20 with widened color intervals",
    r"每个 model 单独画": "Plot each model individually",
    r"便于控制线型": "To facilitate linestyle control",
    r"仅 QWE3-nt / QWE3-t": "Only QWE3-nt / QWE3-t",
    r"参考线": "Reference line",
    r"绘制": "Plot",
    r"数据": "Data",
    r"加载": "Load",
    r"结果": "Result",
    r"画一个": "Plot one",
    r"实验": "Experiment",
    r"任务": "Task",
}

def translate_text(text):
    for cn, en in TRANSLATIONS.items():
        text = re.sub(cn, en, text)
    # Remove any remaining Chinese characters (optional, but good for cleaning)
    # text = re.sub(r'[\u4e00-\u9fa5]+', '[TRANSLATE ME]', text)
    return text

def process_notebook(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changed = False
    for cell in nb.get('cells', []):
        source = cell.get('source', [])
        new_source = []
        for line in source:
            new_line = translate_text(line)
            if new_line != line:
                changed = True
            new_source.append(new_line)
        cell['source'] = new_source
        
    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  ✓ Updated {file_path}")
    else:
        print(f"  - No changes in {file_path}")

if __name__ == "__main__":
    base_dir = Path(r"P:\Workspace\01_Research\SCE-Bench-Multi-scale-Behavioral-Alignment-of-LLM-Agents-in-Real-world-Microeconomics\visualization\plots_scripts")
    for nb_file in base_dir.glob("*.ipynb"):
        process_notebook(nb_file)
