"""
数据合并模块。

将 LLM 预测结果与人类真实回答 (Ground Truth) 进行合并。

复用自:
- src/analysis/exp1-1/append_ground_truth.py
- src/analysis/expN50/append_ground_truth.py
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


# Ground Truth 配置
TASK_GT_CONFIG = {
    "spending": {
        "gt_file": "sceProfile.pkl",
        "gt_col": "Q26v2part2"
    },
    "labor": {
        "gt_file": "labor_original.pkl",
        "gt_col": "oo2c3"
    },
    "credit": {
        "gt_file": "credit_original.pkl",
        "gt_col": "N17b_2"
    }
}


def merge_ground_truth(
    result_file: Path,
    cache_dir: Path,
    output_dir: Path,
    task_name: str
) -> Optional[Path]:
    """
    将单个 LLM 结果文件与 Ground Truth 合并。

    逻辑复用自 exp1-1/append_ground_truth.py

    参数:
        result_file: LLM 结果 CSV 文件路径
        cache_dir: Ground Truth PKL 缓存目录
        output_dir: 输出目录
        task_name: 任务名称 (spending/labor/credit)

    返回:
        合并后的文件路径，失败返回 None
    """
    if task_name not in TASK_GT_CONFIG:
        print(f"  ✗ 未知任务: {task_name}")
        return None

    config = TASK_GT_CONFIG[task_name]
    gt_file = config["gt_file"]
    gt_col = config["gt_col"]

    # 1. 读取 LLM 结果
    if not result_file.exists():
        print(f"  ✗ 结果文件不存在: {result_file}")
        return None

    print(f"  读取结果: {result_file.name}")
    try:
        df_res = pd.read_csv(result_file)
    except Exception as e:
        print(f"  ✗ 读取CSV失败: {e}")
        return None

    # 2. 读取 Ground Truth (PKL)
    gt_path = cache_dir / gt_file
    if not gt_path.exists():
        print(f"  ✗ GT文件不存在: {gt_path}")
        return None

    print(f"  读取GT: {gt_file}")
    try:
        df_gt = pd.read_pickle(gt_path)
    except Exception as e:
        print(f"  ✗ 读取PKL失败: {e}")
        return None

    # 3. 验证 GT 列存在
    if gt_col not in df_gt.columns:
        print(f"  ✗ 列 '{gt_col}' 不存在于 {gt_file}")
        print(f"    可用列: {list(df_gt.columns)[:10]}...")
        return None

    # 4. 类型转换 (确保 join key 一致)
    # date: 转为 int
    if 'date' in df_res.columns and 'date' in df_gt.columns:
        df_res['date'] = pd.to_numeric(df_res['date'], errors='coerce').fillna(0).astype(int)
        df_gt['date'] = pd.to_numeric(df_gt['date'], errors='coerce').fillna(0).astype(int)

    # userid: 转为 str
    df_res['userid'] = df_res['userid'].astype(str)
    df_gt['userid'] = df_gt['userid'].astype(str)

    # 5. 提取 GT 子集 (避免列冲突)
    gt_subset = df_gt[['userid', 'date', gt_col]].copy()

    # 6. Left Join 合并
    print(f"  合并中 (userid + date)...")
    merged_df = pd.merge(df_res, gt_subset, on=['userid', 'date'], how='left')

    # 7. 统计覆盖率
    filled_rate = merged_df[gt_col].notna().mean()
    print(f"  GT覆盖率: {filled_rate:.2%}")

    # 8. 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = result_file.stem + "_withHumanData.csv"
    output_path = output_dir / output_filename

    merged_df.to_csv(output_path, index=False)
    print(f"  ✓ 保存: {output_path}")

    return output_path


def merge_task_files(
    result_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    task_name: str,
    file_pattern: str = None
) -> List[Path]:
    """
    合并某个任务的所有结果文件。

    支持递归搜索子目录 (用于 N50 实验的模型子目录结构)。

    参数:
        result_dir: 结果目录
        cache_dir: GT 缓存目录
        output_dir: 输出目录
        task_name: 任务名称
        file_pattern: 文件匹配模式，默认 "{task_name}_*.csv"

    返回:
        成功合并的文件路径列表
    """
    if file_pattern is None:
        file_pattern = f"{task_name}_*.csv"

    # 递归搜索
    result_files = list(result_dir.rglob(file_pattern))

    # 排除已合并的文件
    result_files = [f for f in result_files if "_withHumanData" not in f.name]

    if not result_files:
        print(f"[{task_name.upper()}] 未找到匹配文件: {file_pattern}")
        return []

    print(f"[{task_name.upper()}] 找到 {len(result_files)} 个文件")

    merged_files = []
    for result_file in result_files:
        # 保持目录结构
        try:
            rel_path = result_file.relative_to(result_dir)
            file_output_dir = output_dir / rel_path.parent
        except ValueError:
            file_output_dir = output_dir

        output_path = merge_ground_truth(
            result_file=result_file,
            cache_dir=cache_dir,
            output_dir=file_output_dir,
            task_name=task_name
        )

        if output_path:
            merged_files.append(output_path)

    return merged_files


def merge_all_tasks(
    result_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    tasks: List[str] = None
) -> Dict[str, List[Path]]:
    """
    批量合并所有任务。

    参数:
        result_dir: 结果目录
        cache_dir: GT 缓存目录
        output_dir: 输出目录
        tasks: 任务列表，默认 ["spending", "labor", "credit"]

    返回:
        {task_name: [output_paths]} 字典
    """
    tasks = tasks or ["spending", "labor", "credit"]

    print("=" * 60)
    print("合并 Ground Truth")
    print("=" * 60)
    print(f"结果目录: {result_dir}")
    print(f"GT目录: {cache_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    results = {}
    for task_name in tasks:
        merged_files = merge_task_files(
            result_dir=result_dir,
            cache_dir=cache_dir,
            output_dir=output_dir,
            task_name=task_name
        )
        results[task_name] = merged_files

    return results
