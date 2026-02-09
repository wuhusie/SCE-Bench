# -*- coding: utf-8 -*-
"""
验证 exp9 记忆模式的数据拼接是否正确

用法:
  python verify_memory.py --experiment spending --limit 20
"""

import argparse
import sys
from pathlib import Path

# 路径设置
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from sce.config import load_experiment_config
from sce.experiments.exp9 import Exp9SpendingExperiment, Exp9CreditExperiment, Exp9LaborExperiment

EXPERIMENT_CLASSES = {
    'spending': Exp9SpendingExperiment,
    'credit': Exp9CreditExperiment,
    'labor': Exp9LaborExperiment,
}


def main():
    parser = argparse.ArgumentParser(description="验证 exp9 记忆拼接")
    parser.add_argument('--experiment', type=str, required=True, choices=['spending', 'credit', 'labor'])
    parser.add_argument('--limit', type=int, default=20, help="显示的行数")
    parser.add_argument('--user', type=str, default=None, help="指定查看某个 userid")
    args = parser.parse_args()

    # 加载配置和实验类
    task = args.experiment
    config = load_experiment_config(task)
    ExperimentClass = EXPERIMENT_CLASSES[task]
    experiment = ExperimentClass(config, config, f'exp9_{task}')

    # 加载数据 (会自动计算 previousGt)
    print(f"\n{'='*60}")
    print(f"加载 {task} 数据并计算 previousGt...")
    print(f"{'='*60}\n")

    df = experiment.load_data()

    # 准备 prompts
    print(f"\n生成 prompts...")
    df = experiment.prepare_prompts(df)

    # 获取 GT 列名
    gt_col = ExperimentClass.GT_COLUMN
    print(f"\nGT 列: {gt_col}")
    print(f"总记录数: {len(df)}")
    print(f"有记忆 (previousGt 非空): {df['previousGt'].notna().sum()}")
    print(f"无记忆 (previousGt 为空): {df['previousGt'].isna().sum()}")

    # 如果指定了 userid，只看这个用户
    if args.user:
        df_show = df[df['userid'] == args.user].copy()
        if len(df_show) == 0:
            print(f"\n未找到 userid={args.user}")
            return
        print(f"\n查看用户 {args.user} 的所有记录:")
    else:
        # 找一个有多次记录的用户来展示
        user_counts = df['userid'].value_counts()
        multi_record_users = user_counts[user_counts >= 3].index.tolist()

        if multi_record_users:
            sample_user = multi_record_users[0]
            df_show = df[df['userid'] == sample_user].copy()
            print(f"\n示例用户 {sample_user} (共 {len(df_show)} 条记录):")
        else:
            df_show = df.head(args.limit)
            print(f"\n前 {args.limit} 条记录:")

    # 按日期排序
    df_show = df_show.sort_values('date')

    # 显示关键列
    print(f"\n{'='*80}")
    print(f"{'date':<10} {'userid':<15} {gt_col:<15} {'previousGt':<15} {'has_memory'}")
    print(f"{'='*80}")

    for _, row in df_show.iterrows():
        gt_val = row.get(gt_col, 'N/A')
        prev_gt = row.get('previousGt', None)
        has_mem = 'Yes' if prev_gt is not None and str(prev_gt) != 'nan' else 'No'
        prev_gt_str = str(prev_gt) if has_mem == 'Yes' else '-'

        print(f"{row['date']:<10} {str(row['userid']):<15} {str(gt_val):<15} {prev_gt_str:<15} {has_mem}")

    # 验证：检查 previousGt 是否等于上一行的 gt_col
    print(f"\n{'='*60}")
    print("验证 previousGt 计算正确性...")
    print(f"{'='*60}")

    errors = 0
    df_sorted = df.sort_values(['userid', 'date'])

    for userid, group in df_sorted.groupby('userid'):
        group = group.sort_values('date')
        prev_val = None
        for idx, row in group.iterrows():
            expected = prev_val
            actual = row.get('previousGt')

            # 比较 (处理 NaN)
            if expected is None:
                if actual is not None and str(actual) != 'nan':
                    errors += 1
                    print(f"  错误: userid={userid}, date={row['date']}, 期望=None, 实际={actual}")
            else:
                if str(actual) != str(expected):
                    errors += 1
                    print(f"  错误: userid={userid}, date={row['date']}, 期望={expected}, 实际={actual}")

            prev_val = row.get(gt_col)

    if errors == 0:
        print("✅ previousGt 计算正确!")
    else:
        print(f"❌ 发现 {errors} 处错误")

    # 显示一个完整的 prompt 示例
    print(f"\n{'='*60}")
    print("完整 Prompt 示例 (有记忆的记录):")
    print(f"{'='*60}")

    # 找一条有记忆的记录
    df_with_memory = df[df['previousGt'].notna()]
    if len(df_with_memory) > 0:
        sample_row = df_with_memory.iloc[0]
        system_prompt = experiment.get_system_prompt()
        final_system, final_user, extra = experiment.build_row_prompts(sample_row, system_prompt)

        print(f"\n--- System Prompt ---\n{final_system[:500]}...")
        print(f"\n--- User Prompt ---\n{final_user}")
        print(f"\n--- Extra Fields ---\n{extra}")
    else:
        print("没有找到有记忆的记录")


if __name__ == "__main__":
    main()
