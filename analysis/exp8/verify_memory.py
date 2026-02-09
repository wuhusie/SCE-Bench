# -*- coding: utf-8 -*-
"""
验证 exp8 模型回答记忆的数据拼接是否正确

用法:
  python verify_memory.py --experiment spending --prior-result /path/to/exp1.1/spending_xxx.csv
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

# 路径设置
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from sce.config import load_experiment_config
from sce.experiments.exp8 import Exp8SpendingExperiment, Exp8CreditExperiment, Exp8LaborExperiment

EXPERIMENT_CLASSES = {
    'spending': Exp8SpendingExperiment,
    'credit': Exp8CreditExperiment,
    'labor': Exp8LaborExperiment,
}


def main():
    parser = argparse.ArgumentParser(description="验证 exp8 模型回答记忆拼接")
    parser.add_argument('--experiment', type=str, required=True, choices=['spending', 'credit', 'labor'])
    parser.add_argument('--prior-result', type=str, required=True, help="第一轮实验结果文件路径")
    parser.add_argument('--limit', type=int, default=20, help="显示的行数")
    parser.add_argument('--user', type=str, default=None, help="指定查看某个 userid")
    args = parser.parse_args()

    # 检查第一轮结果文件
    if not Path(args.prior_result).exists():
        print(f"❌ 错误: 第一轮结果文件不存在: {args.prior_result}")
        return

    # 加载第一轮结果，构建映射
    print(f"\n{'='*60}")
    print(f"加载第一轮实验结果: {args.prior_result}")
    print(f"{'='*60}\n")

    prior_df = pd.read_csv(args.prior_result)
    prior_map = {}
    for _, row in prior_df.iterrows():
        key = (row['userid'], row['date'])
        prior_map[key] = row['llm_response']
    print(f"第一轮结果: {len(prior_map)} 条 (userid, date) → llm_response 映射")

    # 加载配置和实验类
    task = args.experiment
    config = load_experiment_config(task)
    ExperimentClass = EXPERIMENT_CLASSES[task]
    experiment = ExperimentClass(config, config, f'exp8_{task}')

    # 设置第一轮结果路径
    experiment.prior_result_path = args.prior_result

    # 加载数据 (会自动计算 previousLLM)
    print(f"\n加载 {task} 数据并计算 previousLLM...")
    df = experiment.load_data()

    # 准备 prompts
    print(f"生成 prompts...")
    df = experiment.prepare_prompts(df)

    print(f"\n总记录数: {len(df)}")
    print(f"有记忆 (previousLLM 非空): {df['previousLLM'].notna().sum()}")
    print(f"无记忆 (previousLLM 为空): {df['previousLLM'].isna().sum()}")

    # ================================================================
    # 验证 previousLLM 计算正确性
    # ================================================================
    print(f"\n{'='*60}")
    print("验证 previousLLM 计算正确性...")
    print(f"{'='*60}")

    df_sorted = df.sort_values(['userid', 'date'])
    errors = 0
    checked = 0

    for userid, group in df_sorted.groupby('userid'):
        group = group.sort_values('date')
        dates = group['date'].tolist()

        for i, (idx, row) in enumerate(group.iterrows()):
            current_date = row['date']
            actual_prev_llm = row.get('previousLLM')

            # 首条记录应该没有 previousLLM
            if i == 0:
                if pd.notna(actual_prev_llm):
                    errors += 1
                    print(f"  ❌ userid={userid}, date={current_date}: 首条记录不应有 previousLLM，实际={actual_prev_llm}")
                continue

            # 非首条：期望值 = 第一轮结果中该用户上一期的 llm_response
            prev_date = dates[i - 1]
            expected_prev_llm = prior_map.get((userid, prev_date))

            # 比较
            if expected_prev_llm is None:
                if pd.notna(actual_prev_llm):
                    errors += 1
                    print(f"  ❌ userid={userid}, date={current_date}: 期望=None (上期不在第一轮结果中), 实际={actual_prev_llm}")
            else:
                if str(actual_prev_llm) != str(expected_prev_llm):
                    errors += 1
                    print(f"  ❌ userid={userid}, date={current_date}: 期望={expected_prev_llm}, 实际={actual_prev_llm}")

            checked += 1

    if errors == 0:
        print(f"✅ previousLLM 计算正确! (验证了 {checked} 条非首条记录)")
    else:
        print(f"❌ 发现 {errors} 处错误 (共验证 {checked} 条)")

    # ================================================================
    # 显示示例用户数据
    # ================================================================
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

    # 显示关键列，包括第一轮的 llm_response 用于对比
    print(f"\n{'='*100}")
    print(f"{'date':<10} {'userid':<15} {'prior_llm (第一轮)':<20} {'previousLLM (计算值)':<20} {'match'}")
    print(f"{'='*100}")

    for _, row in df_show.iterrows():
        userid = row['userid']
        date = row['date']
        prev_llm = row.get('previousLLM', None)
        has_mem = pd.notna(prev_llm)

        # 获取该条记录在第一轮的 llm_response
        prior_llm = prior_map.get((userid, date), '-')
        prior_llm_str = str(prior_llm)[:15] + '...' if len(str(prior_llm)) > 15 else str(prior_llm)

        prev_llm_str = str(prev_llm)[:15] + '...' if has_mem and len(str(prev_llm)) > 15 else (str(prev_llm) if has_mem else '-')

        # 对于非首条，检查 previousLLM 是否等于上一期的 prior_llm
        print(f"{date:<10} {str(userid):<15} {prior_llm_str:<20} {prev_llm_str:<20}")

    # 显示一个完整的 prompt 示例
    print(f"\n{'='*60}")
    print("完整 Prompt 示例 (有记忆的记录):")
    print(f"{'='*60}")

    # 找一条有记忆的记录
    df_with_memory = df[df['previousLLM'].notna()]
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
