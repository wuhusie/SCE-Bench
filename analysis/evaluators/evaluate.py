"""
Evaluation entry script.

Usage:
    # Full workflow (Merge + Evaluate)
    python analysis/evaluators/evaluate.py --mode full --exp exp1 --model Qwen3-30B

    # Merge only
    python analysis/evaluators/evaluate.py --mode merge --exp exp1

    # Evaluate only
    python analysis/evaluators/evaluate.py --mode eval --type pointwise --input /path --output /path
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

import yaml

# Add project root to path
# evaluate.py is in analysis/evaluators/, so parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common.task_config import get_task_config, list_available_tasks
from analysis.common.merge import merge_all_tasks
from analysis.evaluators import PointwiseEvaluator, DistributionEvaluator


def run_merge(
    result_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    tasks: list = None
):
    """Run data merging."""
    return merge_all_tasks(
        result_dir=result_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        tasks=tasks
    )


def run_eval(
    eval_type: str,
    input_dir: str,
    output_dir: str,
    tasks: list = None,
    confidence_level: float = 0.90
) -> dict:
    """Run evaluation."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    tasks = tasks or ["spending", "labor", "credit"]

    print("=" * 60)
    print(f"Evaluation Type: {eval_type}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)

    if eval_type == "pointwise":
        evaluator = PointwiseEvaluator()
    else:
        evaluator = DistributionEvaluator(config={"confidence_level": confidence_level})

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for task_name in tasks:
        print(f"\n[{task_name.upper()}]")
        try:
            result = evaluator.evaluate_task(
                task_name=task_name,
                input_dir=input_dir,
                output_dir=output_dir,
                task_config=get_task_config(task_name)
            )
            results[task_name] = result
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[task_name] = {"error": str(e)}

    output_file = output_dir / f"metrics_{eval_type}.json"
    
    # Convert numpy types to native Python types
    results_serializable = convert_to_serializable(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {output_file}")
    return results_serializable


def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types. """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    else:
        return obj


def run_full(
    base_dir: Path,
    exp_name: str,
    model_name: str = None,
    tasks: list = None
):
    """Run full workflow: Merge + Pointwise Evaluation + Distribution Evaluation."""
    tasks = tasks or ["spending", "labor", "credit"]
    cache_dir = base_dir / "data" / "sce" / "cache"

    # Directory Structure:
    # result/exp1/N1/model_name/*.csv
    # result/exp1/N50/model_name/*.csv
    # result_cleaned/exp1/N1/model_name/*_withHumanData.csv
    # result_cleaned/exp1/N50/model_name/*_withHumanData.csv

    # 1. Merge all experimental data
    print("\n" + "=" * 60)
    print("Step 1/3: Merging experimental data")
    print("=" * 60)
    run_merge(
        result_dir=base_dir / "result" / exp_name,
        cache_dir=cache_dir,
        output_dir=base_dir / "result_cleaned" / exp_name,
        tasks=tasks
    )

    # Get list of models to evaluate
    n1_dir = base_dir / "result_cleaned" / exp_name / "N1"
    n50_dir = base_dir / "result_cleaned" / exp_name / "N50"

    if model_name:
        # Model specified, only evaluate this model
        n1_models = [model_name] if (n1_dir / model_name).exists() else []
        n50_models = [model_name] if (n50_dir / model_name).exists() else []
    else:
        # No model specified, evaluate all models
        n1_models = [d.name for d in n1_dir.iterdir() if d.is_dir()] if n1_dir.exists() else []
        n50_models = [d.name for d in n50_dir.iterdir() if d.is_dir()] if n50_dir.exists() else []

    # 2. Pointwise Evaluation (N1)
    print("\n" + "=" * 60)
    print(f"Step 2/3: Pointwise Evaluation (N1) - {len(n1_models)} models")
    print("=" * 60)
    for model in n1_models:
        print(f"\n>>> Model: {model}")
        run_eval(
            eval_type="pointwise",
            input_dir=n1_dir / model,
            output_dir=base_dir / "result_analysed" / exp_name / "N1" / model,
            tasks=tasks
        )

    # 3. Distribution Evaluation (N50)
    print("\n" + "=" * 60)
    print(f"Step 3/3: Distribution Evaluation (N50) - {len(n50_models)} models")
    print("=" * 60)
    for model in n50_models:
        print(f"\n>>> Model: {model}")
        run_eval(
            eval_type="distribution",
            input_dir=n50_dir / model,
            output_dir=base_dir / "result_analysed" / exp_name / "N50" / model,
            tasks=tasks
        )

    print("\n" + "=" * 60)
    print("All completed")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Experiment Evaluation")
    parser.add_argument("--mode", choices=["full", "merge", "eval"], default="eval",
                        help="Run mode: full=full workflow, merge=merge only, eval=eval only")

    # full/merge mode parameters
    parser.add_argument("--base-dir", type=Path, default=Path("/root/autodl-fs"))
    parser.add_argument("--exp", type=str, help="Experiment name (e.g., exp1)")
    parser.add_argument("--model", type=str, help="Model name")

    # eval mode parameters
    parser.add_argument("--type", "-t", choices=["pointwise", "distribution"])
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--tasks", nargs="+", default=["spending", "labor", "credit"])
    parser.add_argument("--confidence-level", type=float, default=0.90)

    # Configuration file
    parser.add_argument("--config", "-c", type=Path)

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        run_eval(
            eval_type=config["evaluation"]["type"],
            input_dir=config["paths"]["input_dir"],
            output_dir=config["paths"]["output_dir"],
            tasks=config.get("tasks", ["spending", "labor", "credit"]),
            confidence_level=config.get("distribution_settings", {}).get("confidence_level", 0.90)
        )
    elif args.mode == "full":
        if not args.exp:
            print("Error: --mode full requires --exp parameter")
            sys.exit(1)
        run_full(args.base_dir, args.exp, args.model, args.tasks)
    elif args.mode == "merge":
        if not args.exp:
            print("Error: --mode merge requires --exp parameter")
            sys.exit(1)
        cache_dir = args.base_dir / "data" / "sce" / "cache"
        run_merge(
            result_dir=args.base_dir / "result" / args.exp,
            cache_dir=cache_dir,
            output_dir=args.base_dir / "result_cleaned" / args.exp,
            tasks=args.tasks
        )
    elif args.mode == "eval":
        if not args.type or not args.input or not args.output:
            print("Error: --mode eval requires --type, --input, --output parameters")
            sys.exit(1)
        run_eval(args.type, args.input, args.output, args.tasks, args.confidence_level)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
