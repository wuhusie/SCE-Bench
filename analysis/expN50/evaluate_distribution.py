# import pandas as pd
# import numpy as np
# import json
# import ast
# from pathlib import Path
# import sys

# # Add project root to path
# PROJECT_ROOT = Path(__file__).resolve().parents[3]
# sys.path.insert(0, str(PROJECT_ROOT))

# from src.analysis.common.metrics import compute_mape

# # Configuration
# BASE_DIR = Path(r"/root/autodl-fs")
# MODEL_SUBDIR = "Qwen3-30B-A3B-Instruct-2507-FP8"

# CLEAN_DATA_DIR = BASE_DIR / "result_cleaned" / "exp1.3" / MODEL_SUBDIR
# OUTPUT_DIR = BASE_DIR / "result_analysed" / "exp1.3" / MODEL_SUBDIR

# def parse_response(response_str):
#     """Parse the string representation of list to actual list."""
#     try:
#         # Handles "[1, 2, 3]" format
#         return ast.literal_eval(response_str)
#     except:
#         return None

# def calculate_ecdf(samples, truth):
#     """
#     Calculate Empirical Cumulative Distribution Function value for truth in samples.
#     Formula: F_hat(h) = (1/N) * sum(I(s_i <= h))
#     """
#     if not samples or len(samples) == 0:
#         return None
    
#     # Ensure samples are numeric
#     try:
#         samples = [float(x) for x in samples]
#         truth = float(truth)
#     except ValueError:
#         return None
        
    
#     # Calculate Rank with Tie-Breaking (Randomized/Average Rank)
#     # This is crucial for discrete data or mode collapse (e.g. all samples = 0, truth = 0).
#     # Original formula 'sum(s<=t)/N' would give 1.0, which fails the <=0.95 check.
#     # New formula puts exact matches in the middle.
#     count_less = sum(1 for s in samples if s < truth)
#     count_equal = sum(1 for s in samples if s == truth)
    
#     return (count_less + 0.5 * count_equal) / len(samples)

# import matplotlib.pyplot as plt
# import seaborn as sns

# def plot_rank_histogram(ecdfs, output_path):
#     """
#     Plot histogram of ECDF values (PIT histogram).
#     Ideal: Uniform distribution.
#     U-shape: Under-confident.
#     A-shape / Inverted U: Over-confident.
#     """
#     plt.figure(figsize=(8, 6))
#     sns.histplot(ecdfs, bins=20, stat="density", kde=False, element="step", fill=True, alpha=0.3)
#     plt.axhline(1.0, color='r', linestyle='--', label="Ideal (Uniform)")
#     plt.title("Rank Histogram (PIT)")
#     plt.xlabel("Empirical CDF (Rank)")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.savefig(output_path)
#     plt.close()
#     print(f"   📊 Saved Rank Histogram: {output_path}")

# def plot_reliability_diagram(ecdfs, output_path):
#     """
#     Plot Reliability Diagram (Calibration Curve) for different confidence intervals.
#     """
#     alphas = np.linspace(0.1, 0.9, 9)
#     observed_coverages = []
    
#     for alpha in alphas:
#         # For a central interval of width alpha:
#         # Lower bound = (1 - alpha) / 2
#         # Upper bound = (1 + alpha) / 2
#         lower = (1 - alpha) / 2
#         upper = (1 + alpha) / 2
        
#         # Calculate coverage
#         coverage = np.mean([(e >= lower) and (e <= upper) for e in ecdfs])
#         observed_coverages.append(coverage)
        
#     plt.figure(figsize=(8, 8))
#     plt.plot(alphas, observed_coverages, "o-", label="Model")
#     plt.plot([0, 1], [0, 1], "k--", label="Ideal")
#     plt.xlabel("Expected Confidence Level (Target Probability)")
#     plt.ylabel("Observed Coverage (Acceptance Rate)")
#     plt.title("Reliability Diagram")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(output_path)
#     plt.close()
#     print(f"   📊 Saved Reliability Diagram: {output_path}")

# def evaluate_file(file_path):
#     print(f"\n📊 Analyzing: {file_path.name}")
#     try:
#         df = pd.read_csv(file_path)
#     except Exception as e:
#         print(f"❌ Failed to read CSV: {e}")
#         return None

#     # Identify columns
#     gt_col = df.columns[-1]
    
#     if 'llm_response' not in df.columns:
#         print(f"❌ 'llm_response' column missing in {file_path.name}")
#         return None

#     print(f"   (Ground Truth Column detected: '{gt_col}')")

#     results = []
#     ecdf_values = [] # Store all ECDF values for plotting
    
#     for _, row in df.iterrows():
#         # 1. Parse LLM samples
#         samples = parse_response(row['llm_response'])
#         if not samples or not isinstance(samples, list):
#             continue
            
#         # 2. Get Truth
#         truth = row[gt_col]
#         if pd.isna(truth):
#             continue

#         # 3. Calculate ECDF
#         ecdf = calculate_ecdf(samples, truth)
#         if ecdf is None:
#             continue
            
#         ecdf_values.append(ecdf)
            
#         # 4. Check Hit (90% CI: 0.05 <= ECDF <= 0.95)
#         is_hit = 0.05 <= ecdf <= 0.95
        
#         results.append({
#             'userid': row.get('userid', ''),
#             'ecdf': ecdf,
#             'is_hit': is_hit,
#             'truth': truth,
#             'mean_pred': np.mean(samples)
#         })

#     if not results:
#         print("⚠️ No valid rows found for evaluation.")
#         return None

#     # Aggregation
#     df_eval = pd.DataFrame(results)
    
#     total_valid = len(df_eval)
#     hit_count = df_eval['is_hit'].sum()
#     coverage_rate = hit_count / total_valid
    
#     # Calculate Mean Bias
#     df_eval['bias'] = df_eval['mean_pred'] - df_eval['truth']
#     mean_bias = df_eval['bias'].mean()
    
#     # Calculate MAPE (using mean of samples vs truth)
#     mape = compute_mape(df_eval['truth'].values, df_eval['mean_pred'].values)

#     print(f"   ✅ Valid Samples: {total_valid}")
#     print(f"   🎯 Coverage Rate (90% CI): {coverage_rate:.2%} (Target: ~90%)")
#     print(f"   📏 Mean Bias: {mean_bias:.4f}")
#     print(f"   📉 MAPE: {mape:.2f}%")

#     # Generate Plots
#     hist_path = OUTPUT_DIR / f"{file_path.stem}_rank_histogram.png"
#     reliability_path = OUTPUT_DIR / f"{file_path.stem}_reliability.png"
    
#     plot_rank_histogram(ecdf_values, hist_path)
#     plot_reliability_diagram(ecdf_values, reliability_path)

#     # Interpretation
#     interpretation = "Well-calibrated"
#     if coverage_rate < 0.8:
#         interpretation = "Over-confident (Distribution too narrow)"
#         print(f"   ⚠️  Result: {interpretation}")
#     elif coverage_rate > 0.95:
#         interpretation = "Under-confident (Distribution too wide)"
#         print(f"   ⚠️  Result: {interpretation}")
#     else:
#         print(f"   ✅ Result: {interpretation}")
        
#     return {
#         "file": file_path.name,
#         "n_samples": total_valid,
#         "metric_method": "Acceptance Rate via Percentile Rank (90% CI target)",
#         "coverage_rate": coverage_rate,
#         "interpretation": interpretation,
#         "mean_bias": mean_bias,
#         "mape": mape,
#         "rank_histogram": str(hist_path.name),
#         "reliability_diagram": str(reliability_path.name),
#         "status": "success"
#     }

# def main():
#     if not OUTPUT_DIR.exists():
#         OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#         print(f"📂 Created output directory: {OUTPUT_DIR}")

#     if not CLEAN_DATA_DIR.exists():
#         print(f"❌ Cleaned data directory not found: {CLEAN_DATA_DIR}")
#         return

#     print(f"🚀 Starting Distribution Evaluation in: {CLEAN_DATA_DIR}")
    
#     # Process all files ending with _withHumanData.csv
#     files = list(CLEAN_DATA_DIR.glob("*_withHumanData.csv"))
    
#     if not files:
#         print("⚠️ No data files found. Please run 'append_ground_truth.py' first.")
#         return
        
#     summary_report = []

#     for f in files:
#         res = evaluate_file(f)
#         if res:
#             summary_report.append(res)
            
#     # Save Report
#     if summary_report:
#         report_path = OUTPUT_DIR / "evaluation_report.json"
#         with open(report_path, 'w', encoding='utf-8') as f:
#             json.dump(summary_report, f, indent=4)
#         print(f"\n📄 Full report saved to: {report_path}")

# if __name__ == "__main__":
#     main()
import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.common.metrics import compute_mape, compute_js_divergence

# Configuration
BASE_DIR = Path(r"/root/autodl-fs")
MODEL_SUBDIR = "Qwen3-30B-A3B-Instruct-2507-FP8"

CLEAN_DATA_DIR = BASE_DIR / "result_cleaned" / "exp1.3" / MODEL_SUBDIR
OUTPUT_DIR = BASE_DIR / "result_analysed" / "exp1.3" / MODEL_SUBDIR

def parse_response(response_str):
    """Parse the string representation of list to actual list."""
    try:
        # Handles "[1, 2, 3]" format
        return ast.literal_eval(response_str)
    except:
        return None

def calculate_ecdf(samples, truth):
    """
    Calculate Empirical Cumulative Distribution Function value for truth in samples.
    Formula: F_hat(h) = (1/N) * sum(I(s_i <= h))
    """
    if not samples or len(samples) == 0:
        return None
    
    # Ensure samples are numeric
    try:
        samples = [float(x) for x in samples]
        truth = float(truth)
    except ValueError:
        return None
        
    
    # Calculate Rank with Tie-Breaking (Randomized/Average Rank)
    # This is crucial for discrete data or mode collapse (e.g. all samples = 0, truth = 0).
    # Original formula 'sum(s<=t)/N' would give 1.0, which fails the <=0.95 check.
    # New formula puts exact matches in the middle.
    count_less = sum(1 for s in samples if s < truth)
    count_equal = sum(1 for s in samples if s == truth)
    
    return (count_less + 0.5 * count_equal) / len(samples)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_rank_histogram(ecdfs, output_path):
    """
    Plot histogram of ECDF values (PIT histogram).
    Ideal: Uniform distribution.
    U-shape: Under-confident.
    A-shape / Inverted U: Over-confident.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(ecdfs, bins=20, stat="density", kde=False, element="step", fill=True, alpha=0.3)
    plt.axhline(1.0, color='r', linestyle='--', label="Ideal (Uniform)")
    plt.title("Rank Histogram (PIT)")
    plt.xlabel("Empirical CDF (Rank)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"   📊 Saved Rank Histogram: {output_path}")

def plot_reliability_diagram(ecdfs, output_path):
    """
    Plot Reliability Diagram (Calibration Curve) for different confidence intervals.
    """
    alphas = np.linspace(0.1, 0.9, 9)
    observed_coverages = []
    
    for alpha in alphas:
        # For a central interval of width alpha:
        # Lower bound = (1 - alpha) / 2
        # Upper bound = (1 + alpha) / 2
        lower = (1 - alpha) / 2
        upper = (1 + alpha) / 2
        
        # Calculate coverage
        coverage = np.mean([(e >= lower) and (e <= upper) for e in ecdfs])
        observed_coverages.append(coverage)
        
    plt.figure(figsize=(8, 8))
    plt.plot(alphas, observed_coverages, "o-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("Expected Confidence Level (Target Probability)")
    plt.ylabel("Observed Coverage (Acceptance Rate)")
    plt.title("Reliability Diagram")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"   📊 Saved Reliability Diagram: {output_path}")

def evaluate_file(file_path):
    print(f"\n📊 Analyzing: {file_path.name}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return None

    # Identify columns
    gt_col = df.columns[-1]
    
    if 'llm_response' not in df.columns:
        print(f"❌ 'llm_response' column missing in {file_path.name}")
        return None

    print(f"   (Ground Truth Column detected: '{gt_col}')")

    results = []
    ecdf_values = [] # Store all ECDF values for plotting
    
    for _, row in df.iterrows():
        # 1. Parse LLM samples
        samples = parse_response(row['llm_response'])
        if not samples or not isinstance(samples, list):
            continue
            
        # 2. Get Truth
        truth = row[gt_col]
        if pd.isna(truth):
            continue

        # 3. Calculate ECDF
        ecdf = calculate_ecdf(samples, truth)
        if ecdf is None:
            continue
            
        ecdf_values.append(ecdf)
            
        # 4. Check Hit (90% CI: 0.05 <= ECDF <= 0.95)
        is_hit = 0.05 <= ecdf <= 0.95
        
        results.append({
            'userid': row.get('userid', ''),
            'ecdf': ecdf,
            'is_hit': is_hit,
            'truth': truth,
            'mean_pred': np.mean(samples)
        })

    if not results:
        print("⚠️ No valid rows found for evaluation.")
        return None

    # Aggregation
    df_eval = pd.DataFrame(results)
    
    total_valid = len(df_eval)
    hit_count = df_eval['is_hit'].sum()
    coverage_rate = hit_count / total_valid
    
    # Calculate Mean Bias
    df_eval['bias'] = df_eval['mean_pred'] - df_eval['truth']
    mean_bias = df_eval['bias'].mean()
    
    # Calculate MAPE (using mean of samples vs truth)
    mape = compute_mape(df_eval['truth'].values, df_eval['mean_pred'].values)

    # Calculate JS Divergence (Human Truth vs List Mean)
    js_div = compute_js_divergence(df_eval['truth'].values, df_eval['mean_pred'].values)

    print(f"   ✅ Valid Samples: {total_valid}")
    print(f"   🎯 Coverage Rate (90% CI): {coverage_rate:.2%} (Target: ~90%)")
    print(f"   📏 Mean Bias: {mean_bias:.4f}")
    print(f"   📉 MAPE: {mape:.2f}%")
    if js_div is not None:
        print(f"   📊 JS Divergence: {js_div:.4f}")

    # Generate Plots
    hist_path = OUTPUT_DIR / f"{file_path.stem}_rank_histogram.png"
    reliability_path = OUTPUT_DIR / f"{file_path.stem}_reliability.png"
    
    plot_rank_histogram(ecdf_values, hist_path)
    plot_reliability_diagram(ecdf_values, reliability_path)

    # Interpretation
    interpretation = "Well-calibrated"
    if coverage_rate < 0.8:
        interpretation = "Over-confident (Distribution too narrow)"
        print(f"   ⚠️  Result: {interpretation}")
    elif coverage_rate > 0.95:
        interpretation = "Under-confident (Distribution too wide)"
        print(f"   ⚠️  Result: {interpretation}")
    else:
        print(f"   ✅ Result: {interpretation}")
        
    return {
        "file": file_path.name,
        "n_samples": total_valid,
        "metric_method": "Acceptance Rate via Percentile Rank (90% CI target)",
        "coverage_rate": coverage_rate,
        "interpretation": interpretation,
        "mean_bias": mean_bias,
        "mape": mape,
        "js_divergence": js_div,
        "rank_histogram": str(hist_path.name),
        "reliability_diagram": str(reliability_path.name),
        "status": "success"
    }

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📂 Created output directory: {OUTPUT_DIR}")

    if not CLEAN_DATA_DIR.exists():
        print(f"❌ Cleaned data directory not found: {CLEAN_DATA_DIR}")
        return

    print(f"🚀 Starting Distribution Evaluation in: {CLEAN_DATA_DIR}")
    
    # Process all files ending with _withHumanData.csv
    files = list(CLEAN_DATA_DIR.glob("*_withHumanData.csv"))
    
    if not files:
        print("⚠️ No data files found. Please run 'append_ground_truth.py' first.")
        return
        
    summary_report = []

    for f in files:
        res = evaluate_file(f)
        if res:
            summary_report.append(res)
            
    # Save Report
    if summary_report:
        report_path = OUTPUT_DIR / "evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=4)
        print(f"\n📄 Full report saved to: {report_path}")

if __name__ == "__main__":
    main()
