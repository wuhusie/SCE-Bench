"""
JS Divergence Grouped Merged Visualization Script.

Compares multiple models against Human ground truth in the same plot:
- Group 1: GPT-5 Thinking vs. No-Thinking
- Group 2: Gemini Thinking vs. No-Thinking
- Group 3: GPT-3.5 and GPT-4o-mini
- Group 4: Qwen Thinking vs. No-Thinking

Usage:
    python js_divergence_grouped.py
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import path configurations from config; plotting parameters are defined within this script.
from config import CACHE_RAW_SAMPLES, PIC_DIR, MODEL_GROUPS


# ============================================================================
# Plotting Parameter Configuration (Defined here for absolute override)
# ============================================================================

# ----------------------------------------------------------------------------
# Model Display Name Mapping
# Maps model directory names to short names for charts
# ----------------------------------------------------------------------------
MODEL_DISPLAY_NAMES = {
    "gpt-3.5-turbo": "GPT3.5",                       # OpenAI GPT-3.5 Turbo model
    "gpt-4o-mini": "GPT4om",                         # OpenAI GPT-4o-mini model
    "gpt-5-mini-medium": "GPT5m-nt",                 # OpenAI GPT-5-mini No-Thinking mode
    "gpt-5-mini-minimal": "GPT5m-t",                 # OpenAI GPT-5-mini Thinking mode
    "gemini-3-flash-preview-thinking": "GEM3f-t",   # Google Gemini 3 Flash Thinking mode
    "gemini-3-flash-preview-nothinking": "GEM3f-nt", # Google Gemini 3 Flash No-Thinking mode
    "Qwen3-30B-A3B-Instruct-2507-FP8": "QWE3-nt",   # Alibaba Qwen3-30B No-Thinking mode
    "Qwen3-30B-Thinking": "QWE3-t",                 # Alibaba Qwen3-30B Thinking mode
}

# ----------------------------------------------------------------------------
# Task Configuration
# Defines data column names, file pattern, Y-axis labels, etc., for each task
# ----------------------------------------------------------------------------
TASK_CONFIGS = {
    "spending": {
        "llm_col": "llm_response",                   # LLM response column name
        "human_col": "Q26v2part2",                   # Human data column name
        "file_pattern": "spending_*_withHumanData.csv",  # Data file matching pattern
        "ylabel": "Spending Change(%)",           # Y-axis label: Expected income growth percentage
        "cleaning_method": "iqr",                    # Data cleaning method: Interquartile Range (IQR)
    },
    "labor": {
        "llm_col": "llm_response",                   # LLM response column name
        "human_col": "oo2c3",                        # Human data column name
        "file_pattern": "labor_*_withHumanData.csv", # Data file matching pattern
        "ylabel": "Acceptance Prob.(%)",             # Y-axis label: Acceptance probability percentage
        "cleaning_method": "range",                  # Data cleaning method: Range filtering
        "valid_range": (0, 100),                     # Valid range: 0-100%
    },
    "credit": {
        "llm_col": "llm_response",                   # LLM response column name
        "human_col": "N17b_2",                       # Human data column name
        "file_pattern": "credit_*_withHumanData.csv", # Data file matching pattern
        "ylabel": "Application Prob.(%)",            # Y-axis label: Application probability percentage
        "cleaning_method": "range",                  # Data cleaning method: Range filtering
        "valid_range": (0, 100),                     # Valid range: 0-100%
    }
}

# ----------------------------------------------------------------------------
# KDE Distribution Plot Style Configuration
# Controls overall chart appearance: fonts, line widths, resolution, etc.
# ----------------------------------------------------------------------------
KDE_STYLE_CONFIG = {
    'figure.dpi': 300,                               # Image display resolution
    'savefig.dpi': 300,                              # Image saving resolution
    'font.family': 'serif',                          # Font family: Serif
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],  # Priority list for serif fonts
    'font.size': 28,                                 # Global default font size
    'axes.labelsize': 34,                            # Font size for axis labels
    'xtick.labelsize': 28,                           # Font size for X-tick labels
    'ytick.labelsize': 28,                           # Font size for Y-tick labels
    'legend.fontsize': 30,                           # Font size for legends
    'lines.linewidth': 3.0,                          # Default line width
    'axes.linewidth': 2.0,                           # Axis spine line width
}

# ----------------------------------------------------------------------------
# Chart Size Configuration
# ----------------------------------------------------------------------------
FIGURE_SIZE = (10, 8)                                # Chart size: 10 in wide x 8 in high

# ----------------------------------------------------------------------------
# Grouping Configuration
# Defines model groups, each containing a group name and a list of model directory names
# ----------------------------------------------------------------------------
MODEL_GROUPS = [
    ("GPT-5", ["gpt-5-mini-medium", "gpt-5-mini-minimal"]),       # GPT-5 Thinking vs. No-Thinking
    ("Gemini", ["gemini-3-flash-preview-nothinking", "gemini-3-flash-preview-thinking"]),  # Gemini Thinking vs. No-Thinking
    ("GPT-3.5_4o-mini", ["gpt-3.5-turbo", "gpt-4o-mini"]),        # GPT-3.5 and GPT-4o-mini
    ("Qwen", ["Qwen3-30B-A3B-Instruct-2507-FP8", "Qwen3-30B-Thinking"]),  # Qwen Thinking vs. No-Thinking
]

# ----------------------------------------------------------------------------
# Color Configuration
# Unified three-color scheme: Human + Two Models
# ----------------------------------------------------------------------------
HUMAN_COLOR = {
    "edge": "#7B6FA3",                               # Human edge color: Purple
    "fill": "#BEB8DA"                                # Human fill color: Light purple
}
MODEL_1_COLOR = {
    "edge": "#1F77B4",                               # Model 1 edge color: Standard academic blue
    "fill": "#AEC7E8"                                # Model 1 fill color: Light blue
}
MODEL_2_COLOR = {
    "edge": "#E76F51",                               # Model 2 edge color: Coral orange
    "fill": "#FACEC3"                                # Model 2 fill color: Light coral
}

# ----------------------------------------------------------------------------
# KDE Plotting Parameters
# ----------------------------------------------------------------------------
KDE_FILL_ALPHA = 0.4                                 # KDE fill transparency
KDE_EDGE_LINEWIDTH = 1.5                             # KDE edge line width
HUMAN_LINESTYLE = "--"                               # Human KDE edge linestyle: Dashed
MODEL_LINESTYLE = "-"                                # Model KDE edge linestyle: Solid

# ----------------------------------------------------------------------------
# JS Divergence Calculation Parameters
# ----------------------------------------------------------------------------
JS_N_POINTS = 200                                    # Number of grid points for KDE calculation
JS_PERCENTILE_LOW = 1                                # Lower quantile for X-axis range
JS_PERCENTILE_HIGH = 99                              # Upper quantile for X-axis range
JS_EPSILON = 1e-10                                   # Small constant to prevent log(0)

# ----------------------------------------------------------------------------
# Legend Configuration
# ----------------------------------------------------------------------------
LEGEND_LOC = "upper right"                           # Legend location: upper right corner
LEGEND_FRAMEON = True                                # Whether to show legend frame
LEGEND_FANCYBOX = True                               # Whether the frame has rounded corners
LEGEND_FRAMEALPHA = 0.95                             # Legend background transparency
LEGEND_FONTSIZE = 30                                 # Legend font size

# ----------------------------------------------------------------------------
# Grid Line Configuration
# ----------------------------------------------------------------------------
GRID_LINESTYLE = ':'                                 # Grid linestyle: Dotted
GRID_ALPHA = 0.5                                     # Grid transparency
GRID_LINEWIDTH = 1.5                                 # Grid line width

# ----------------------------------------------------------------------------
# Axis Label Configuration
# ----------------------------------------------------------------------------
XLABEL_LABELPAD = 15                                 # Spacing between X-label and axis
YLABEL_LABELPAD = 15                                 # Spacing between Y-label and axis

# ----------------------------------------------------------------------------
# Output Configuration
# ----------------------------------------------------------------------------
OUTPUT_DPI = 300                                     # Output image resolution
Y_MAX_SCALE = 1.2                                    # Y-axis upper limit scaling factor (Peak * this value)


# ============================================================================
# Function Definitions
# ============================================================================

def get_display_name(model_key: str) -> str:
    """Get the display name of the model."""
    if model_key == "Human":
        return "Human"
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def compute_js_divergence(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    x_range: Tuple[float, float] = None,
    n_points: int = JS_N_POINTS
) -> float:
    """
    Calculate JS Divergence between two distributions.

    JS Divergence is a symmetric version of KL Divergence, ranging between [0, ln(2)].
    A smaller value indicates higher similarity between the two distributions.

    Parameters:
        samples_p: Array of samples from the first distribution
        samples_q: Array of samples from the second distribution
        x_range: Calculation range, defaults to 1%-99% quantiles
        n_points: Number of KDE grid points

    Returns:
        JS Divergence value, or NaN if calculation fails
    """
    if len(samples_p) < 2 or len(samples_q) < 2:
        return np.nan

    try:
        # Determine calculation range
        if x_range is None:
            all_samples = np.concatenate([samples_p, samples_q])
            x_min = np.percentile(all_samples, JS_PERCENTILE_LOW)
            x_max = np.percentile(all_samples, JS_PERCENTILE_HIGH)
            x_range = (x_min, x_max)

        # Build KDE estimators
        kde_p = gaussian_kde(samples_p)
        kde_q = gaussian_kde(samples_q)

        # Create grid
        x_grid = np.linspace(x_range[0], x_range[1], n_points)

        # Calculate density values
        p_values = kde_p(x_grid)
        q_values = kde_q(x_grid)

        # Normalize (Ensure integral equals 1)
        p_values = p_values / np.trapezoid(p_values, x_grid)
        q_values = q_values / np.trapezoid(q_values, x_grid)

        # Calculate midpoint distribution M = (P + Q) / 2
        m_values = 0.5 * (p_values + q_values)

        # Add small constant to prevent log(0)
        p_values = np.maximum(p_values, JS_EPSILON)
        q_values = np.maximum(q_values, JS_EPSILON)
        m_values = np.maximum(m_values, JS_EPSILON)

        # Calculate KL(P||M) and KL(Q||M)
        kl_pm = np.trapezoid(p_values * np.log(p_values / m_values), x_grid)
        kl_qm = np.trapezoid(q_values * np.log(q_values / m_values), x_grid)

        # JS Divergence = (KL(P||M) + KL(Q||M)) / 2
        js_div = 0.5 * kl_pm + 0.5 * kl_qm

        return js_div

    except Exception as e:
        print(f"  [WARN] JS calc error: {e}")
        return np.nan


def get_kde_peak(samples: np.ndarray, x_range: Tuple[float, float], n_points: int = 200) -> float:
    """
    Calculate the peak KDE density.

    Parameters:
        samples: Array of samples
        x_range: Calculation range
        n_points: Number of grid points

    Returns:
        Peak density value, or 0 if calculation fails
    """
    if len(samples) < 2:
        return 0
    try:
        kde = gaussian_kde(samples)
        x_grid = np.linspace(x_range[0], x_range[1], n_points)
        return np.max(kde(x_grid))
    except:
        return 0


def load_cache(path: Path) -> Dict[str, Any]:
    """Load cache from pkl file."""
    if not path.exists():
        raise FileNotFoundError(f"Cache file does not exist: {path}\nPlease run data/prepare_cache.py first.")

    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_grouped_kde(
    human_samples: np.ndarray,
    model_samples: Dict[str, np.ndarray],
    group_name: str,
    task_name: str,
    x_range: Tuple[float, float],
    output_path: Path,
    y_max: float = None
):
    """
    Plot grouped KDE density comparison (multiple models + Human).

    Parameters:
        human_samples: Array of human samples
        model_samples: {model_dir_name: array_of_samples}
        group_name: Name of the group
        task_name: Name of the task
        x_range: X-axis range
        output_path: Path to the output file
        y_max: Upper limit for Y-axis
    """
    # Reset and apply style configuration
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(KDE_STYLE_CONFIG)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Get X-axis label from task configuration
    task_config = TASK_CONFIGS.get(task_name, {})
    xlabel = task_config.get("ylabel", "Value")

    # Draw Human KDE (Bottom layer, fill + dashed edge)
    sns.kdeplot(
        human_samples, ax=ax,
        color=HUMAN_COLOR["edge"],
        fill=True,
        facecolor=HUMAN_COLOR["fill"],
        alpha=KDE_FILL_ALPHA,
        linewidth=0,
        label="Human"
    )
    sns.kdeplot(
        human_samples, ax=ax,
        color=HUMAN_COLOR["edge"],
        linewidth=KDE_EDGE_LINEWIDTH,
        linestyle=HUMAN_LINESTYLE,
        label="_nolegend_"
    )

    # Store JS values for annotation
    js_values = {}

    # List of model colors (assigned in order)
    model_color_list = [MODEL_1_COLOR, MODEL_2_COLOR]

    # Draw KDE for each model
    for idx, (model_key, samples) in enumerate(model_samples.items()):
        display_name = get_display_name(model_key)
        colors = model_color_list[idx % 2]

        # Calculate JS Divergence against Human
        js_value = compute_js_divergence(samples, human_samples, x_range)
        js_values[display_name] = js_value

        # Draw KDE (with fill shadow)
        sns.kdeplot(
            samples, ax=ax,
            color=colors["edge"],
            fill=True,
            facecolor=colors["fill"],
            alpha=KDE_FILL_ALPHA,
            linewidth=0,
            label=f"{display_name} (JS={js_value:.4f})"
        )
        # Draw edge line
        sns.kdeplot(
            samples, ax=ax,
            color=colors["edge"],
            linewidth=KDE_EDGE_LINEWIDTH,
            linestyle=MODEL_LINESTYLE,
            label="_nolegend_"
        )

    # Set axis labels
    ax.set_xlabel(xlabel, labelpad=XLABEL_LABELPAD)
    ax.set_ylabel("Density", labelpad=YLABEL_LABELPAD)

    # Set Y-axis range
    if y_max is not None:
        ax.set_ylim(0, y_max)

    # Configure legend
    # All task legends placed in top right corner
    legend_loc = "upper right"

    ax.legend(
        loc=legend_loc,
        frameon=LEGEND_FRAMEON,
        fancybox=LEGEND_FANCYBOX,
        framealpha=0.9,             # Background transparency 10%
        fontsize=24,                # Reduced font size (from 30 -> 24)
        labelspacing=0.3,           # Compressed line spacing
    )

    # Configure grid lines
    ax.grid(
        True,
        linestyle=GRID_LINESTYLE,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=OUTPUT_DPI)
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='JS Divergence Grouped Merged Visualization Script')
    parser.add_argument('--task', nargs='+', choices=['spending', 'labor', 'credit'], 
                        help='Specify tasks to run (spending, labor, credit)')
    parser.add_argument('--group', nargs='+', 
                        help='Specify groups to run (supports fuzzy matching, e.g., GPT, Gemini)')
    args = parser.parse_args()

    print("=" * 60)
    print("  JS Divergence Grouped Visualization")
    print("=" * 60)

    # Load cache
    print(f"Loading cache: {CACHE_RAW_SAMPLES}")
    all_samples = load_cache(CACHE_RAW_SAMPLES)

    # Determine tasks to run
    tasks_to_run = args.task if args.task else ["spending", "labor", "credit"]

    # Generate charts for each task
    for task_name in tasks_to_run:
        if task_name not in all_samples:
            print(f"\n[{task_name}] No data, skipping")
            continue

        print(f"\n[{task_name.upper()}]")

        task_samples = all_samples[task_name]

        if "Human" not in task_samples:
            print("  [WARN] No Human data, skip")
            continue

        human_samples = task_samples["Human"]

        # Create output directory
        output_dir = PIC_DIR / "js_grouped" / task_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate global X-axis range (using 1%-99% quantiles)
        all_task_samples = np.concatenate([s for s in task_samples.values()])
        x_min = np.percentile(all_task_samples, JS_PERCENTILE_LOW)
        x_max = np.percentile(all_task_samples, JS_PERCENTILE_HIGH)
        x_range = (x_min, x_max)

        # Calculate global Y-axis upper limit (Peak density across all distributions * scale factor)
        y_max = 0
        for name, samples in task_samples.items():
            peak = get_kde_peak(samples, x_range)
            if peak > y_max:
                y_max = peak
        y_max *= Y_MAX_SCALE

        # Generate plots for each group
        for group_name, model_keys in MODEL_GROUPS:
            # Filter groups
            if args.group:
                if not any(g.lower() in group_name.lower() for g in args.group):
                    continue
            
            print(f"  Plotting group: {group_name}")

            # Collect model samples for this group
            group_samples = {}
            for model_key in model_keys:
                if model_key in task_samples:
                    group_samples[model_key] = task_samples[model_key]
                else:
                    print(f"    [WARN] Model {model_key} no data")

            if not group_samples:
                print(f"    [WARN] Group {group_name} no valid model, skip")
                continue

            output_path = output_dir / f"js_{group_name}.pdf"

            plot_grouped_kde(
                human_samples=human_samples,
                model_samples=group_samples,
                group_name=group_name,
                task_name=task_name,
                x_range=x_range,
                output_path=output_path,
                y_max=y_max
            )
            print(f"    [OK] Saved: {output_path}")

    print("\n" + "=" * 60)
    print("  Completed!")
    print("=" * 60)
    print(f"Images saved to: {PIC_DIR / 'js_grouped'}")


if __name__ == "__main__":
    main()
