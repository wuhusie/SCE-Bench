"""
MAE Grouped Merged Visualization Script.

Compares the error distributions of multiple models in the same plot:
- Group 1: GPT-5 Thinking vs. No-Thinking
- Group 2: Gemini Thinking vs. No-Thinking
- Group 3: GPT-3.5 and GPT-4o-mini
- Group 4: Qwen Thinking vs. No-Thinking

Usage:
    python mae_grouped.py
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import path configurations from config; plotting parameters are defined within this script.
from config import CACHE_ERRORS, PIC_DIR, MODEL_GROUPS


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
FIGURE_SIZE = (12, 8)                                # Chart size: 12 in wide x 8 in high



# ----------------------------------------------------------------------------
# Color Configuration
# Unified two-color scheme, consistent across all grouped plots
# ----------------------------------------------------------------------------
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
MEAN_LINE_LINEWIDTH = 2                              # Mean line width
MEAN_LINE_ALPHA = 0.8                                # Mean line transparency
ZERO_LINE_LINEWIDTH = 2                              # Zero error line width
ZERO_LINE_ALPHA = 0.7                                # Zero error line transparency
ZERO_LINE_COLOR = '#333333'                          # Zero error line color: Dark gray

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


def plot_grouped_error_distribution(
    model_errors: Dict[str, np.ndarray],
    group_name: str,
    task_name: str,
    x_range: Tuple[float, float],
    output_path: Path,
    y_max: float = None
):
    """
    Plot grouped error distribution (multiple models).

    Parameters:
        model_errors: {model_dir_name: array_of_errors}
        group_name: Name of the group
        task_name: Name of the task
        x_range: X-axis range
        output_path: Output path
        y_max: Upper limit for Y-axis
    """
    # Reset and apply style configuration
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(KDE_STYLE_CONFIG)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Get Y-axis label from task configuration to construct X-axis label
    task_config = TASK_CONFIGS.get(task_name, {})
    xlabel = f"Error ({task_config.get('ylabel', 'Value').replace(' (%)', '')})"

    # List of model colors (assigned in order)
    model_color_list = [MODEL_1_COLOR, MODEL_2_COLOR]

    # Draw error distribution for each model
    for idx, (model_key, errors) in enumerate(model_errors.items()):
        display_name = get_display_name(model_key)
        colors = model_color_list[idx % 2]

        # Calculate statistics
        mean_error = np.mean(errors)                 # Mean error
        mae = np.mean(np.abs(errors))                # Mean Absolute Error (MAE)

        # Draw KDE (with fill shadow)
        sns.kdeplot(
            errors, ax=ax,
            color=colors["edge"],
            fill=True,
            facecolor=colors["fill"],
            alpha=KDE_FILL_ALPHA,
            linewidth=0,
            label=f"{display_name} (MAE={mae:.2f})"
        )
        # Draw edge line
        sns.kdeplot(
            errors, ax=ax,
            color=colors["edge"],
            linewidth=KDE_EDGE_LINEWIDTH,
            linestyle="-",
            label="_nolegend_"
        )

        # Add mean line (dotted line marking the average error position)
        ax.axvline(
            x=mean_error,
            color=colors["edge"],
            linestyle=':',
            linewidth=MEAN_LINE_LINEWIDTH,
            alpha=MEAN_LINE_ALPHA
        )

    # Add zero line (ideal error position)
    ax.axvline(
        x=0,
        color=ZERO_LINE_COLOR,
        linestyle='--',
        linewidth=ZERO_LINE_LINEWIDTH,
        alpha=ZERO_LINE_ALPHA,
        label='Zero Error'
    )

    # Set axis labels
    ax.set_xlabel(xlabel, labelpad=XLABEL_LABELPAD)
    ax.set_ylabel("Density", labelpad=YLABEL_LABELPAD)

    # Set axis ranges
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_max is not None:
        ax.set_ylim(0, y_max)

    # Configure legend
    # All task legends placed in top left corner
    legend_loc = "upper left"

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
    
    parser = argparse.ArgumentParser(description='MAE Grouped Merged Visualization Script')
    parser.add_argument('--task', nargs='+', choices=['spending', 'labor', 'credit'], 
                        help='Specify tasks to run (spending, labor, credit)')
    parser.add_argument('--group', nargs='+', 
                        help='Specify groups to run (supports fuzzy matching, e.g., GPT, Gemini)')
    args = parser.parse_args()

    print("=" * 60)
    print("  MAE Grouped Visualization")
    print("=" * 60)

    # Load cache
    print(f"Loading cache: {CACHE_ERRORS}")
    all_errors = load_cache(CACHE_ERRORS)

    # Determine tasks to run
    tasks_to_run = args.task if args.task else ["spending", "labor", "credit"]

    # Generate charts for each task
    for task_name in tasks_to_run:
        if task_name not in all_errors:
            print(f"\n[{task_name}] No data, skipping")
            continue

        print(f"\n[{task_name.upper()}]")

        task_errors = all_errors[task_name]

        # Create output directory
        output_dir = PIC_DIR / "mae_grouped" / task_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate global X-axis range (no truncation, using full range)
        all_task_errors = np.concatenate(list(task_errors.values()))
        x_min = np.min(all_task_errors)
        x_max = np.max(all_task_errors)
        x_range = (x_min, x_max)

        # Calculate global Y-axis upper limit (Peak density across all models * scale factor)
        y_max = 0
        for errors in task_errors.values():
            peak = get_kde_peak(errors, x_range)
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

            # Collect model errors for this group
            group_errors = {}
            for model_key in model_keys:
                if model_key in task_errors:
                    group_errors[model_key] = task_errors[model_key]
                else:
                    print(f"    [WARN] Model {model_key} no data")

            if not group_errors:
                print(f"    [WARN] Group {group_name} no valid model, skip")
                continue

            output_path = output_dir / f"mae_{group_name}.pdf"

            plot_grouped_error_distribution(
                model_errors=group_errors,
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
    print(f"Images saved to: {PIC_DIR / 'mae_grouped'}")


if __name__ == "__main__":
    main()
