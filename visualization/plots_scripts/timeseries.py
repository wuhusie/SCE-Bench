"""
Script for plotting timeseries data.

Reads data from pkl cache and plots comparative timeseries of LLM predictions vs. Human ground truth.
Generates separate images for each of the three tasks (spending, labor, credit).

Usage:
    python timeseries.py
"""
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import path configurations from config; plotting parameters are defined within this script.
from config import CACHE_MONTHLY_STATS, PIC_DIR


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
# LaTeX Size Configuration
# Used for generating images sized for academic paper layouts
# ----------------------------------------------------------------------------
LATEX_TEXT_WIDTH = 506.295                           # \textwidth: Full-column width (pt)
LATEX_COL_WIDTH = 241.14749                          # \linewidth: Single-column width (pt)
INCHES_PER_PT = 1 / 72.27                            # Point to Inch conversion factor

# ----------------------------------------------------------------------------
# Timeseries Plot Size Calculation Function
# ----------------------------------------------------------------------------
def set_size(width_pt: float, fraction: float = 1.0, ratio: str | float = 'golden'):
    """
    Calculate figsize suitable for LaTeX.

    Parameters:
        width_pt: Target width in LaTeX (pt)
        fraction: Fraction of the target width the image should occupy
        ratio: 'golden' (golden ratio) or float (custom aspect ratio height/width)

    Returns:
        (width_in, height_in) tuple
    """
    fig_width_pt = width_pt * fraction
    fig_width_in = fig_width_pt * INCHES_PER_PT

    if ratio == 'golden':
        golden_ratio = (5**.5 - 1) / 2               # Golden ratio ≈ 0.618
        fig_height_in = fig_width_in * golden_ratio
    else:
        fig_height_in = fig_width_in * ratio

    return (fig_width_in, fig_height_in)

# ----------------------------------------------------------------------------
# Timeseries Plot Style Configuration
# Adapted for 1 row 3 columns layout, spanning double columns for layout
# ----------------------------------------------------------------------------
TIMESERIES_STYLE_CONFIG = {
    'figure.dpi': 300,                               # Image display resolution
    'savefig.dpi': 300,                              # Image saving resolution
    'font.family': 'serif',                          # Font family: Serif
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],  # Priority list for serif fonts
    'mathtext.fontset': 'stix',                      # Font set for math text
    'font.size': 8,                                  # Global default font size
    'axes.labelsize': 9,                             # Font size for axis labels
    'axes.titlesize': 9,                             # Font size for plot titles
    'xtick.labelsize': 7,                            # Font size for X-tick labels
    'ytick.labelsize': 7,                            # Font size for Y-tick labels
    'legend.fontsize': 7,                            # Font size for legends
    'lines.linewidth': 1.0,                          # Default line width
    'axes.linewidth': 0.5,                           # Axis spine line width
    'xtick.major.width': 0.5,                        # Major tick width for X-axis
    'ytick.major.width': 0.5,                        # Major tick width for Y-axis
    'xtick.major.size': 3,                           # Major tick mark size for X-axis
    'ytick.major.size': 3,                           # Major tick mark size for Y-axis
    'grid.linewidth': 0.3,                           # Grid line width
}

# ----------------------------------------------------------------------------
# Timeseries Plot Size
# 1 row 3 columns layout, each subplot occupying 1/3 width, aspect ratio 0.75
# ----------------------------------------------------------------------------
TIMESERIES_FIGURE_SIZE = set_size(LATEX_TEXT_WIDTH, fraction=1/3, ratio=0.75)

# ----------------------------------------------------------------------------
# Color Configuration
# Replicating classic academic paper color scheme
# ----------------------------------------------------------------------------
MAIN_COLORS = {
    "Human": "#2C2C2C",                              # Charcoal black - Authoritative baseline
    "GPT3.5": "#1F77B4",                             # Standard academic blue
    "GPT4om": "#6BAED6",                             # Light blue
    "GPT5m-t": "#7B4B94",                            # Dark purple
    "GPT5m-nt": "#C9B3DB",                           # Light purple
    "GEM3f-t": "#E76F51",                            # Coral orange
    "GEM3f-nt": "#F4A582",                           # Light coral orange
    "QWE3-nt": "#7ECAC0",                            # Light turquoise
    "QWE3-t": "#2A9D8F",                             # Turquoise
}
DEFAULT_COLOR = "#888888"                            # Default color for unknown models: Gray
# ----------------------------------------------------------------------------
# Line Style Configuration
# Solid line for thinking mode, dashed line for no-thinking mode
# ----------------------------------------------------------------------------
LINE_STYLES = {
    "Human": "-",                                    # Human: Solid
    "GPT3.5": "-",                                   # GPT3.5: Solid
    "GPT5m-t": "-",                                  # GPT5-mini Thinking: Solid
    "GPT5m-nt": "--",                                # GPT5-mini No-Thinking: Dashed
    "GEM3f-t": "-",                                  # Gemini Thinking: Solid
    "GEM3f-nt": "--",                                # Gemini No-Thinking: Dashed
    "QWE3-nt": "--",                                 # Qwen No-Thinking: Dashed
    "QWE3-t": "-",                                   # Qwen Thinking: Solid
}
DEFAULT_LINESTYLE = "-"                              # Default linestyle for unknown models: Solid

# ----------------------------------------------------------------------------
# Line Width Configuration
# Adapted for small size plots, Human slightly thicker for baseline emphasis
# ----------------------------------------------------------------------------
LINE_WIDTHS = {
    "Human": 0.8,                                    # Human line width: slightly thicker, as baseline reference
    "GPT3.5": 0.5,                                   # GPT3.5 line width
    "GPT4om": 0.5,                                   # GPT4o-mini line width
    "GPT5m-t": 0.5,                                  # GPT5-mini Thinking mode line width
    "GPT5m-nt": 0.5,                                 # GPT5-mini No-Thinking mode line width
    "GEM3f-t": 0.5,                                  # Gemini Thinking mode line width
    "GEM3f-nt": 0.5,                                 # Gemini No-Thinking mode line width
    "QWE3-nt": 0.5,                                  # Qwen No-Thinking mode line width
    "QWE3-t": 0.5,                                   # Qwen Thinking mode line width
}
DEFAULT_LINEWIDTH = 2.0                              # Default linewidth for unknown models

# ----------------------------------------------------------------------------
# Plotting Order
# Drawing order from bottom level to top level, Human on top level for visibility
# ----------------------------------------------------------------------------
PLOT_ORDER = [
    "GPT5m-nt",                                      # Layer 1: GPT5-mini No-Thinking
    "GEM3f-nt",                                      # Layer 2: Gemini No-Thinking
    "QWE3-t",                                        # Layer 3: Qwen Thinking
    "GPT3.5",                                        # Layer 4: GPT3.5
    "GPT5m-t",                                       # Layer 5: GPT5-mini Thinking
    "GEM3f-t",                                       # Layer 6: Gemini Thinking
    "QWE3-nt",                                       # Layer 7: Qwen No-Thinking
    "Human",                                         # Layer 8 (Top): Human Baseline
]

# ----------------------------------------------------------------------------
# Line Alpha/Transparency
# ----------------------------------------------------------------------------
LINE_ALPHA = 0.9                                     # Line Alpha/Transparency

# ----------------------------------------------------------------------------
# Grid Line Configuration
# ----------------------------------------------------------------------------
GRID_AXIS = 'y'                                      # Grid line direction: Y-axis only
GRID_ALPHA = 0.3                                     # Grid line transparency
GRID_LINESTYLE = "--"                                # Grid linestyle: Dashed
GRID_COLOR = 'gray'                                  # Grid line color: Gray
GRID_LINEWIDTH = 0.3                                 # Grid line width

# ----------------------------------------------------------------------------
# Y-axis Range Configuration
# ----------------------------------------------------------------------------
Y_PADDING_RATIO = 0.1                                # Y-axis padding ratio (10% padding for top and bottom)

# ----------------------------------------------------------------------------
# Axis Label Configuration
# ----------------------------------------------------------------------------
XLABEL_TEXT = "Date"                                 # X-axis label text
XLABEL_LABELPAD = 4                                  # Spacing between X-label and axis
YLABEL_LABELPAD = 4                                  # Spacing between Y-label and axis
SPENDING_YLABEL_Y = 0.5                            # Vertical position for spending task Y-axis label (special adjustment)

# ----------------------------------------------------------------------------
# Date Format Configuration
# ----------------------------------------------------------------------------
DATE_FORMAT = '%Y'                                   # Date format: Show Year only
DATE_ROTATION = 45                                   # Rotation angle for date labels
DATE_HA = 'right'                                    # Horizontal alignment for date labels

# ----------------------------------------------------------------------------
# Output Configuration
# ----------------------------------------------------------------------------
OUTPUT_DPI = 300                                     # Output image resolution
OUTPUT_PAD_INCHES = 0.02                             # Output image margin


# ============================================================================
# Function Definitions
# ============================================================================

def load_cache(path: Path) -> Dict[str, Any]:
    """Load cache from pkl file."""
    if not path.exists():
        raise FileNotFoundError(f"Cache file does not exist: {path}\nPlease run prepare_cache.py first.")

    with open(path, 'rb') as f:
        return pickle.load(f)


def get_display_name(model_key: str) -> str:
    """Get the display name of the model."""
    if model_key == "Human":
        return "Human"
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def get_plot_order_index(display_name: str) -> int:
    """Get the index of the model in the plotting order."""
    if display_name in PLOT_ORDER:
        return PLOT_ORDER.index(display_name)
    return len(PLOT_ORDER)                           # Put unknown models last


def plot_task_timeseries(
    task_name: str,
    task_stats: Dict[str, pd.DataFrame],
    output_path: Path
):
    """
    Plot the timeseries for a single task.

    Parameters:
        task_name: Task name (spending, labor, credit)
        task_stats: Monthly statistics for each model, with model directory names as keys
        output_path: Path to the output file
    """
    print(f"\nPlotting {task_name} timeseries...")

    # Reset and apply style configuration
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(TIMESERIES_STYLE_CONFIG)

    # Create figure
    fig, ax = plt.subplots(figsize=TIMESERIES_FIGURE_SIZE)

    # Get task configuration
    task_config = TASK_CONFIGS.get(task_name, {})
    ylabel = task_config.get("ylabel", "Value")

    # Collect all data for Y-axis range calculation
    all_values = []

    # Sort models by plotting order (ensuring Human is on the top layer)
    sorted_models = sorted(
        task_stats.keys(),
        key=lambda k: get_plot_order_index(get_display_name(k))
    )

    # Draw timeseries lines for each model
    for model_key in sorted_models:
        # Skip models not in configuration
        if model_key != "Human" and model_key not in MODEL_DISPLAY_NAMES:
            continue

        df = task_stats[model_key]
        if df.empty:
            continue

        display_name = get_display_name(model_key)

        # Get style configuration for this model
        color = MAIN_COLORS.get(display_name, DEFAULT_COLOR)
        linestyle = LINE_STYLES.get(display_name, DEFAULT_LINESTYLE)
        linewidth = LINE_WIDTHS.get(display_name, DEFAULT_LINEWIDTH)
        zorder = get_plot_order_index(display_name) + 1  # zorder starts from 1

        # Draw line
        ax.plot(
            df['date'],
            df['mean'],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=LINE_ALPHA,
            label=display_name,
            zorder=zorder
        )

        all_values.extend(df['mean'].dropna().tolist())

    # Set Y-axis range (set margins at top and bottom)
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_padding = (y_max - y_min) * Y_PADDING_RATIO
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Remove top and right spines
    sns.despine(ax=ax, top=True, right=True)

    # Add Y-axis grid lines
    ax.grid(
        True,
        axis=GRID_AXIS,
        alpha=GRID_ALPHA,
        linestyle=GRID_LINESTYLE,
        color=GRID_COLOR,
        linewidth=GRID_LINEWIDTH
    )

    # Set Y-axis label (special adjustment for spending task position)
    if task_name == "spending":
        ax.set_ylabel(ylabel, labelpad=YLABEL_LABELPAD, y=SPENDING_YLABEL_Y)
    else:
        ax.set_ylabel(ylabel, labelpad=YLABEL_LABELPAD)

    # Set X-axis label
    ax.set_xlabel(XLABEL_TEXT, labelpad=XLABEL_LABELPAD)

    # Configure date format
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))

    # Tilt year labels to avoid overlap
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=DATE_ROTATION, ha=DATE_HA)

    # Legend configuration (Commented out, enable as needed)
    # handles, labels = ax.get_legend_handles_labels()
    # order = []
    # for name in reversed(PLOT_ORDER):
    #     if name in labels:
    #         order.append(labels.index(name))
    #
    # ax.legend(
    #     [handles[i] for i in order],
    #     [labels[i] for i in order],
    #     loc='upper left',
    #     bbox_to_anchor=(1.02, 1),
    #     frameon=True,
    #     fancybox=False,
    #     framealpha=0.95,
    #     edgecolor='#cccccc',
    # )

    # Adjust layout
    plt.tight_layout()

    # Save image
    plt.savefig(output_path, bbox_inches='tight', pad_inches=OUTPUT_PAD_INCHES, dpi=OUTPUT_DPI)
    plt.close()

    print(f"  ✓ Saved to: {output_path}")


def main():
    print("=" * 60)
    print("  Timeseries Plotting")
    print("=" * 60)

    # Load cache
    print(f"Loading cache: {CACHE_MONTHLY_STATS}")
    all_stats = load_cache(CACHE_MONTHLY_STATS)

    # Output directory
    output_dir = PIC_DIR / "timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot for each task
    for task_name in ["spending", "labor", "credit"]:
        if task_name not in all_stats:
            print(f"\n[{task_name}] No data, skipping")
            continue

        task_stats = all_stats[task_name]
        output_path = output_dir / f"{task_name}.pdf"

        plot_task_timeseries(task_name, task_stats, output_path)

    print("\n" + "=" * 60)
    print("  Completed!")
    print("=" * 60)
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
