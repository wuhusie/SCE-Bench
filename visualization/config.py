"""
Visualization module configuration.

Includes path configurations, color schemes, model mappings, etc.
"""
from pathlib import Path
import matplotlib.colors as mcolors
import numpy as np

# ==========================================
# Path Configurations
# ==========================================
# Module directory
MODULE_DIR = Path(__file__).parent

# Project root directory
PROJECT_ROOT = MODULE_DIR.parent.parent

# Data input directory
DATA_DIR = PROJECT_ROOT / "result_cleaned" / "exp1" / "N1"

# Cache directory
CACHE_DIR = MODULE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Cache file paths
CACHE_MONTHLY_STATS = CACHE_DIR / "monthly_stats.pkl"
CACHE_RAW_SAMPLES = CACHE_DIR / "raw_samples.pkl"
CACHE_ERRORS = CACHE_DIR / "errors.pkl"

# Image output directory
PIC_DIR = MODULE_DIR / "pic"
PIC_DIR.mkdir(exist_ok=True)

# ==========================================
# Model configuration
# ==========================================
# Model directory name -> Display name
MODEL_DISPLAY_NAMES = {
    "gpt-3.5-turbo": "GPT3.5",
    "gpt-4o-mini": "GPT4om",
    "gpt-5-mini-medium": "GPT5m-nt",
    "gpt-5-mini-minimal": "GPT5m-t",
    "gemini-3-flash-preview-thinking": "GEM3f-t",
    "gemini-3-flash-preview-nothinking": "GEM3f-nt",
    "Qwen3-30B-A3B-Instruct-2507-FP8": "QWE3-nt",
    "Qwen3-30B-Thinking": "QWE3-t",
}

# ----------------------------------------------------------------------------
# Grouping configurations
# ----------------------------------------------------------------------------
MODEL_GROUPS = [
    ("GPT-5", ["gpt-5-mini-medium", "gpt-5-mini-minimal"]),       # GPT-5 Thinking vs. No-Thinking
    ("Gemini", ["gemini-3-flash-preview-nothinking", "gemini-3-flash-preview-thinking"]),  # Gemini Thinking vs. No-Thinking
    ("GPT-3.5_4o-mini", ["gpt-3.5-turbo", "gpt-4o-mini"]),        # GPT-3.5 and GPT-4o-mini
    ("Qwen", ["Qwen3-30B-A3B-Instruct-2507-FP8", "Qwen3-30B-Thinking"]),  # Qwen Thinking vs. No-Thinking
]

# ==========================================
# Task Configurations
# ==========================================
TASK_CONFIGS = {
    "spending": {
        "llm_col": "llm_response",
        "human_col": "Q26v2part2",
        "file_pattern": "spending_*_withHumanData.csv",
        "ylabel": "Exp. Income Growth(%)",
        "cleaning_method": "iqr",
    },
    "labor": {
        "llm_col": "llm_response",
        "human_col": "oo2c3",
        "file_pattern": "labor_*_withHumanData.csv",
        "ylabel": "Acceptance Prob.(%)",
        "cleaning_method": "range",
        "valid_range": (0, 100),
    },
    "credit": {
        "llm_col": "llm_response",
        "human_col": "N17b_2",
        "file_pattern": "credit_*_withHumanData.csv",
        "ylabel": "Application Prob.(%)",
        "cleaning_method": "range",
        "valid_range": (0, 100),
    }
}

# ==========================================
# Color Configurations (Replicating example style)
# ==========================================
# Main color scheme
MAIN_COLORS = {
    "Human": "#2C2C2C",              # Charcoal black - Authoritative baseline
    "GPT3.5": "#1F77B4",             # Standard academic blue
    "GPT4om": "#6BAED6",             # Light blue
    "GPT5m-t": "#7B4B94",            # Dark purple
    "GPT5m-nt": "#C9B3DB",           # Light purple
    "GEM3f-t": "#E76F51",            # Coral orange
    "GEM3f-nt": "#F4A582",           # Light coral orange
    "QWE3-nt": "#7ECAC0",            # Light turquoise
    "QWE3-t": "#2A9D8F",             # Turquoise
}

# Line style configurations
LINE_STYLES = {
    "Human": "-",
    "GPT3.5": "-",
    "GPT5m-t": "-",
    "GPT5m-nt": "--",
    "GEM3f-t": "-",
    "GEM3f-nt": "--",
    "QWE3-nt": "--",
    "QWE3-t": "-",
}

# Line width configurations (Adapted for small sizes, uniform width, Human slightly thicker)
LINE_WIDTHS = {
    "Human": 0.8,            # Human line is slightly thicker
    "GPT3.5": 0.5,
    "GPT4om": 0.5,
    "GPT5m-t": 0.5,
    "GPT5m-nt": 0.5,
    "GEM3f-t": 0.5,
    "GEM3f-nt": 0.5,
    "QWE3-nt": 0.5,
    "QWE3-t": 0.5,
}

# Plotting order (From bottom to top)
PLOT_ORDER = [
    "GPT5m-nt",
    "GEM3f-nt",
    "QWE3-t",
    "GPT3.5",
    "GPT5m-t",
    "GEM3f-t",
    "QWE3-nt",
    "Human",  # Human is on the top layer
]

# ==========================================
# LaTeX Size Configurations
# ==========================================
LATEX_TEXT_WIDTH = 506.295    # \textwidth: Double-column full width (pt)
LATEX_COL_WIDTH = 241.14749   # \linewidth: Single-column width (pt)
INCHES_PER_PT = 1 / 72.27


def set_size(width_pt, fraction=1.0, ratio='golden'):
    """
    Calculate figsize suitable for LaTeX.

    Parameters:
        width_pt: Target width in LaTeX (pt)
        fraction: Fraction of the target width the image should occupy
        ratio: 'golden' (golden ratio) or float (custom aspect ratio height/width)
    Returns:
        (width_in, height_in)
    """
    fig_width_pt = width_pt * fraction
    fig_width_in = fig_width_pt * INCHES_PER_PT

    if ratio == 'golden':
        golden_ratio = (5**.5 - 1) / 2
        fig_height_in = fig_width_in * golden_ratio
    else:
        fig_height_in = fig_width_in * ratio

    return (fig_width_in, fig_height_in)


# ==========================================
# Timeseries Plot Style Configuration (1 row 3 columns, spanning double columns)
# ==========================================
TIMESERIES_STYLE_CONFIG = {
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'grid.linewidth': 0.3,
}

# Timeseries plot size: 1 row 3 columns, each subplot occupying 1/3 width
TIMESERIES_FIGURE_SIZE = set_size(LATEX_TEXT_WIDTH, fraction=1/3, ratio=0.75)

# ==========================================
# KDE/MAE Distribution Plot Style Configuration
# ==========================================
KDE_STYLE_CONFIG = {
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 28,
    'axes.labelsize': 34,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 30,
    'lines.linewidth': 3.0,
    'axes.linewidth': 2.0,
}

# KDE plot size
KDE_FIGURE_SIZE = (8, 6)

# JS Divergence annotation font size
JS_ANNOTATION_FONTSIZE = 30

# ==========================================
# KDE Distribution Plot Color Scheme
# ==========================================
KDE_COLORS = {
    "Human_fill": "#BEB8DA",         # Light purple fill
    "Human_edge": "#7B6FA3",         # Dark purple edge
    "LLM_fill": "#FCAD9F",           # Light coral fill
    "LLM_edge": "#E85C4A",           # Dark coral red edge
}

# ==========================================
# Compatibility with old configurations (deprecated)
# ==========================================
STYLE_CONFIG = TIMESERIES_STYLE_CONFIG
FIGURE_SIZE = TIMESERIES_FIGURE_SIZE

# ==========================================
# Utility Functions
# ==========================================
def lighten_color(hex_color: str, factor: float = 0.4) -> str:
    """
    Lighten a color (mix with white).

    Parameters:
        hex_color: Hex color value
        factor: Lightening factor, from 0.0 (original) to 1.0 (pure white)

    Returns:
        Lightened hex color value
    """
    c = mcolors.to_rgb(hex_color)
    new_rgb = (1 - factor) * np.array(c) + factor * np.array([1, 1, 1])
    return mcolors.to_hex(new_rgb)


def darken_color(hex_color: str, factor: float = 0.3) -> str:
    """
    Darken a color (mix with black).

    Parameters:
        hex_color: Hex color value
        factor: Darkening factor, from 0.0 (original) to 1.0 (pure black)

    Returns:
        Darkened hex color value
    """
    c = mcolors.to_rgb(hex_color)
    new_rgb = (1 - factor) * np.array(c)
    return mcolors.to_hex(new_rgb)
