"""
Shared plot styling helpers used by every full-day time-series plot
(prediction plots and ionospheric-delay plots alike), so changing a
single constant in config updates every plot consistently.
"""

from src.configs.config import (
    F_L1, F_L2, F_L5,
    PLOT_LABEL_FONTSIZE, PLOT_LABEL_FONTWEIGHT,
    PLOT_TICK_FONTSIZE, PLOT_TICK_FONTWEIGHT,
    PLOT_LEGEND_FONTSIZE, PLOT_LEGEND_FONTWEIGHT,
    PLOT_GRID_ALPHA, PLOT_GRID_LINEWIDTH,
    PLOT_LEGEND_ANCHOR, PLOT_LEGEND_LOC,
    PLOT_LEGEND_ANCHOR_RIGHT, PLOT_LEGEND_LOC_RIGHT,
    get_time_ticks,
)

_TICK_POS, _TICK_LABELS = get_time_ticks()
_FREQ_NAMES = {F_L1: "L1", F_L2: "L2", F_L5: "L5"}


def _freq_name(freq) -> str:
    return _FREQ_NAMES.get(freq, str(freq))


def _day_or_date(target_day: int, date_label: str = None) -> str:
    """Prefer a real calendar date label; fall back to 'dayN'."""
    return date_label if date_label else f"day{target_day}"


def _make_filename(*parts: object) -> str:
    """Build a `part1_part2..._partN.png` filename, skipping empty parts."""
    clean = [str(p) for p in parts if p not in (None, "")]
    return "_".join(clean) + ".png"


def _style_timeseries_ax(ax, ylabel: str):
    """Shared axis styling for every full-day time-series plot — reads
    fonts/grid settings from config so every plot stays visually
    consistent."""
    ax.set_xlabel("Time of Day (UTC)", fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE, fontweight=PLOT_LABEL_FONTWEIGHT)
    ax.set_xticks(_TICK_POS)
    ax.set_xticklabels(_TICK_LABELS, fontsize=PLOT_TICK_FONTSIZE, fontweight=PLOT_TICK_FONTWEIGHT)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_FONTSIZE)
    for t in ax.get_yticklabels():
        t.set_fontweight(PLOT_TICK_FONTWEIGHT)
    ax.grid(True, alpha=PLOT_GRID_ALPHA, linewidth=PLOT_GRID_LINEWIDTH)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _legend_above(ax, ncol: int = 2):
    """Legend on a fixed anchor above the axes (config-driven position) so
    it never overlaps the data, regardless of where the curve peaks."""
    ax.legend(
        loc=PLOT_LEGEND_LOC,
        bbox_to_anchor=PLOT_LEGEND_ANCHOR,
        ncol=ncol,
        frameon=False,
        prop={"weight": PLOT_LEGEND_FONTWEIGHT, "size": PLOT_LEGEND_FONTSIZE},
        handlelength=2.2,
        columnspacing=1.5,
    )


def _legend_right(ax, ncol: int = 4):
    """Legend anchored below the x-axis label, spread horizontally — for
    plots where a top legend would compete with the data (e.g. the
    4-entry L1+L5 combined delay plot)."""
    ax.legend(
        loc=PLOT_LEGEND_LOC_RIGHT,
        bbox_to_anchor=PLOT_LEGEND_ANCHOR_RIGHT,
        ncol=ncol,
        frameon=False,
        prop={"weight": PLOT_LEGEND_FONTWEIGHT, "size": PLOT_LEGEND_FONTSIZE},
        handlelength=2.2,
        columnspacing=1.5,
    )