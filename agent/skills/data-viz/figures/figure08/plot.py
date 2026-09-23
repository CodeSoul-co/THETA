"""Figure 08: PARP1 变体柱状图.

Run from the repository root: python -m figures.figure08.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

from matplotlib.lines import Line2D
from vizlib.common import grouped_bars


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    ax, loc = grouped_bars(c, rows, 0.65, 4.5)
    ax.axhline(1, c='#888', ls=':', lw=1, zorder=0)
    ax.axhline(0, c='black', ls=':', lw=1)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis='x', length=0)
    c.fig.legend([Line2D([], [], marker='o', ls='', mfc=cc, mec='#555', ms=12) for cc in s['colors']], s['groups'], loc='upper center', bbox_to_anchor=(0.556, 0.974), ncol=4, frameon=True, edgecolor='#bbb', fontsize=22, columnspacing=0.8, handlelength=0.7, handletextpad=0.4, borderpad=0.15)
    if c.annotations:
        for i, pair in enumerate(s['pvalues']):
            for j, label in enumerate(pair):
                x, mean, error, v = loc[i, j + 2]
                ax.text(x, max(v.max(), mean + error) + 0.55, label, rotation=90, ha='center', va='bottom', fontsize=19)


if __name__ == "__main__":
    from render import main
    main(default_figure=8)
