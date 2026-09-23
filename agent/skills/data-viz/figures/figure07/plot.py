"""Figure 07: 基因表达分组柱状图.

Run from the repository root: python -m figures.figure07.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

from matplotlib.patches import Patch
from vizlib.common import grouped_bars


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    ax, loc = grouped_bars(c, rows, 0.72, 3.85)
    for tick in ax.get_xticklabels():
        tick.set_fontstyle('italic')
    ax.tick_params(axis='x', length=0)
    c.fig.legend([Patch(fc=cc, ec='#555', lw=1.5) for cc in s['colors']], s['groups'], title='Day15', loc='upper left', bbox_to_anchor=(0.768, 0.998), frameon=False, fontsize=19, title_fontsize=22, handlelength=1.2, labelspacing=0.45, handletextpad=0.35)
    if c.annotations:
        for i, pair in enumerate(s['pvalues']):
            for j, label in enumerate(pair):
                ax.text(loc[i, j + 1][0], s['pvalue_y'][i][j], label, rotation=90, ha='center', va='bottom', fontsize=20)


if __name__ == "__main__":
    from render import main
    main(default_figure=7)
