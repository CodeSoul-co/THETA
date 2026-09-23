"""Figure 12: 分组基因气泡矩阵.

Run from the repository root: python -m figures.figure12.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
import matplotlib
from matplotlib import colors as mcolors
from vizlib.common import nums, subset, clean


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    cm = mcolors.LinearSegmentedColormap.from_list('expression', s['colors'])
    norm = mcolors.Normalize(s['vmin'], s['vmax'])
    y = 24
    step = 32.6
    axes = []
    for section in s['sections']:
        genes = section['genes']
        h = len(genes) * step + 4
        ax = c.ax([214, y, 440, h])
        clean(ax, True, '#555')
        axes.append(ax)
        rr = subset(rows, section=section['name'])
        xs = []
        ys = []
        for r in rr:
            xs.append(s['cell_types'].index(r['cell_type']))
            ys.append(genes.index(r['gene']))
        percent = nums(rr, 'percent')
        if np.any((percent < 0) | (percent > 100)):
            raise ValueError('Dot percentages must be 0–100')
        ax.scatter(xs, ys, s=percent / 100 * s['max_dot_area'], c=nums(rr, 'expression'), cmap=cm, norm=norm, ec='#777', lw=0.65)
        ax.set_xlim(-0.55, len(s['cell_types']) - 0.45)
        ax.set_ylim(len(genes) - 0.4, -0.6)
        ax.set_yticks(range(len(genes)), genes)
        ax.set_xticks([])
        ax.tick_params(length=0, pad=5)
        strip = c.ax([44, y, 36, h])
        strip.set_facecolor('#dedede')
        strip.set_xticks([])
        strip.set_yticks([])
        strip.spines[:].set_visible(False)
        strip.text(0.5, 0.5, section['name'], rotation=90, va='center', ha='center', fontsize=16, fontweight='bold')
        y += h + 12
    axes[-1].set_xticks(range(len(s['cell_types'])), s['cell_types'], rotation=90, ha='center', fontsize=15)
    cb = c.ax([688, 275, 31, 214])
    colorbar = c.fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cm), cax=cb, ticks=[1, 2, 3])
    colorbar.solids.set_rasterized(False)
    cb.tick_params(length=0)
    c.text(688, 218, 'Avg\nexpr', fontsize=17, va='top')
    leg = c.ax([684, 530, 134, 250])
    leg.axis('off')
    leg.text(0, 1, 'Pct nuclei\nexpr > 0', fontsize=17, va='top')
    for i, p in enumerate([20, 40, 60, 80, 100]):
        yy = 0.69 - i * 0.13
        leg.scatter(0.14, yy, s=p / 100 * s['max_dot_area'], fc='white', ec='#777', lw=0.8)
        leg.text(0.33, yy, f'{p}%', va='center', fontsize=17)
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)


if __name__ == "__main__":
    from render import main
    main(default_figure=12)
