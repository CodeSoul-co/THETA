"""Figure 19: 相关性 哑铃 气泡 与条形组合图.

Run from the repository root: python -m figures.figure19.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
import matplotlib
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from vizlib.common import nums, clean


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    cells = s['cells']
    rr = [next((r for r in rows if r['cell_type'] == cell)) for cell in cells]
    N = len(rr)
    ys = np.arange(N)
    ax = c.ax([190, 82, 482, 680])
    expr = c.ax([686, 82, 184, 680])
    bar = c.ax([881, 82, 116, 680])
    for aa in [ax, expr, bar]:
        clean(aa, True, '#999')
        aa.set_ylim(N - 0.0, -1)
        aa.set_yticks(ys)
        aa.tick_params(length=0)
    ax.set_xlim(s['xlim'])
    ax.set_yticklabels(cells, fontsize=11.5)
    ax.set_xticks(np.arange(-0.3, 0.41, 0.1))
    ax.tick_params(axis='x', length=3, labelsize=14)
    ax.grid(axis='both', color='#e9e9e9', ls=(0, (2, 7)), lw=0.7, zorder=0)
    wt = nums(rr, 'wt')
    ko = nums(rr, 'ko')
    ax.hlines(ys, wt, ko, color='#999', lw=0.85)
    ax.scatter(wt, ys, c='black', s=22, zorder=3)
    ax.scatter(ko, ys, c=[s['colors'][0] if i < N / 2 else s['colors'][1] for i in ys], s=22, zorder=3)
    ax.set_title('TGFβRII-dependent signature', fontsize=15, pad=13)
    ax.set_xlabel('Correlation between expression and closeness\nto the cell', fontsize=14, labelpad=4)
    c.text(179, 58, 'Cell Type', ha='right', va='top', fontsize=15, fontweight='bold')
    erows = c.load('figure19_expression.csv')
    genes = s['genes']
    cm = mcolors.LinearSegmentedColormap.from_list('expr', ['#9ec7d1', '#e3df4b', '#dbc12a', '#d33e15'])
    norm = mcolors.Normalize(s['vmin'], s['vmax'])
    perc = nums(erows, 'percent')
    if np.any((perc < 0) | (perc > 100)):
        raise ValueError('Expression percentages must be 0–100')
    expr.scatter([genes.index(r['gene']) for r in erows], [cells.index(r['cell_type']) for r in erows], s=perc / 70 * s['max_dot_area'], c=nums(erows, 'expression'), cmap=cm, norm=norm, edgecolors='#777', lw=0.6, zorder=3)
    expr.set_xlim(-0.8, len(genes) - 0.2)
    expr.set_yticklabels([])
    expr.set_xticks(range(len(genes)), genes, rotation=80, fontstyle='italic', fontsize=14)
    expr.set_title('TGFβ    present.', fontsize=15, pad=13)
    c.text(782, 11, 'Gene Expression', fontsize=15, ha='center', va='top')
    expr.plot([0, 1], [1.065, 1.065], transform=expr.transAxes, c='#888', lw=0.8, clip_on=False)
    for i, r in enumerate(rr):
        v = float(r['ks'])
        if not 0 <= v <= 1:
            raise ValueError('KS statistic must be in [0,1]')
        bar.barh(i, v, 0.78, color=s['relation_colors'][r['relation']], edgecolor='none')
    bar.set_yticklabels([])
    bar.set_xlim(0, max(0.45, nums(rr, 'ks').max() * 1.03))
    bar.set_xticks([0, 0.25], ['0', '.25'])
    bar.tick_params(axis='x', labelsize=14, length=3)
    bar.grid(axis='x', c='#eee', ls=(0, (2, 5)), lw=0.7)
    bar.set_title('proximity\ndifferences', fontsize=14, linespacing=0.8, pad=13)
    bar.set_xlabel('KS\nWT vs\nTGFβRII KO', fontsize=14, labelpad=3, linespacing=1)
    c.text(72, 858, 'P14 CD8 T Cell', fontsize=15, va='top')
    for k, (name, cc) in enumerate([('WT', 'black'), ('TGFβRII KO', s['colors'][1])]):
        c.fig.add_artist(Line2D([78 / s['width']], [1 - (895 + k * 27) / s['height']], marker='o', ms=5, c=cc, ls=''))
        c.text(97, 895 + k * 27, name, fontsize=15, va='center')
    c.text(327, 858, 'Mean expression\nin group', fontsize=14, ha='center', va='top', linespacing=1)
    cb = c.ax([259, 911, 132, 14])
    colorbar = c.fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cm), cax=cb, orientation='horizontal', ticks=[0, 2])
    colorbar.solids.set_rasterized(False)
    cb.tick_params(labelsize=14, length=2, pad=1)
    cb.spines[:].set_color('#888')
    leg = c.ax([456, 858, 166, 98])
    leg.axis('off')
    leg.text(0.5, 1, 'Fraction of cells\nin group (%)', ha='center', va='top', fontsize=14, linespacing=1)
    for x, p in zip([0.2, 0.43, 0.67], [10, 40, 70]):
        leg.scatter(x, 0.4, s=p / 70 * s['max_dot_area'], fc='#ddd', ec='#999', lw=0.6)
        leg.text(x, 0.08, str(p), ha='center', fontsize=14)
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    c.text(652, 858, 'TGFβRII KO cells are', fontsize=15, va='top')
    leg = c.ax([659, 887, 240, 64])
    leg.axis('off')
    for i, (name, cc) in enumerate(s['relation_colors'].items()):
        yy = 0.77 - i * 0.31
        leg.add_patch(Rectangle((0, yy - 0.1), 0.06, 0.21, fc=cc, ec='none'))
        leg.text(0.09, yy, name, fontsize=14, va='center')


if __name__ == "__main__":
    from render import main
    main(default_figure=19)
