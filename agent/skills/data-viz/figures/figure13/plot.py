"""Figure 13: 双对数散点与边缘直方图.

Run from the repository root: python -m figures.figure13.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
from vizlib.common import nums, subset, col, clean, label_axes, boxes


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    ax = c.ax()
    clean(ax)
    label_axes(ax, s)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(axis='y', which='major', ls=(0, (7, 7)), color='#ddd', lw=0.8)
    top = c.ax([179, 0, 643, 124])
    right = c.ax([859, 159, 130, 648])
    top.axis('off')
    right.axis('off')
    binsx = np.geomspace(*s['xlim'], s['bins'])
    binsy = np.geomspace(*s['ylim'], s['bins'])
    top.set_xscale('log')
    right.set_yscale('log')
    top.set_xlim(s['xlim'])
    right.set_ylim(s['ylim'])
    lx = nums(rows, 'genome_bp')
    ly = nums(rows, 'cds')
    if (lx <= 0).any() or (ly <= 0).any():
        raise ValueError('Logarithmic coordinates must be positive')
    fit = np.polyfit(np.log10(lx), np.log10(ly), 1)
    residuals = []
    for g, cc in zip(s['groups'][::-1], s['colors'][::-1]):
        rr = subset(rows, group=g)
        x = nums(rr, 'genome_bp')
        y = nums(rr, 'cds')
        ax.scatter(x, y, s=65 if 'WGS' in g else 51, c=cc, alpha=0.6, lw=0, label=g, zorder=2)
        top.hist(x, bins=binsx, fc=col(cc, 0.25), ec=cc, lw=0.9)
        right.hist(y, bins=binsy, orientation='horizontal', fc=col(cc, 0.25), ec=cc, lw=0.9)
        residuals.append(np.log10(y) - np.polyval(fit, np.log10(x)))
    x = np.geomspace(*s['xlim'], 100)
    ax.plot(x, 10 ** np.polyval(fit, np.log10(x)), c='#c63827', ls=(0, (5, 4)), lw=2, zorder=3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(0.027, 0.974), frameon=True, fancybox=False, fontsize=17, handlelength=0.8, markerscale=1.5, labelspacing=0.4)
    ins = c.ax([454, 636, 381, 149])
    clean(ins, True, '#b8b8b8')
    ins.set_title('Residuals (log)', fontsize=19, pad=10)
    boxes(ins, residuals, [2, 1, 0], s['colors'][::-1], vert=False, width=0.7, lw=1.2)
    ins.set_xlim(-0.75, 1.4)
    ins.set_ylim(-0.6, 2.6)
    ins.set_yticks([])
    ins.set_xticks([-0.5, 0, 0.5, 1])
    ins.tick_params(labelsize=17, length=3)
    if c.annotations:
        for x, y1, y2, label in [(0.7, 1, 2, '**'), (0.98, 0, 1, '****'), (1.23, 0, 2, '****')]:
            ins.plot([x - 0.04, x, x, x - 0.04], [y1, y1, y2, y2], c='#555', lw=1.5)
            ins.text(x + 0.1, (y1 + y2) / 2, label, rotation=90, va='center', fontsize=17)


if __name__ == "__main__":
    from render import main
    main(default_figure=13)
