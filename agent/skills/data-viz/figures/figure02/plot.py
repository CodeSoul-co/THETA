"""Figure 02: PCoA 与边缘箱线图.

Run from the repository root: python -m figures.figure02.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
from matplotlib.patches import Ellipse
from vizlib.common import nums, subset, col, clean, label_axes, boxes


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    ax = c.ax()
    clean(ax, True, '#999')
    label_axes(ax, s)
    ax.axhline(0, c='#999', ls=':', lw=0.8)
    ax.axvline(0, c='#999', ls=':', lw=0.8)
    for j, (g, color) in enumerate(zip(s['groups'], s['colors'])):
        rr = subset(rows, group=g)
        x = nums(rr, 'x')
        y = nums(rr, 'y')
        ax.scatter(x, y, s=200, c=color, alpha=0.95, edgecolors='none', label=g, zorder=3)
        if c.annotations and 'ellipses' in s:
            ex, ey, w, h, angle = s['ellipses'][j]
        else:
            vals, vecs = np.linalg.eigh(np.cov(np.vstack([x, y])))
            order = vals.argsort()[::-1]
            vals = vals[order]
            v = vecs[:, order[0]]
            ex, ey = (x.mean(), y.mean())
            w, h = 2 * np.sqrt(np.maximum(vals, 0) * 5.991)
            angle = np.degrees(np.arctan2(v[1], v[0]))
        ax.add_patch(Ellipse((ex, ey), w, h, angle=angle, fc='none', ec=col(color, 0.75), lw=1.4))
    ax.set_xticks([-0.4, 0, 0.4])
    ax.set_yticks([-0.4, -0.2, 0, 0.2])
    ax.tick_params(length=0)
    ax.text(0.98, 0.97, s['panel_label'], ha='right', va='top', transform=ax.transAxes)
    ax.legend(loc='lower right', frameon=False, fontsize=18, handlelength=0.7, handletextpad=0.2, labelspacing=0, markerfirst=False)
    top = c.ax([128, 26, 710, 173])
    right = c.ax([854, 218, 178, 700])
    ann = c.ax([854, 26, 178, 173])
    for aa in [top, right, ann]:
        clean(aa, True, '#999')
        aa.set_xticks([])
        aa.set_yticks([])
    top.set_xlim(s['xlim'])
    right.set_ylim(s['ylim'])
    top.set_ylim(-0.6, 2.6)
    right.set_xlim(-0.65, 2.65)
    boxes(top, [nums(subset(rows, group=g), 'x') for g in s['groups']], [0, 1, 2], s['colors'], False, 0.72, True, c.rng)
    boxes(right, [nums(subset(rows, group=g), 'y') for g in s['groups']], [0, 1, 2], s['colors'], True, 0.72, True, c.rng)
    if c.annotations:
        ann.text(0.5, 0.5, s['annotation'], ha='center', va='center', fontsize=19, linespacing=1.7, transform=ann.transAxes)


if __name__ == "__main__":
    from render import main
    main(default_figure=2)
