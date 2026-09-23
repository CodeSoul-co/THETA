"""Figure 05: 半小提琴 雨云图.

Run from the repository root: python -m figures.figure05.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
from matplotlib.patches import Patch
from vizlib.common import nums, subset, clean, label_axes, boxes, density


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    ax = c.ax()
    clean(ax, True)
    label_axes(ax, s)
    ax.spines['right'].set_color('#bbb')
    ax.spines['top'].set_color('#bbb')
    grid = np.linspace(s['ylim'][0], s['ylim'][1], 500)
    for j, (g, cc) in enumerate(zip(s['groups'], s['colors'])):
        rr = subset(rows, group=g)
        v = nums(rr, 'value')
        d = density(v, grid, s['bandwidth'])
        d = d / d.max() * 0.32
        active = (grid >= v.min()) & (grid <= v.max())
        ax.fill_betweenx(grid, j + 0.22, j + 0.22 + d, where=active, fc=cc, ec='#444', lw=1.4)
        jitter = nums(rr, 'jitter') * 0.2 if 'jitter' in rr[0] else c.rng.uniform(-0.1, 0.1, len(rr))
        ax.scatter(j - 0.04 + jitter, v, s=6, c=cc, alpha=0.7, lw=0)
        boxes(ax, [v], [j + 0.22], ['white'], width=0.074, showfliers=False, lw=1.2)
    ax.set_xlim(-0.65, 3.6)
    ax.set_xticks(range(len(s['groups'])), s['groups'], fontweight='bold', fontstyle='italic')
    ax.set_ylabel(s['ylabel'], fontweight='bold')
    ax.set_yticks(range(0, 31, 5))
    ax.tick_params(length=2)
    ax.legend([Patch(fc=cc, ec='#333') for cc in s['colors']], s['groups'], loc='center left', bbox_to_anchor=(1.02, 0.48), frameon=False, prop={'weight': 'bold', 'style': 'italic', 'size': 17}, handlelength=0.8, handleheight=1, labelspacing=0.15, handletextpad=0.4)


if __name__ == "__main__":
    from render import main
    main(default_figure=5)
