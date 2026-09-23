"""Figure 18: 进化年龄分面箱线图.

Run from the repository root: python -m figures.figure18.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
from vizlib.common import subset, clean


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    for p, title in enumerate(s['panels']):
        x = [106, 399, 673, 930][p]
        w = [141, 141, 141, 141][p]
        ax = c.ax([x, 149, w, 359])
        clean(ax)
        ax.set_ylim(s['ylims'][p])
        ax.set_xlim(-0.65, 1.65)
        gradient = np.linspace(0, 1, 100)[:, None]
        cm = mcolors.LinearSegmentedColormap.from_list('wash', [s['gradient_colors'][p], 'white'])
        ax.pcolormesh([-0.65, 1.65], np.linspace(*s['ylims'][p], 101), gradient, cmap=cm, shading='flat', zorder=0, rasterized=False)
        rr = subset(rows, panel=title)
        for j, r in enumerate(rr):
            d = {key: float(r[src]) for key, src in [('whislo', 'low'), ('q1', 'q1'), ('med', 'median'), ('q3', 'q3'), ('whishi', 'high')]}
            d['fliers'] = []
            if not d['whislo'] <= d['q1'] <= d['med'] <= d['q3'] <= d['whishi']:
                raise ValueError('Invalid five-number summary')
            ax.bxp([d], positions=[j], widths=0.61, patch_artist=True, manage_ticks=False, showfliers=False, boxprops={'facecolor': '#999b98' if j == 0 else '#7086c6' if p == 0 else s['colors'][p], 'edgecolor': '#333', 'linewidth': 1.7}, medianprops={'color': '#444', 'linewidth': 1.3}, whiskerprops={'color': '#555', 'linewidth': 1.2}, capprops={'linewidth': 0})
        ax.set_xticks([0, 1], [r['group'] for r in rr], rotation=47, ha='right', rotation_mode='anchor', fontsize=22, fontstyle='italic')
        ax.set_ylabel(s['ylabel'], fontsize=22, labelpad=10)
        ax.tick_params(length=5, labelsize=20)
        hx = [58, 358, 627, 895][p]
        header = c.ax([hx, 28, 212, 56])
        header.axis('off')
        header.add_patch(Rectangle((0, 0), 1, 1, fc=s['colors'][p], ec='none'))
        header.text(0.5, 0.5, title, ha='center', va='center', color='white', fontsize=14)
        if c.annotations:
            c.text(x + w / 2, 102, s['pvalues'][p], ha='center', va='top', fontsize=21)
            ax.plot([0.05, 0.95], [1.027, 1.027], transform=ax.transAxes, c='#888', lw=1.2, clip_on=False)


if __name__ == "__main__":
    from render import main
    main(default_figure=18)
