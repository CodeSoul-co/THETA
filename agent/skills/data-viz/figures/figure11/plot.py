"""Figure 11: 环形系统发育树与热图.

Run from the repository root: python -m figures.figure11.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.patches import Wedge
from matplotlib.collections import LineCollection
from vizlib.common import col, tree_layout


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    n = s['id']
    leafrows = c.load(s['leaf_file'])
    nodes, children, leaves, theta, dist = tree_layout(rows, leafrows, s['start_angle'], s['gap_angle'])
    box = [29, 45, 758, 758]
    ax = c.ax(box)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.025, 1.025)
    ax.set_ylim(-1.025, 1.025)
    cmap = dict(zip(s['phyla'], s['colors']))
    maxd = max(dist.values())
    rad = {node: 0.755 * d / maxd for node, d in dist.items()}
    rad['root'] = 0
    for r in leaves:
        rad[r['node']] = 0.735
    for name, lo, hi in s['genus_labels']:
        color = '#cce7ed' if name == 'Bacteroides' else '#c4e5db' if name in ['Clostridium', 'Enterococcus'] else '#eae7ed'
        ax.add_patch(Wedge((0, 0), 0.8, lo, hi, width=0.48, fc=col(color, 0.45), ec='none', zorder=0))
    segs = []
    cs = []

    def xy(r, t):
        return [r * np.cos(t), r * np.sin(t)]
    for node, r in nodes.items():
        p = r['parent']
        color = cmap[r['phylum']]
        tt = np.linspace(theta[p], theta[node], max(3, int(abs(theta[p] - theta[node]) * 80)))
        points = np.column_stack([rad[p] * np.cos(tt), rad[p] * np.sin(tt)])
        segs.extend(np.stack([points[:-1], points[1:]], axis=1))
        cs.extend(['#8a8a87'] * (len(points) - 1))
        segs.append([xy(rad[p], theta[node]), xy(rad[node], theta[node])])
        cs.append(color if not children[node] else '#858583')
    ax.add_collection(LineCollection(segs, colors=cs, linewidths=0.85, zorder=2))
    step = (360 - s['gap_angle']) / len(leaves)
    heat = plt.get_cmap('Reds')
    for r in leaves:
        t = np.rad2deg(theta[r['node']])
        cc = cmap[r['phylum']]
        frac = float(r['fraction'])
        bsh = int(r['bsh'])
        if not 0 <= frac <= 1 or bsh not in [0, 1]:
            raise ValueError('Tree annotation fraction must be in [0,1]; bsh must be 0 or 1')
        rings = [(0.861, 0.933, heat(np.clip(frac / 0.6, 0, 1))), (0.933, 1.006, '#82bdcf' if bsh else '#f0f6f2')]
        for lo, hi, color in rings:
            ax.add_patch(Wedge((0, 0), hi, t - step * 0.47, t + step * 0.47, width=hi - lo, fc=color, ec='#888', lw=0.55))
    for name, lo, hi in s['genus_labels']:
        t = (lo + hi) / 2
        xy0 = xy(0.79, np.deg2rad(t))
        rotation = t - 90
        if rotation < -90:
            rotation += 180
        ax.text(*xy0, name, rotation=rotation, ha='center', va='center', fontsize=13, fontweight='bold' if name in ['Bacteroides', 'Clostridium', 'Enterococcus'] else 'normal')
    c.text(808, 47, 'Fraction of total\nBBAAs detected', fontsize=17, va='top')
    cb = c.ax([814, 124, 33, 183])
    norm = mcolors.Normalize(0, 0.6)
    colorbar = c.fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='Reds'), cax=cb, ticks=[0, 0.2, 0.4, 0.6])
    colorbar.solids.set_rasterized(False)
    cb.tick_params(length=0)
    cb.set_yticklabels(['0', '0.2', '0.4', '0.6'])
    cb.spines[:].set_visible(False)
    leg = c.ax([810, 352, 280, 402])
    leg.axis('off')
    leg.text(0, 1, 'Presence of $\\mathit{bsh}$ gene', fontsize=17)
    for i, (label, cc) in enumerate([('Absent', '#f0f6f2'), ('Present', '#82bdcf')]):
        leg.add_patch(Rectangle((0, 0.91 - i * 0.09), 0.1, 0.073, fc=cc, ec='none'))
        leg.text(0.15, 0.944 - i * 0.09, label, va='center', fontsize=17)
    leg.text(0, 0.7, 'Phylum', fontsize=18)
    for i, (name, cc) in enumerate(zip(s['phyla'], s['colors'])):
        yy = 0.63 - i * 0.075
        leg.plot([0.01, 0.13], [yy, yy], c=cc, lw=1)
        leg.plot(0.07, yy, 'o', mfc=cc, mec=cc, ms=10)
        leg.text(0.17, yy, name, fontsize=16, va='center')
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)


if __name__ == "__main__":
    from render import main
    main(default_figure=11)
