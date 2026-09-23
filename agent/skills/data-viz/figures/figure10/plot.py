"""Figure 10: 环形系统发育树与分类注释.

Run from the repository root: python -m figures.figure10.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Wedge
from matplotlib.collections import LineCollection
from vizlib.common import tree_layout


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    n = s['id']
    leafrows = c.load(s['leaf_file'])
    nodes, children, leaves, theta, dist = tree_layout(rows, leafrows, s['start_angle'], s['gap_angle'])
    box = [11, 11, 764, 764]
    ax = c.ax(box)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.025, 1.025)
    ax.set_ylim(-1.025, 1.025)
    cmap = dict(zip(s['phyla'], s['colors']))
    maxd = max(dist.values())
    rad = {node: 0.755 * d / maxd for node, d in dist.items()}
    rad['root'] = 0
    for node in rad:
        rad[node] = 0.09 + rad[node] if node != 'root' else 0
    segs = []
    cs = []

    def xy(r, t):
        return [r * np.cos(t), r * np.sin(t)]
    for node, r in nodes.items():
        p = r['parent']
        color = cmap[r['phylum']]
        segs.append([xy(rad[p], theta[p]), xy(rad[node], theta[node])])
        cs.append(color)
    ax.add_collection(LineCollection(segs, colors=cs, linewidths=1.5, zorder=2))
    step = (360 - s['gap_angle']) / len(leaves)
    heat = plt.get_cmap('Reds')
    for r in leaves:
        t = np.rad2deg(theta[r['node']])
        cc = cmap[r['phylum']]
        rings = [(0.845, 0.89, dict(zip(['WGS', 'MAG', 'SAG'], s['ring_colors']))[r['type']]), (0.907, 0.948, cc), (0.972, 1.015, ['#f4f8f5', '#b3d8b6', '#60a972'][int(r['presence'])])]
        for lo, hi, color in rings:
            ax.add_patch(Wedge((0, 0), hi, t - step * 0.47, t + step * 0.47, width=hi - lo, fc=color, ec='none', lw=0.55))
    ax.text(0.88, -0.035, 'Type', rotation=90, fontsize=9, va='top')
    ax.text(0.94, -0.035, 'Phylum', rotation=90, fontsize=9, va='top')
    ax.text(1, -0.035, 'Presence', rotation=90, fontsize=9, va='top')
    leg = c.ax([877, 135, 245, 570])
    leg.axis('off')

    def entry(y, text, color):
        leg.add_patch(Rectangle((0, y - 0.017), 0.078, 0.035, fc=color, ec='#777', lw=0.5))
        leg.text(0.12, y, text, va='center', fontsize=10.5)
    leg.text(0, 1, 'Genome Type', fontsize=14)
    for i, (name, cc) in enumerate(zip(['WGS', 'MAG', 'SAG'], s['ring_colors'])):
        entry(0.957 - i * 0.038, name, cc)
    leg.text(0, 0.766, 'Occurrence in samples', fontsize=14)
    for i, (name, cc) in enumerate(zip(['Present in ≥10 photic samples', 'Present in at least 1 photic sample', 'Absent from photic samples'], ['#2c711a', '#99c58e', 'white'])):
        entry(0.726 - i * 0.038, name, cc)
    leg.text(0, 0.531, 'Phylum (Top 20)', fontsize=14)
    for i, (name, cc) in enumerate(zip(s['phyla'][:10], s['colors'][:10])):
        entry(0.49 - i * 0.038, name, cc)


if __name__ == "__main__":
    from render import main
    main(default_figure=10)
