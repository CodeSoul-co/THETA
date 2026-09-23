"""Figure 09: 双轴嵌套柱状图.

Run from the repository root: python -m figures.figure09.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
from matplotlib.patches import Patch
from vizlib.common import nums, clean


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    ax = c.ax()
    ax2 = ax.twinx()
    clean(ax, True)
    clean(ax2, True)
    ax.set_ylim(s['left_ylim'])
    ax2.set_ylim(s['right_ylim'])
    ax.set_ylabel(s['ylabel'])
    ax2.set_ylabel(s['right_ylabel'], rotation=-90, labelpad=38)
    x = np.arange(len(rows))
    v = nums(rows, 'indigo')
    b = nums(rows, 'daptomycin')
    ve = nums(rows, 'indigo_error')
    be = nums(rows, 'dap_error')
    ax2.bar(x, b, 0.67, color=s['colors'][1], edgecolor='#555', lw=1.4, zorder=1)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.bar(x, v, 0.43, color=s['colors'][0], edgecolor='#555', lw=1.3, zorder=2)
    for aa, vals, err, width in [(ax, v, ve, 0.43), (ax2, b, be, 0.67)]:
        aa.errorbar(x, vals, yerr=err, fmt='none', ecolor='black', capsize=5, elinewidth=1)
        if c.annotations:
            # These are reference-view display symbols, not measured replicates.
            for xx, m, e in zip(x, vals, err):
                aa.scatter(xx + c.rng.uniform(-width * 0.17, width * 0.17, 3), [m - e, m, m + e], s=17, c='black', lw=0, zorder=4)
    ax.axhline(s['reference_lines'][0], c='#727fb4', lw=1.2)
    ax2.axhline(s['reference_lines'][1], c='#b99353', lw=1.2)
    ax.set_xlim(-0.65, len(rows) - 0.35)
    ax.set_xticks(x, [r['category'] for r in rows], rotation=49, ha='right')
    ax.set_yticks([0, 1, 2, 3])
    ax2.set_yticks([0, 50, 100, 150])
    ax.tick_params(axis='x', length=8)
    c.fig.legend([Patch(fc=cc, ec='#777') for cc in s['colors']], ['Indigoidine', 'Daptomycin'], ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(0.585, 0.962), fontsize=20, handlelength=0.7, columnspacing=2)
    if c.annotations:
        ax.text(0.95, 2.78, s['annotation'], fontstyle='italic', fontsize=19)


if __name__ == "__main__":
    from render import main
    main(default_figure=9)
