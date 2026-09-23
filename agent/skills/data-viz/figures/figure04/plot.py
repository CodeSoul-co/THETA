"""Figure 04: 时间序列与 iAUC 插图.

Run from the repository root: python -m figures.figure04.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
from matplotlib.patches import Patch
from vizlib.common import nums, subset, clean, label_axes


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    ax = c.ax()
    clean(ax)
    label_axes(ax, s)
    ax.axvspan(s['split_day'], 40, color='#e7e7e6', zorder=0)
    ax.axvline(s['split_day'], ls='--', color='black', lw=2.8)
    for g, color in zip(s['groups'], s['colors']):
        rr = subset(rows, group=g)
        order = np.argsort(nums(rr, 'day'))
        x = nums(rr, 'day')[order]
        y = nums(rr, 'mean')[order]
        e = nums(rr, 'error')[order]
        ax.errorbar(x, y, yerr=e, color=color, lw=8, marker='o', ms=23, mec='black', mew=2.6, ecolor='black', elinewidth=2.5, capsize=13, capthick=2.6, clip_on=False)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 20, 40, 60])
    c.fig.legend([Patch(fc=cc, ec='black', lw=2) for cc in s['colors']], s['groups'], ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.55, 1.005), fontsize=31, handlelength=1.8, columnspacing=0.5, handletextpad=0.2, labelspacing=0)
    inset_rows = c.load('figure04_insets.csv')
    for k, (phase, box, xlim, ticks) in enumerate([('Exposure', [229, 210, 325, 163], [-200, 800], [-200, 300, 800]), ('Cessation', [625, 210, 330, 163], [0, 400], [0, 200, 400])]):
        aa = c.ax(box)
        clean(aa)
        aa.spines['left'].set_visible(False)
        aa.set_xlim(xlim)
        aa.set_ylim(-0.6, 3.6)
        aa.set_yticks([])
        aa.set_xticks(ticks)
        aa.set_title('iAUC: ' + phase, fontsize=32, pad=7)
        for j, (g, cc) in enumerate(zip(s['groups'], s['colors'])):
            vals = nums(subset(inset_rows, phase=phase, group=g), 'value')
            y = 3 - j
            aa.barh(y, vals.mean(), 0.56, facecolor='white', edgecolor='black', lw=1.5)
            aa.scatter(vals, y + c.rng.uniform(-0.2, 0.2, len(vals)), s=21, c=cc, alpha=0.75, lw=0)
            aa.errorbar(vals.mean(), y, xerr=vals.std(ddof=1) / np.sqrt(len(vals)), fmt='none', ecolor='black', capsize=5)
        aa.axvline(0, c='black', lw=1.6)
        if c.annotations:
            for x, y in [(xlim[1] * 0.72, 2.5), (xlim[1] * 0.85, 0.65)]:
                aa.text(x, y, '****', rotation=90, ha='center', va='center', fontsize=30)
    if c.annotations:
        for text, x, lo, hi in s['annotations']:
            ax.plot([x, x], [lo, hi], c='black', lw=2.3)
            ax.text(x + 0.45, (lo + hi) / 2, text, rotation=90, va='center', fontsize=30)


if __name__ == "__main__":
    from render import main
    main(default_figure=4)
