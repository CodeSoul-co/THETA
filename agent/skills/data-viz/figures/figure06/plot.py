"""Figure 06: 断轴分组柱状图.

Run from the repository root: python -m figures.figure06.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

from matplotlib.lines import Line2D
from vizlib.common import nums, subset, clean, bracket, summary


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    bottom = c.ax([231, 444, 838, 269])
    top = c.ax([231, 130, 838, 271])
    bottom.set_zorder(3)
    top.set_zorder(2)
    for ax in [bottom, top]:
        clean(ax)
        ax.set_xlim(-0.75, 6.3)
    bottom.set_ylim(s['bottom_ylim'])
    top.set_ylim(s['top_ylim'])
    top.spines['bottom'].set_visible(False)
    top.set_xticks([])
    top.set_yticks([10000, 20000, 30000, 40000])
    bottom.set_yticks([0, 0.5, 1, 1.5, 2])
    for i, category in enumerate(s['categories']):
        for j, (g, cc) in enumerate(zip(s['groups'], s['colors'])):
            x = i * 3.7 + j
            rr = subset(rows, category=category, group=g)
            mean, error = summary(rr, c.summary)
            v = nums(rr, 'value')
            jitter = c.rng.uniform(-0.28, 0.28, len(v))
            for ax in [bottom, top]:
                if ax is top and i == 0:
                    continue
                ax.bar(x, mean, 0.65, fc=cc, ec='#444', lw=1.8)
                ax.errorbar(x, mean, yerr=error, fmt='none', ecolor='#444', capsize=16, elinewidth=1.8, capthick=1.8)
                ax.scatter(x + jitter, v, s=65, c=cc, ec='#555', lw=1.2, zorder=3)
    bottom.set_xticks([1, 4.7], s['categories'], fontweight='bold')
    bottom.tick_params(axis='x', pad=20, length=15)
    for ax, y in [(bottom, 1), (top, 0)]:
        ax.plot([-0.016, 0.016], [y + 0.05, y - 0.05], transform=ax.transAxes, c='#555', lw=1.7, clip_on=False)
    c.text(70, 405, s['ylabel'], rotation=90, ha='center', va='center', fontsize=32, fontweight='bold')
    c.fig.legend([Line2D([], [], marker='o', ls='', mec='#555', mfc=cc, ms=13) for cc in s['colors']], s['groups'], loc='upper left', bbox_to_anchor=(0.215, 0.99), frameon=False, fontsize=31, handlelength=0.6, handletextpad=0.45, labelspacing=0.35)
    if c.annotations:
        for text, a, b, y in s['annotations']:
            bracket(bottom if y < 3 else top, a, b, y, text, height=0.13 if y < 3 else 2800, lw=2.3, fontsize=34)


if __name__ == "__main__":
    from render import main
    main(default_figure=6)
