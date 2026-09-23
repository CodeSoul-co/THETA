"""Figure 03: 多队列 PCoA 组合图.

Run from the repository root: python -m figures.figure03.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from vizlib.common import nums, subset, clean, label_axes, boxes


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    ax = c.ax()
    clean(ax, True, '#999')
    label_axes(ax, s)
    ax.xaxis.set_label_position('top')
    ax.set_xticks([])
    ax.set_yticks([])
    for study, color in zip(s['studies'], s['colors']):
        for group, marker in [('control', 'o'), ('IBD', '^')]:
            rr = subset(rows, study=study, group=group)
            if rr:
                ax.scatter(nums(rr, 'x'), nums(rr, 'y'), s=27, c=color, marker=marker, alpha=0.72, lw=0)
    if c.annotations:
        ax.text(0.02, 0.98, s['annotation'], va='top', transform=ax.transAxes, fontsize=31)
    ax.legend([Line2D([], [], marker=m, c='black', ls='', ms=5) for m in ['o', '^']], ['control', 'IBD'], title='Group', loc='lower left', frameon=False, fontsize=14, title_fontsize=18, labelspacing=1)
    studies = s['studies']
    study_colors = s['colors']
    groups = ['control', 'IBD']
    group_colors = ['#b84c37', '#6487c9']
    r1 = c.ax([714, 78, 137, 614])
    r2 = c.ax([875, 78, 73, 614])
    b1 = c.ax([73, 715, 611, 138])
    b2 = c.ax([73, 882, 611, 90])
    for aa in [r1, r2, b1, b2]:
        clean(aa, True, '#999')
        aa.set_xticks([])
        aa.set_yticks([])
    for aa in [r1, r2]:
        aa.set_ylim(s['ylim'])
    for aa in [b1, b2]:
        aa.set_xlim(s['xlim'])
    boxes(r1, [nums(subset(rows, study=g), 'y') for g in studies], range(6), study_colors, width=0.7, lw=1.3)
    boxes(r2, [nums(subset(rows, group=g), 'y') for g in groups], range(2), group_colors, width=0.7, lw=1.3)
    boxes(b1, [nums(subset(rows, study=g), 'x') for g in studies], range(6), study_colors, vert=False, width=0.7, lw=1.3)
    boxes(b2, [nums(subset(rows, group=g), 'x') for g in groups], range(2), group_colors, vert=False, width=0.7, lw=1.3)
    r1.set_xlim(-0.6, 5.6)
    r2.set_xlim(-0.55, 1.55)
    b1.set_ylim(-0.6, 5.6)
    b2.set_ylim(-0.55, 1.55)
    r1.set_title('Study', pad=11)
    r2.set_title('Group', pad=11)
    r2.yaxis.tick_right()
    r2.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    r2.tick_params(labelsize=13, length=0)
    b1.set_ylabel('Study')
    b2.set_ylabel('Group')
    b2.set_xticks([-0.25, 0, 0.25, 0.5])
    b2.tick_params(labelsize=13, length=3)
    legend = c.ax([706, 710, 315, 330])
    legend.axis('off')
    legend.legend([Patch(fc=cc, ec='#333') for cc in study_colors], studies, title='Study', loc='upper left', frameon=False, fontsize=14, title_fontsize=18, handlelength=1.2, labelspacing=0.8, borderaxespad=0)


if __name__ == "__main__":
    from render import main
    main(default_figure=3)
