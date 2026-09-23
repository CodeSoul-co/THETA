"""Figure 17: 区域年代分布 雨云图.

Run from the repository root: python -m figures.figure17.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

from matplotlib.patches import Rectangle
from vizlib.common import nums, subset, col, line_data


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    ax = c.ax([43, 24, 1002, 726])
    ax.set_xlim(s['xlim'])
    ax.set_ylim(-1.28, 4.67)
    ax.spines[:].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#555')
    ax.set_yticks([])
    ax.set_xticks(range(16))
    ax.set_xlabel(s['xlabel'], fontsize=23)
    ax.tick_params(labelsize=22)
    stats = c.load('figure17_boxes.csv')
    rug = c.load('figure17_rug.csv')
    for k, (g, cc) in enumerate(zip(s['groups'], s['colors'])):
        x, y = line_data(subset(rows, region=g), 'age')
        base = 4 - k
        ax.fill_between(x, base, base + y, color=cc, lw=0, zorder=2)
        b = subset(stats, region=g)[0]
        low, q1, med, q3, high = [float(b[key]) for key in ['low', 'q1', 'median', 'q3', 'high']]
        if not low <= q1 <= med <= q3 <= high:
            raise ValueError('Invalid five-number summary')
        yy = base - 0.18
        ax.plot([low, high], [yy, yy], c='#555', lw=1.5)
        ax.add_patch(Rectangle((q1, yy - 0.055), q3 - q1, 0.11, fc=col(cc, 0.58), ec='#333', lw=1.4, zorder=3))
        ax.plot([med, med], [yy - 0.055, yy + 0.055], c='black', lw=5, zorder=4)
        rr = subset(rug, region=g)
        xx = nums(rr, 'age')
        ww = nums(rr, 'weight')
        ax.vlines(xx, yy - 0.1, yy - 0.1 - ww, colors='#777', alpha=0.38, lw=0.6, zorder=1)
        c.text(15, 52 + k * 122, g, fontsize=23, ha='left', va='center', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})


if __name__ == "__main__":
    from render import main
    main(default_figure=17)
