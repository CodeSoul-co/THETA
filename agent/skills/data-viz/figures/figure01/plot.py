"""Figure 01: 空心柱状图与散点.

Run from the repository root: python -m figures.figure01.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

from vizlib.common import nums, subset, clean, label_axes, bracket, summary


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    ax = c.ax()
    clean(ax)
    label_axes(ax, s)
    for j, (g, color) in enumerate(zip(s['groups'], s['colors'])):
        rr = subset(rows, group=g)
        mean, error = summary(rr, c.summary)
        ax.bar(j, mean, 0.82, facecolor='white', edgecolor=color, lw=4)
        ax.errorbar(j, mean, yerr=error, fmt='none', ecolor=color, elinewidth=2.1, capsize=18, capthick=2.1)
        ax.scatter(j + c.rng.uniform(-0.36, 0.36, len(rr)), nums(rr, 'value'), s=180, c=color, edgecolors='none', zorder=3)
    ax.set_xlim(-0.85, 2.85)
    ax.set_xticks(range(3), s['groups'], rotation=42, ha='right', fontstyle='italic')
    ax.tick_params(length=15, width=3)
    if c.annotations:
        for text, a, b, y in s['annotations']:
            bracket(ax, a, b, y, text, lw=2.4)


if __name__ == "__main__":
    from render import main
    main(default_figure=1)
