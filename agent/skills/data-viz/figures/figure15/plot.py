"""Figure 15: 流式细胞术山脊矩阵.

Run from the repository root: python -m figures.figure15.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

from vizlib.common import subset, col, clean, line_data


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    for j, marker in enumerate(s['markers']):
        ax = c.ax([122 + j * 187.2, 38, 187.2, 518])
        clean(ax, True, '#888')
        ax.set_xlim(s['xlim'])
        ax.set_ylim(-0.2, 14.85)
        ax.set_yticks([])
        for k in range(14):
            group = s['groups'][k // 2]
            condition = '−' if k % 2 == 0 else '+'
            rr = subset(rows, marker=marker, group=group, condition=condition)
            x, y = line_data(rr)
            base = 13 - k
            cc = s['colors'][k // 2]
            ax.fill_between(x, base, base + y, color=col(cc, 0.2 if k % 2 == 0 else 0.94), lw=0, zorder=k + 1)
            ax.plot(x, base + y, color=col(cc, 0.35 if k % 2 == 0 else 0.55), lw=0.9, zorder=k + 1)
            ax.axhline(base, c=col(cc, 0.36), lw=0.65)
            if j == 0:
                ax.text(-0.44, base + 0.38, condition, ha='center', va='center', transform=ax.get_yaxis_transform(), fontsize=17)
        ax.set_xticks([0, 2, 4, 6], ['$10^0$', '$10^2$', '$10^4$', '$10^6$'], rotation=55, fontsize=6)
        ax.tick_params(length=2, pad=0)
        ax.set_xlabel(marker, fontsize=20, fontweight='bold', labelpad=8)
    for k, (g, cc) in enumerate(zip(s['groups'], s['colors'])):
        c.text(982, 93 + k * 69.4, g, color=cc, fontweight='bold', fontsize=20, ha='center', va='center')


if __name__ == "__main__":
    from render import main
    main(default_figure=15)
