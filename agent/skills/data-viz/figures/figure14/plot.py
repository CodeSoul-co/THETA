"""Figure 14: 三组分子动力学山脊图.

Run from the repository root: python -m figures.figure14.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
from vizlib.common import subset, col, line_data


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    for p, (top, bottom) in enumerate([(25, 273), (391, 636), (750, 1018)]):
        ax = c.ax([40, top, 904, bottom - top])
        ax.set_xlim(s['xlims'][p])
        ax.set_ylim(0, (bottom - top) / 37.8)
        ax.spines[:].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#888')
        ax.set_yticks([])
        for k, g in enumerate([0, 8, 12, 16, 20, 24]):
            rr = subset(rows, panel=p, group=g)
            x, y = line_data(rr)
            base = 5 - k
            color = s['palettes'][p][k]
            ax.fill_between(x, base, base + y, color=color, zorder=k * 2, linewidth=0)
            ax.plot(x, base + y, c=col('white', 0.8), lw=1.6, zorder=k * 2 + 1)
            ax.plot(x, np.full_like(x, base), c=color, lw=1.3, zorder=k * 2 + 1)
            ax.text(x[0], base + 0.1, f'{g} ns', color=color, fontweight='bold', fontsize=25, va='bottom')
        ax.set_xlabel(s['xlabels'][p], fontsize=25, labelpad=4)
        ax.tick_params(axis='x', length=0, pad=10, labelsize=24)


if __name__ == "__main__":
    from render import main
    main(default_figure=14)
