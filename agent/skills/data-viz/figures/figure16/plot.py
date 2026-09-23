"""Figure 16: 细胞群表达山脊矩阵.

Run from the repository root: python -m figures.figure16.plot
Input contracts and data provenance: README.md / README.en.md in this folder.
"""

import numpy as np
from vizlib.common import subset, clean, line_data


def draw(c, rows):
    """Draw this figure from its CSV rows and style configuration."""
    s = c.s
    for j, marker in enumerate(s['markers']):
        ax = c.ax([91 + j * 167, 58, 158, 319])
        clean(ax, True, '#777')
        ax.set_xlim(s['xlim'])
        ax.set_ylim(-0.6, 9.8)
        ax.set_yticks([])
        for k, (group, cc) in enumerate(zip(s['groups'], s['colors'])):
            x, y = line_data(subset(rows, marker=marker, group=group))
            base = 8 - k
            ax.fill_between(x, base, base + y, fc=cc, lw=0, zorder=k + 1)
            ax.plot(x, base + y, c='#444', lw=1.1, zorder=k + 1)
            mass = np.r_[0, np.cumsum((y[1:] + y[:-1]) * np.diff(x) / 2)]
            if c.annotations or mass[-1] > 0:
                med = s['medians'][j][k] if c.annotations else np.interp(mass[-1] / 2, mass, x)
                ax.plot([med, med], [base, base + np.interp(med, x, y)], c='#555', lw=0.9, zorder=k + 2)
        ax.set_xticks([0, 2, 4, 6])
        ax.tick_params(length=3, pad=2, labelsize=12)
        title = c.ax([91 + j * 167, 28, 158, 30])
        clean(title, True, '#777')
        title.set_xticks([])
        title.set_yticks([])
        title.text(0.5, 0.45, marker, ha='center', va='center', fontsize=15)
        if j == 0:
            ax.set_yticks(range(9), s['groups'][::-1])
            ax.tick_params(axis='y', length=0, labelsize=13, pad=7)
    c.text(36, 172, 'Vγ9Vδ2', rotation=90, ha='center', va='center', fontsize=12)
    c.text(36, 296, 'Vδ1', rotation=90, ha='center', va='center', fontsize=12)
    c.text(592, 402, 'Expression', ha='center', va='top', color='#888', fontsize=8)


if __name__ == "__main__":
    from render import main
    main(default_figure=16)
