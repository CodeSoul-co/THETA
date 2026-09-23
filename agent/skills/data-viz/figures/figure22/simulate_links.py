"""Regenerate ONLY the synthetic example flow table; no corpus analysis is performed."""
import argparse
import csv
from pathlib import Path
import numpy as np


def generate(seed=202622):
    rng = np.random.default_rng(seed)
    rows = []
    for i, count in enumerate([36,30,12,38,36,36,46,66]):
        p = np.array([.08,.08,.09,.07,.07,.025,.025,.07,.12,.375])
        if i == 0:
            p = np.array([.025,.025,.025,.025,.025,.025,.025,.025,.30,.50])
        if i == 7:
            p = np.array([.18,.15,.17,.12,.09,.02,.01,.03,.08,.15])
        if i == 1:
            p[5] += .16
        if i == 2:
            p[6] += .7
        p /= p.sum()
        for j in range(count):
            k = int(rng.choice(10, p=p))
            rows.append(dict(link_id=f'l{len(rows):03d}',document_id=f'd{i}',topic_id=f't{k}',
                             weight=round(float(rng.uniform(.55,1.5)),3),
                             source_position=round((j+.5)/count,5),
                             target_position=round(float(rng.uniform(.02,.98)),5)))
    return rows


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='New CSV path; existing files are never overwritten')
    parser.add_argument('--seed',type=int,default=202622)
    args=parser.parse_args()
    rows=generate(args.seed)
    with args.out.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} synthetic links to {args.out}')
