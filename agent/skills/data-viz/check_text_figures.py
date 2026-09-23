"""Validate natural-language template structure and removable auxiliary layers."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from figures.figure21.plot import layout
from figures.figure22.simulate_links import generate
from render import ROOT,render
from scripts.prepare_workspace import prepare
from vizlib.common import read_csv


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid structure accepted')


def main():
    leaves=[dict(node='a',order='0',words='alpha'),dict(node='b',order='1',words='beta')]
    layout([dict(node='root',left='a',right='b',height='1')],leaves)
    reject(lambda:layout([dict(node='root',left='a',right='missing',height='1')],leaves))
    reject(lambda:layout([dict(node='root',left='a',right='a',height='1')],leaves))
    reject(lambda:layout([dict(node='root',left='a',right='b',height='-1')],leaves))
    reject(lambda:layout([dict(node='root',left='a',right='b',height='nan')],leaves))
    three=leaves+[dict(node='c',order='2',words='gamma')]
    reject(lambda:layout([dict(node='ab',left='a',right='b',height='2'),dict(node='root',left='ab',right='c',height='1')],three))
    with tempfile.TemporaryDirectory(prefix='text-figures-check-') as temp:
        temp=Path(temp)
        # Reproduce the committed synthetic link snapshot independently.
        p=temp/'simulated.csv'
        rows=generate()
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        assert p.read_bytes()==(ROOT/'figures/figure22/data/figure22.csv').read_bytes()
        for number, flag, removed in [(20,'show_mesh','figure20_mesh.csv'),(22,'show_dissemination','figure22_summary.csv')]:
            source=ROOT/f'figures/figure{number}'
            data=temp/f'data{number}';shutil.copytree(source/'data',data)
            (data/removed).unlink()
            s=json.loads((source/'style.json').read_text());axes=s['axes'][:]
            s[flag]=False
            style=temp/f'style{number}.json';style.write_text(json.dumps(s))
            render(number,data_dir=data,style_file=style,out=temp/f'out{number}',formats=('png','svg'),annotations='none')
            assert json.loads(style.read_text())['axes']==axes
            svg=(temp/f'out{number}/figure{number}.svg').read_text()
            assert '<image' not in svg
            if number==22:
                assert 'Dissemination' not in svg and 'Documents' in svg and 'Words' in svg
        # Reject an inverted box summary rather than silently drawing it.
        data=temp/'bad-summary';shutil.copytree(ROOT/'figures/figure22/data',data)
        rows=read_csv(data/'figure22_summary.csv');rows[0]['q1']='0.99'
        with (data/'figure22_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        reject(lambda:render(22,data_dir=data,out=temp/'bad-output',formats=('png',)))
        # The three new modules must work when copied out of the installed skill.
        project=prepare(temp/'standalone',[20,21,22])
        subprocess.run([sys.executable,'render.py','--figure','20,21,22','--format','png','svg',
                        '--annotations','none'],cwd=project,check=True)
        for n in [20,21,22]:
            assert (project/f'output/figure{n}.png').stat().st_size>1000
        # The scaffold recommendation must not collide with copied figure20.
        assert '--id 23' in (project/'README.en.md').read_text()
    print('PASS: hierarchy validation, seeded links, removable mesh/boxes, and isolated text templates.')


if __name__=='__main__':
    main()
