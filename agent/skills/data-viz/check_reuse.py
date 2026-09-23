"""Check all figures, their data contracts, and the add-a-figure workflow."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from render import ROOT, discover, render
from vizlib.common import line_data, nums, summary, tree_layout
from scripts.new_figure import create

# A meaningful numeric change for each specialized chart; new value-based charts work automatically.
CHANGES = {
    1: ("value", .82), 2: ("x", .81), 3: ("x", .81), 4: ("mean", .82),
    5: ("value", .82), 6: ("value", .82), 7: ("value", .82), 8: ("value", .82),
    9: ("indigo", .82), 10: ("length", 1.6), 11: ("length", 1.6),
    12: ("expression", .45), 13: ("cds", .82), 14: ("density", .75),
    15: ("density", .75), 16: ("density", .75), 17: ("density", .75),
    18: ("median", .9), 19: ("wt", .82),
    20: ("x", .82), 21: ("height", .82), 22: ("weight", .65),
}


def check_contract(folder):
    meta = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    hashes = json.loads((folder / "example_hashes.json").read_text(encoding="utf-8"))
    for lang in ("zh", "en"):
        assert meta["raw_data_required"][lang] and meta["provenance"][lang]
    for doc in ("README.md", "README.en.md"):
        assert (folder / doc).stat().st_size > 100
    paths = set()
    for item in meta["files"]:
        path = folder / item["path"]
        paths.add(path.name)
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == item["row_count"], path
        assert list(rows[0]) == [field["name"] for field in item["fields"]], path
        assert all(field["unit"] and field["description"]["zh"] and field["description"]["en"] for field in item["fields"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == hashes[path.name] == item["sha256"], path
    assert paths == set(hashes) == {p.name for p in (folder / "data").glob("*.csv")}


def check_links():
    docs = [ROOT / name for name in ("README.md", "README.en.md", "CONTRIBUTING.md", "CONTRIBUTING.en.md", "SKILL.md")]
    docs += list((ROOT / "docs").glob("*.md")) + list((ROOT / "figures").glob("figure*/*.md"))
    for doc in docs:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if target.startswith(("http:", "https:", "#")):
                continue
            assert (doc.parent / target.split("#")[0]).exists(), (doc, target)


def main(compare_previews=False):
    assert summary([{"value": "1"}, {"value": "3"}], "samples") == (2., 1.)
    x, y = line_data([{"value": v} for v in [-1, -.1, 0, .1, 1]])
    assert np.all(np.diff(x) > 0) and np.isfinite(y).all() and np.isclose(y.max(), 1.4)
    assert abs(x[np.argmax(y)]) < .1
    for invalid in ("nan", "inf", ""):
        try:
            nums([{"value": invalid}], "value")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid numeric input: {invalid}")
    try:
        tree_layout([{"node": "a", "parent": "b", "length": "1"}, {"node": "b", "parent": "a", "length": "1"}], [], 0, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("A disconnected tree cycle was accepted")
    check_links()
    results = []
    with tempfile.TemporaryDirectory(prefix="figure-gallery-check-") as temp:
        temp = Path(temp)
        for number, folder in discover().items():
            check_contract(folder)
            baseline = render(number, out=temp / "baseline", formats=("png", "svg"))
            assert baseline["example_data_unchanged"]
            a = np.asarray(Image.open(temp / f"baseline/figure{number:02d}.png").convert("RGB"))
            if compare_previews:
                expected = np.asarray(Image.open(folder / "preview.png").convert("RGB"))
                assert np.array_equal(a, expected), f"figure{number:02d}: changed from pre-split preview"
            svg = (temp / f"baseline/figure{number:02d}.svg").read_text(encoding="utf-8")
            assert "<image" not in svg, f"figure{number:02d}: embedded bitmap in SVG"
            data = temp / f"data/figure{number:02d}"
            shutil.copytree(folder / "data", data)
            style = json.loads((folder / "style.json").read_text(encoding="utf-8"))
            path = data / style["data_file"]
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            key, factor = CHANGES.get(number, ("value", .82))
            assert key in rows[0], f"Add figure{number:02d}'s change field to CHANGES"
            for row in rows[:1] if number in (10, 11) else rows:
                row[key] = str(float(row[key]) * factor)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            changed = render(number, data_dir=data, out=temp / "changed", formats=("png",))
            assert not changed["reference_annotations"] and not changed["example_data_unchanged"]
            b = np.asarray(Image.open(temp / f"changed/figure{number:02d}.png").convert("RGB"))
            assert a.shape == b.shape == (style["height"], style["width"], 3)
            difference = int(np.count_nonzero(np.any(a != b, axis=2)))
            assert difference > 100, f"figure{number:02d}: changed CSV did not affect the output"
            results.append({"figure": number, "changed_pixels": difference, "old_annotations_disabled": True})
            print(f"PASS figure{number:02d}: data contract, PNG/SVG, replacement ({difference} pixels)", flush=True)
        # Test standalone discovery in a project with no screenshots or pre-existing figures.
        project = temp / "new_project"
        project.mkdir()
        shutil.copy2(ROOT / "render.py", project / "render.py")
        shutil.copytree(ROOT / "vizlib", project / "vizlib", ignore=shutil.ignore_patterns("__pycache__"))
        new = create(20, "新增案例", "New example", root=project)
        assert list(discover(project)) == [20]
        check_contract(new)
        subprocess.run([sys.executable, "-m", "figures.figure20.plot", "--format", "png", "--out", "output"], cwd=project, check=True)
        assert (project / "output/figure20.png").is_file()
        try:
            create(20, "duplicate", "duplicate", root=project)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Scaffolding overwrote an existing figure")
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output/reuse_check.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(results)} figures, bilingual links, metadata, and isolated new-figure workflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-previews", action="store_true", help="Strict local refactor check; requires the same fonts as committed previews")
    main(parser.parse_args().compare_previews)
