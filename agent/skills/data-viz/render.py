"""Discover and render the gallery. Each figures/figureNN/plot.py owns its drawing code."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from vizlib.common import ROOT, Figure, read_csv
import matplotlib.pyplot as plt


def discover(root=ROOT):
    """A folder with plot.py and style.json is a figure; no central registry to edit."""
    found = {}
    for folder in sorted((Path(root) / "figures").glob("figure*")):
        if not re.fullmatch(r"figure\d{2,}", folder.name):
            continue
        if not (folder / "plot.py").is_file() or not (folder / "style.json").is_file():
            continue
        style = json.loads((folder / "style.json").read_text(encoding="utf-8"))
        number = int(folder.name.removeprefix("figure"))
        if style["id"] != number or number in found:
            raise ValueError(f"Duplicate or inconsistent figure ID in {folder.name}")
        found[number] = folder
    return dict(sorted(found.items()))


def dataset_matches(folder, data_dir, data_file):
    """Use the bundled manifest, never a fingerprint provided by replacement data."""
    manifest = folder / "example_hashes.json"
    if not manifest.is_file():
        return False
    hashes = json.loads(manifest.read_text(encoding="utf-8"))
    if data_file not in hashes:
        return False
    return all(
        (data_dir / name).is_file()
        and hashlib.sha256((data_dir / name).read_bytes()).hexdigest() == digest
        for name, digest in hashes.items()
    )


def render(number, data_dir=None, style_file=None, out=ROOT / "output", formats=("png", "svg"),
           scale=1.0, annotations="auto", summary_mode="auto", root=ROOT):
    folder = discover(root)[number]
    style_path = Path(style_file) if style_file else folder / "style.json"
    style = json.loads(style_path.read_text(encoding="utf-8"))
    if any(style[k] <= 0 for k in ("width", "height", "dpi")) or scale <= 0:
        raise ValueError("Canvas dimensions, DPI and scale must be positive")
    data_dir = Path(data_dir) if data_dir else folder / "data"
    if (data_dir / folder.name).is_dir():
        data_dir = data_dir / folder.name
    rows = read_csv(data_dir / style["data_file"])
    matches = dataset_matches(folder, data_dir, style["data_file"])
    annotation_enabled = annotations == "reference" or (annotations == "auto" and matches)
    mode = ("provided" if matches else "samples") if summary_mode == "auto" else summary_mode
    spec = importlib.util.spec_from_file_location(f"gallery_{folder.name}", folder / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = Figure(style, data_dir, annotation_enabled, mode)
    try:
        module.draw(context, rows)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for fmt in formats:
            destination = out / f"{folder.name}.{fmt}"
            # Fixed canvas: automatic tight cropping would change the reference dimensions.
            context.fig.savefig(destination, dpi=style["dpi"] * scale, format=fmt, facecolor="white")
            paths.append(destination.name)
        return {
            "figure": number, "title": style["title"], "files": paths, "font": context.font,
            "reference_annotations": annotation_enabled, "summary_mode": mode,
            "example_data_unchanged": matches,
            "provenance": style["provenance"] if matches else "User-supplied data; original annotations disabled in auto mode.",
        }
    finally:
        plt.close(context.fig)


def main(default_figure=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", default=str(default_figure) if default_figure else "all")
    parser.add_argument("--list", action="store_true", help="List discovered figures")
    parser.add_argument("--data-dir", type=Path, help="One figure's CSV folder, or a parent containing figureNN folders")
    parser.add_argument("--style", type=Path, help="Replacement style.json; single figure only")
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    parser.add_argument("--format", nargs="+", choices=["png", "svg", "pdf"], default=["png", "svg"])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--annotations", choices=["auto", "none", "reference"], default="auto")
    parser.add_argument("--summary", choices=["auto", "provided", "samples"], default="auto")
    args = parser.parse_args()
    available = discover()
    if args.list:
        for number, folder in available.items():
            style = json.loads((folder / "style.json").read_text(encoding="utf-8"))
            print(f"{number:02d}  {style.get('title_en', style['title'])}")
        return
    try:
        numbers = list(available) if args.figure == "all" else [int(n) for n in args.figure.split(",")]
    except ValueError:
        parser.error("--figure expects all or comma-separated integers")
    if any(n not in available for n in numbers):
        parser.error("Unknown figure ID; run --list")
    if args.style and len(numbers) != 1:
        parser.error("--style is only supported when selecting one figure")
    reports = []
    for number in numbers:
        report = render(number, args.data_dir, args.style, args.out, args.format,
                        args.scale, args.annotations, args.summary)
        reports.append(report)
        print(f"figure{number:02d}: " + ", ".join(report["files"]), flush=True)
    (args.out / "render_log.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
