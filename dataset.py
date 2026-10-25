"""
Download the Roboflow dataset and describe its composition.

Shared by train.py. Can also be run on its own to inspect the data before
training:

    uv run python dataset.py
"""

import json
from collections import Counter
from pathlib import Path

import yaml

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent

WORKSPACE = "deep-learning-computer-vision-project"
PROJECT = "rotten-fruit-detector"
VERSION = 3
EXPORT_FORMAT = "yolov11"

DATASET_DIR = ROOT / "datasets" / f"{PROJECT}-{VERSION}"
OUTPUTS_DIR = ROOT / "outputs"
SPLITS = ("train", "valid", "test")

# Box area as a share of the image, following COCO's small/medium/large idea
# but relative to image size, since every image is resized to 640x640.
SIZE_BUCKETS = (("small", 0.01), ("medium", 0.10), ("large", 1.01))


def download_dataset() -> Path:
    """Download the dataset version once; reuse it on later runs."""
    if (DATASET_DIR / "data.yaml").is_file():
        return DATASET_DIR
    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download(EXPORT_FORMAT, location=str(DATASET_DIR), overwrite=True)
    return DATASET_DIR


def class_names(dataset_dir: Path = DATASET_DIR) -> list[str]:
    names = yaml.safe_load((dataset_dir / "data.yaml").read_text())["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def size_bucket(area: float) -> str:
    for name, upper in SIZE_BUCKETS:
        if area < upper:
            return name
    return SIZE_BUCKETS[-1][0]


def describe_split(split_dir: Path, names: list[str]) -> dict:
    images = sorted((split_dir / "images").glob("*"))
    instances = Counter()
    images_with_class = Counter()
    sizes = {name: Counter() for name in names}
    areas = {name: [] for name in names}
    empty_images = 0
    boxes_per_image = []

    for image in images:
        label = split_dir / "labels" / f"{image.stem}.txt"
        lines = label.read_text().strip().splitlines() if label.is_file() else []
        if not lines:
            empty_images += 1
        boxes_per_image.append(len(lines))
        seen = set()
        for line in lines:
            parts = line.split()
            name = names[int(parts[0])]
            # Detection labels are "cls cx cy w h"; polygons have more points.
            if len(parts) == 5:
                w, h = float(parts[3]), float(parts[4])
            else:
                xs, ys = [float(v) for v in parts[1::2]], [float(v) for v in parts[2::2]]
                w, h = max(xs) - min(xs), max(ys) - min(ys)
            instances[name] += 1
            seen.add(name)
            areas[name].append(w * h)
            sizes[name][size_bucket(w * h)] += 1
        images_with_class.update(seen)

    def median(values: list[float]) -> float | None:
        values = sorted(values)
        return round(values[len(values) // 2], 4) if values else None

    return {
        "images": len(images),
        "empty_images": empty_images,
        "mean_boxes_per_image": round(sum(boxes_per_image) / max(len(images), 1), 2),
        "instances": {name: instances[name] for name in names},
        "images_with_class": {name: images_with_class[name] for name in names},
        "median_box_area": {name: median(areas[name]) for name in names},
        "size_buckets": {name: dict(sizes[name]) for name in names},
    }


def describe_dataset(dataset_dir: Path = DATASET_DIR) -> dict:
    names = class_names(dataset_dir)
    stats = {
        "source": f"{WORKSPACE}/{PROJECT}/{VERSION}",
        "classes": names,
        "splits": {s: describe_split(dataset_dir / s, names) for s in SPLITS if (dataset_dir / s).is_dir()},
    }
    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def print_stats(stats: dict) -> None:
    names = stats["classes"]
    splits = stats["splits"]
    print(f"Dataset {stats['source']}")
    print(f"{'split':<7}{'images':>8}{'empty':>7}{'boxes/img':>11}")
    for split, s in splits.items():
        print(f"{split:<7}{s['images']:>8}{s['empty_images']:>7}{s['mean_boxes_per_image']:>11}")

    print(f"\n{'class':<14}" + "".join(f"{split:>8}" for split in splits) + f"{'median area':>13}  small/med/large (all splits)")
    for name in names:
        counts = "".join(f"{splits[split]['instances'][name]:>8}" for split in splits)
        merged = Counter()
        for s in splits.values():
            merged.update(s["size_buckets"][name])
        area = splits.get("train", next(iter(splits.values())))["median_box_area"][name]
        buckets = "/".join(str(merged[b]) for b, _ in SIZE_BUCKETS)
        print(f"{name:<14}{counts}{area if area is not None else '-':>13}  {buckets}")


if __name__ == "__main__":
    download_dataset()
    print_stats(describe_dataset())
