"""
Training pipeline: download the data, finetune yolo11n, evaluate it against
two baselines on the test split, sweep the cost threshold, and save
everything app.py needs.

    uv run python main.py train

Environment overrides, for quick runs: EPOCHS, IMGSZ, BATCH, and
SKIP_TRAIN=1 to re-evaluate the weights already in weights/ without training.
"""

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

import costs
from dataset import DATASET_DIR, OUTPUTS_DIR, class_names, describe_dataset, download_dataset

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
RUNS_DIR = ROOT / "runs"

TRAINING_INFO_PATH = WEIGHTS_DIR / "training_info.json"

BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", 50))
IMGSZ = int(os.environ.get("IMGSZ", 640))
BATCH = int(os.environ.get("BATCH", 16))
PATIENCE = 15
SEED = 0

EVAL_CONF = 0.001                       # keep low-confidence boxes so AP and the sweep see them
MATCH_IOU = 0.5                         # a prediction belongs to a fruit at IoU >= 0.5
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
REVIEW_CAPS = (0.02, 0.05, 0.10, 0.20, 1.0)
CHOSEN_REVIEW_CAP = 0.10                # max share of fruit sent to a person; argued in REPORT.md

# The pretrained COCO model only knows fruit types, never rot, so the best it
# can do is call every fruit it finds fresh.
COCO_TO_OURS = {47: "good apple", 46: "good banana", 49: "good orange"}


def write_data_yaml() -> Path:
    """Roboflow's data.yaml uses ../ paths that Ultralytics resolves wrongly."""
    path = DATASET_DIR / "data_abs.yaml"
    path.write_text(yaml.safe_dump({
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(class_names())),
    }))
    return path


def train(data_yaml: Path) -> dict:
    """Finetune, copy the best weights into weights/, and record how it was trained."""
    started = time.time()
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(BASE_MODEL)
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE,
                seed=SEED, device=device, project=str(RUNS_DIR), name="finetune", exist_ok=True)
    save_dir = Path(model.trainer.save_dir)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    shutil.copy(save_dir / "weights" / "best.pt", WEIGHTS_PATH)
    # Training curves, for the report.
    for name in ("results.csv", "results.png"):
        if (save_dir / name).is_file():
            shutil.copy(save_dir / name, OUTPUTS_DIR / f"training_{name}")
    history = pd.read_csv(save_dir / "results.csv")
    history.columns = history.columns.str.strip()
    info = {
        "base_model": BASE_MODEL, "epochs_requested": EPOCHS, "epochs_run": int(len(history)),
        "best_epoch": int(history.loc[history["metrics/mAP50-95(B)"].idxmax(), "epoch"]),
        "imgsz": IMGSZ, "batch": BATCH, "patience": PATIENCE, "seed": SEED,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seconds": round(time.time() - started),
        "ultralytics": __import__("ultralytics").__version__, "torch": torch.__version__,
    }
    TRAINING_INFO_PATH.write_text(json.dumps(info, indent=2))
    return info


def plot_cost_curve(sweep: pd.DataFrame, best_t: float) -> None:
    """Static cost-vs-threshold chart for the report; the app draws a live one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep["threshold"], sweep["cost_per_1000"], lw=2.5, label="central: FN €10, base rate 5%")
    for col, label in (("cost_fn_A", "FN €1.5 (A)"), ("cost_fn_C", "FN €20 (C)"),
                       ("cost_base_0.02", "base rate 2%"), ("cost_base_0.15", "base rate 15%")):
        ax.plot(sweep["threshold"], sweep[col], lw=1, ls="--", label=label)
    best_cost = sweep.loc[sweep["threshold"] == best_t, "cost_per_1000"].iloc[0]
    ax.scatter([best_t], [best_cost], color="red", zorder=5, label=f"chosen t = {best_t:.2f}")
    ax.set_xlabel("rot-score threshold (discard if score ≥ threshold)")
    ax.set_ylabel("expected cost, € per 1,000 fruits")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cost_curve.png", dpi=150)
    plt.close(fig)


def ultralytics_test_metrics(data_yaml: Path, names: list[str]) -> dict:
    """Ultralytics' own test-split metrics and confusion matrix for the finetuned model."""
    metrics = YOLO(WEIGHTS_PATH).val(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH,
                                     project=str(RUNS_DIR), name="test", exist_ok=True, plots=True)
    box = metrics.box
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        per_class[names[int(c)]] = {"precision": float(box.p[i]), "recall": float(box.r[i]),
                                    "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i])}
    for plot in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = Path(metrics.save_dir) / plot
        if src.is_file():
            shutil.copy(src, OUTPUTS_DIR / plot)
    return {
        "map50": float(box.map50), "map50_95": float(box.map),
        "precision": float(box.mp), "recall": float(box.mr),
        "per_class": per_class,
        # Rows are predicted class, columns true class; last index is background.
        "confusion_matrix": metrics.confusion_matrix.matrix.astype(int).tolist(),
        "confusion_labels": names + ["background"],
    }


def load_ground_truth(image: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    rows = [line.split() for line in label.read_text().strip().splitlines()] if label.is_file() else []
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    arr = np.array([[float(v) for v in row[:5]] for row in rows])
    cx, cy, w, h = arr[:, 1] * width, arr[:, 2] * height, arr[:, 3] * width, arr[:, 4] * height
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, arr[:, 0].astype(int)


def match_predictions(iou: np.ndarray, gt_cls, pred_cls, pred_conf) -> np.ndarray:
    """Greedy, highest confidence first: (n_pred, n_iou_thresholds) true-positive flags."""
    tp = np.zeros((len(pred_cls), len(IOU_THRESHOLDS)), dtype=bool)
    order = np.argsort(-pred_conf)
    for k, threshold in enumerate(IOU_THRESHOLDS):
        taken = np.zeros(len(gt_cls), dtype=bool)
        for j in order:
            candidates = np.where((gt_cls == pred_cls[j]) & ~taken & (iou[:, j] >= threshold))[0]
            if candidates.size:
                taken[candidates[np.argmax(iou[candidates, j])]] = True
                tp[j, k] = True
    return tp


def evaluate(model: YOLO | None, names: list[str], label: str, class_map: dict | None = None):
    """Run one model over the test split; return detection metrics and per-fruit scores.

    model=None is the trivial baseline that predicts nothing. class_map maps the
    model's class ids onto ours (for the COCO model); unmapped classes are dropped.
    """
    bad = np.array([n.startswith("bad") for n in names])
    images = sorted((DATASET_DIR / "test" / "images").glob("*"))
    all_tp, all_conf, all_pred_cls, all_target_cls, fruits = [], [], [], [], []

    if model is None:
        results = [None] * len(images)
    else:
        classes = list(class_map) if class_map else None
        results = model.predict([str(p) for p in images], conf=EVAL_CONF, imgsz=IMGSZ, classes=classes,
                                stream=True, verbose=False, max_det=300)

    for image, result in zip(images, results):
        if result is None:
            height, width = 640, 640
            pred_boxes, pred_conf, pred_cls = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        else:
            height, width = result.orig_shape
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_conf = result.boxes.conf.cpu().numpy()
            pred_cls = result.boxes.cls.cpu().numpy().astype(int)
            if class_map:
                pred_cls = np.array([names.index(class_map[c]) for c in pred_cls], dtype=int)
        gt_boxes, gt_cls = load_ground_truth(image, width, height)
        iou = (box_iou(torch.tensor(gt_boxes), torch.tensor(pred_boxes)).numpy()
               if len(gt_boxes) and len(pred_boxes) else np.zeros((len(gt_boxes), len(pred_boxes))))

        all_tp.append(match_predictions(iou, gt_cls, pred_cls, pred_conf))
        all_conf.append(pred_conf)
        all_pred_cls.append(pred_cls)
        all_target_cls.append(gt_cls)

        # Decision view: one row per real fruit. Its rot score is the highest
        # confidence of any "bad" box on it; a fruit nobody flags scores 0 and is packed.
        for g in range(len(gt_cls)):
            on_fruit = iou[g] >= MATCH_IOU if len(pred_cls) else np.zeros(0, dtype=bool)
            rot = pred_conf[on_fruit & bad[pred_cls]] if len(pred_cls) else np.zeros(0)
            fresh = pred_conf[on_fruit & ~bad[pred_cls]] if len(pred_cls) else np.zeros(0)
            fruits.append({
                "image": image.name, "true_class": names[gt_cls[g]],
                "fruit": names[gt_cls[g]].split()[1], "is_rotten": bool(bad[gt_cls[g]]),
                "rot_score": float(rot.max()) if rot.size else 0.0,
                "fresh_score": float(fresh.max()) if fresh.size else 0.0,
            })

    tp = np.concatenate(all_tp)
    target_cls = np.concatenate(all_target_cls)
    per_class = {n: {"ap50": 0.0, "ap50_95": 0.0} for n in names}
    if len(tp):
        out = ap_per_class(tp, np.concatenate(all_conf), np.concatenate(all_pred_cls), target_cls)
        ap, classes = out[5], out[6]
        for row, c in zip(ap, classes):
            per_class[names[int(c)]] = {"ap50": float(row[0]), "ap50_95": float(row.mean())}
    summary = {
        "model": label,
        "map50": float(np.mean([v["ap50"] for v in per_class.values()])),
        "map50_95": float(np.mean([v["ap50_95"] for v in per_class.values()])),
        "per_class": per_class,
    }
    return summary, pd.DataFrame(fruits)


def decision_summary(fruits: pd.DataFrame, threshold: float) -> dict:
    """Rotten-vs-fresh results on the test split at one threshold, counted per fruit."""
    flagged = fruits["rot_score"] >= threshold
    rotten = fruits["is_rotten"]
    tp, fn = int((flagged & rotten).sum()), int((~flagged & rotten).sum())
    fp, tn = int((flagged & ~rotten).sum()), int((~flagged & ~rotten).sum())
    rates = costs.outcome_rates(fruits["rot_score"], rotten, threshold, threshold)
    per_fruit = {}
    for fruit, group in fruits.groupby("fruit"):
        g_flag, g_rot = group["rot_score"] >= threshold, group["is_rotten"]
        per_fruit[fruit] = {
            "rotten_recall": float((g_flag & g_rot).sum() / max(g_rot.sum(), 1)),
            "rotten_precision": float((g_flag & g_rot).sum() / max(g_flag.sum(), 1)),
            "n_rotten": int(g_rot.sum()), "n_good": int((~g_rot).sum()),
        }
    return {
        "threshold": threshold, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "rotten_recall": tp / max(tp + fn, 1), "rotten_precision": tp / max(tp + fp, 1),
        "cost_per_1000": costs.cost_per_1000(rates), "per_fruit": per_fruit,
    }


def cost_sweep_table(fruits: pd.DataFrame) -> pd.DataFrame:
    scores, rotten = fruits["rot_score"], fruits["is_rotten"]
    table = pd.DataFrame({"threshold": costs.THRESHOLDS, "cost_per_1000": costs.sweep(scores, rotten)})
    for name, fn_cost in costs.FN_SCENARIOS.items():
        table[f"cost_fn_{name.split(':')[0]}"] = costs.sweep(scores, rotten, cost_fn=fn_cost)
    for rate in costs.BASE_RATE_SCENARIOS:
        table[f"cost_base_{rate:.2f}"] = costs.sweep(scores, rotten, base_rate=rate)
    rates = [costs.outcome_rates(scores, rotten, t, t) for t in costs.THRESHOLDS]
    table["rotten_recall"] = [r["rotten"]["fail"] for r in rates]
    table["good_discarded"] = [r["good"]["fail"] for r in rates]
    return table


def main() -> None:
    download_dataset()
    describe_dataset()
    names = class_names()
    data_yaml = write_data_yaml()

    if os.environ.get("SKIP_TRAIN") == "1" and WEIGHTS_PATH.is_file():
        print(f"SKIP_TRAIN=1: evaluating existing {WEIGHTS_PATH}")
        training_info = json.loads(TRAINING_INFO_PATH.read_text()) if TRAINING_INFO_PATH.is_file() else {}
    else:
        training_info = train(data_yaml)

    official = ultralytics_test_metrics(data_yaml, names)
    finetuned, fruits = evaluate(YOLO(WEIGHTS_PATH), names, "yolo11n finetuned")
    coco, coco_fruits = evaluate(YOLO(BASE_MODEL), names, "yolo11n COCO zero-shot", COCO_TO_OURS)
    nothing, nothing_fruits = evaluate(None, names, "predict nothing")
    fruits.to_csv(OUTPUTS_DIR / "test_scores.csv", index=False)

    sweep = cost_sweep_table(fruits)
    sweep.to_csv(OUTPUTS_DIR / "cost_sweep.csv", index=False)
    best_t = float(sweep.loc[sweep["cost_per_1000"].idxmin(), "threshold"])
    plot_cost_curve(sweep, best_t)
    scenario_minima = {col: float(sweep.loc[sweep[col].idxmin(), "threshold"])
                       for col in sweep.columns if col.startswith("cost_")}

    bands = []
    for cap in REVIEW_CAPS:
        band = costs.best_band(fruits["rot_score"], fruits["is_rotten"], cap)
        band["review_cap"] = cap
        band["reviewers_needed"] = costs.reviewers_needed(band["review_share"])
        bands.append(band)
    pd.DataFrame(bands).to_csv(OUTPUTS_DIR / "review_bands.csv", index=False)
    chosen_band = next(b for b in bands if b["review_cap"] == CHOSEN_REVIEW_CAP)

    alternatives = {
        "do nothing (pack everything)": costs.BASE_RATE * costs.COST_FN * 1000,
        "person checks everything": costs.COST_REVIEW * 1000,
        "model alone": float(sweep["cost_per_1000"].min()),
        "model + review band": chosen_band["cost_per_1000"],
    }

    baselines = []
    for summary, scores in ((finetuned, fruits), (coco, coco_fruits), (nothing, nothing_fruits)):
        baselines.append({**{k: summary[k] for k in ("model", "map50", "map50_95")},
                          "decision_at_0.5": decision_summary(scores, 0.5),
                          "min_cost_per_1000": float(costs.sweep(scores["rot_score"], scores["is_rotten"]).min()),
                          "per_class": summary["per_class"]})

    metrics = {
        "positive_class": "rotten (any 'bad *' class)",
        "training": training_info,
        "finetuned_ultralytics_test": official,
        "baselines": baselines,
    }
    operating_point = {
        "costs": {"fp": costs.COST_FP, "fn": costs.COST_FN, "tp": costs.COST_TP, "tn": costs.COST_TN,
                  "review": costs.COST_REVIEW, "base_rate": costs.BASE_RATE},
        "single_threshold": decision_summary(fruits, best_t),
        "scenario_minima": scenario_minima,
        "review_band": chosen_band,
        "alternatives_cost_per_1000": alternatives,
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUTS_DIR / "operating_point.json").write_text(json.dumps(operating_point, indent=2))

    print("\n=== Test split ===")
    for b in baselines:
        d = b["decision_at_0.5"]
        print(f"{b['model']:<24} mAP50 {b['map50']:.3f}  mAP50-95 {b['map50_95']:.3f}  "
              f"rotten recall@0.5 {d['rotten_recall']:.3f}  min cost/1000 EUR {b['min_cost_per_1000']:.1f}")
    print(f"\nCheapest single threshold: {best_t:.2f} -> EUR {sweep['cost_per_1000'].min():.1f} per 1,000 fruits")
    print(f"Review band (cap {CHOSEN_REVIEW_CAP:.0%}): {chosen_band}")
    print(f"Alternatives per 1,000: {alternatives}")


if __name__ == "__main__":
    main()
