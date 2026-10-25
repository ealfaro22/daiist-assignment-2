"""
Gradio dashboard for the rotten-fruit sorter.

    uv run python main.py app

Loads the committed weights in weights/ and the evaluation outputs that
train.py saved in outputs/. It never trains anything: moving a threshold
recomputes costs from the saved per-fruit test scores.
"""

import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import costs

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights" / "best.pt"
OUTPUTS_DIR = ROOT / "outputs"

METRICS = json.loads((OUTPUTS_DIR / "metrics.json").read_text())
OPERATING_POINT = json.loads((OUTPUTS_DIR / "operating_point.json").read_text())
DATASET_STATS = json.loads((OUTPUTS_DIR / "dataset_stats.json").read_text())
FRUITS = pd.read_csv(OUTPUTS_DIR / "test_scores.csv")
SWEEP = pd.read_csv(OUTPUTS_DIR / "cost_sweep.csv")

CHOSEN_T = OPERATING_POINT["single_threshold"]["threshold"]
SELECTION = OPERATING_POINT["single_threshold"]["selection"]
CHOSEN_BAND = OPERATING_POINT["review_band"]
GROUP_IOU = 0.5        # boxes overlapping this much are the same fruit, as in train.py
MIN_SHOW_CONF = 0.05   # hide boxes too weak to matter for any threshold setting

_model = None


def model():
    """Load the YOLO weights on first use so the app starts quickly."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(WEIGHTS_PATH)
    return _model


# ---------------------------------------------------------------- image tab

def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def group_into_fruits(boxes, conf, cls, names):
    """Merge overlapping boxes into one fruit; rot score = strongest 'bad' box on it."""
    order = np.argsort(-conf)
    used = np.zeros(len(conf), dtype=bool)
    fruits = []
    for i in order:
        if used[i]:
            continue
        members = (~used) & (iou_one_to_many(boxes[i], boxes) >= GROUP_IOU)
        members[i] = True
        used |= members
        bad = np.array([names[c].startswith("bad") for c in cls[members]])
        rot_conf = conf[members][bad]
        fruits.append({"box": boxes[i], "label": names[cls[i]],
                       "rot_score": float(rot_conf.max()) if rot_conf.size else 0.0})
    return fruits


def decide(score, t_low, t_high):
    if score >= t_high:
        return "FAIL (discard)", (220, 40, 40)
    if score >= t_low:
        return "REVIEW", (240, 170, 0)
    return "PASS (pack)", (40, 170, 70)


def predict_image(image, t_low, t_high):
    if image is None:
        return None, pd.DataFrame()
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    result = model().predict(image, conf=MIN_SHOW_CONF, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    fruits = group_into_fruits(boxes, conf, cls, result.names)

    canvas = image.copy()
    rows = []
    for n, fruit in enumerate(fruits, 1):
        decision, colour = decide(fruit["rot_score"], t_low, t_high)
        x1, y1, x2, y2 = fruit["box"].astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(canvas, f"#{n} {decision.split()[0]} rot={fruit['rot_score']:.2f}",
                    (x1 + 4, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        rows.append({"fruit": n, "top box label": fruit["label"],
                     "rot score": round(fruit["rot_score"], 3), "decision": decision})
    return canvas, pd.DataFrame(rows)


# ---------------------------------------------------------- operating tab

def outcome_table(rates, base_rate):
    c = costs.per_1000(rates, base_rate)
    return pd.DataFrame({
        "true class (per 1,000 fruits)": ["rotten", "good"],
        "pass (packed)": [c["FN"], c["TN"]],
        "review (person)": [c["review_rotten"], c["review_good"]],
        "fail (discarded)": [c["TP"], c["FP"]],
    }).round(1)


def cost_curve(t_low, t_high, base_rate, cost_fn):
    scores, rotten = FRUITS["rot_score"], FRUITS["is_rotten"]
    live = costs.sweep(scores, rotten, cost_fn=cost_fn, base_rate=base_rate)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=SWEEP["threshold"], y=SWEEP["cost_per_1000"], name="central assumptions",
                             line=dict(color="#1f77b4", width=3)))
    if abs(base_rate - costs.BASE_RATE) > 1e-9 or abs(cost_fn - costs.COST_FN) > 1e-9:
        fig.add_trace(go.Scatter(x=costs.THRESHOLDS, y=live, name="current sliders",
                                 line=dict(color="#ff7f0e", dash="dash")))
    chosen_cost = float(SWEEP.loc[SWEEP["threshold"] == CHOSEN_T, "cost_per_1000"].iloc[0])
    fig.add_trace(go.Scatter(x=[CHOSEN_T], y=[chosen_cost], mode="markers+text", name="chosen threshold",
                             marker=dict(size=12, color="#d62728"), text=[f"t={CHOSEN_T:.2f}"],
                             textposition="top center"))
    fig.add_trace(go.Scatter(x=[SELECTION["literal_minimum"]], y=[SELECTION["literal_minimum_cost"]],
                             mode="markers", name="literal minimum (fragile)",
                             marker=dict(size=10, color="grey", symbol="x")))
    fig.add_vrect(x0=t_low, x1=t_high, fillcolor="#f0b400", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.update_layout(xaxis_title="rot-score threshold (discard if score ≥ threshold)",
                      yaxis_title="expected cost, € per 1,000 fruits", height=420,
                      margin=dict(l=40, r=20, t=30, b=40), legend=dict(orientation="h", y=-0.2))
    return fig


def update_operating(t_low, t_high, base_rate, cost_fn, reviewer_error):
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    rates = costs.outcome_rates(FRUITS["rot_score"], FRUITS["is_rotten"], t_low, t_high)
    cost = costs.cost_per_1000(rates, base_rate, cost_fn, reviewer_error=reviewer_error)
    share = costs.review_share(rates, base_rate)
    c = costs.per_1000(rates, base_rate)
    alternatives = pd.DataFrame({
        "option": ["do nothing (pack everything)", "person checks everything",
                   f"model alone at t={CHOSEN_T:.2f}", "current band"],
        "€ per 1,000 fruits": [
            base_rate * cost_fn * 1000,
            costs.cost_per_1000(costs.everything_reviewed(), base_rate, cost_fn, reviewer_error=reviewer_error),
            costs.cost_per_1000(costs.outcome_rates(FRUITS["rot_score"], FRUITS["is_rotten"],
                                                    CHOSEN_T, CHOSEN_T), base_rate, cost_fn),
            cost,
        ],
    }).round(2)
    rotten_packed = c["FN"] + reviewer_error * c["review_rotten"]
    good_discarded = c["FP"] + reviewer_error * c["review_good"]
    summary = (
        f"### € {cost:,.2f} per 1,000 fruits\n"
        f"- Sent to a person: **{share:.1%}** of fruit → "
        f"{costs.FRUITS_PER_DAY * share:,.0f} fruits/day → "
        f"**{costs.reviewers_needed(share):.1f} reviewers** "
        f"(at {costs.REVIEW_SECONDS:.0f} s each, {costs.REVIEWER_HOURS_PER_DAY:.0f} productive h/day)\n"
        f"- Rotten fruit packed: {rotten_packed:.2f} per 1,000, of which "
        f"{reviewer_error * c['review_rotten']:.2f} are reviewer mistakes\n"
        f"- Good fruit discarded: {good_discarded:.1f} per 1,000"
    )
    return summary, outcome_table(rates, base_rate), alternatives, cost_curve(t_low, t_high, base_rate, cost_fn)


# ------------------------------------------------------------ results tab

def per_class_table():
    official = METRICS["finetuned_ultralytics_test"]["per_class"]
    return pd.DataFrame([{"class": name, **{k: round(v, 3) for k, v in vals.items()}}
                         for name, vals in official.items()])


def baseline_table():
    rows = []
    for b in METRICS["baselines"]:
        d = b["decision_at_0.5"]
        rows.append({"model": b["model"], "mAP50": round(b["map50"], 3), "mAP50-95": round(b["map50_95"], 3),
                     "rotten recall @0.5": round(d["rotten_recall"], 3),
                     "rotten precision @0.5": round(d["rotten_precision"], 3),
                     "min € per 1,000": round(b["min_cost_per_1000"], 1)})
    return pd.DataFrame(rows)


def per_fruit_table():
    per_fruit = OPERATING_POINT["single_threshold"]["per_fruit"]
    return pd.DataFrame([{"fruit": f, **{k: round(v, 3) if isinstance(v, float) else v for k, v in vals.items()}}
                         for f, vals in per_fruit.items()])


def confusion_figure():
    labels = METRICS["finetuned_ultralytics_test"]["confusion_labels"]
    matrix = np.array(METRICS["finetuned_ultralytics_test"]["confusion_matrix"])
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels, colorscale="Blues", text=matrix,
                               texttemplate="%{text}", showscale=False))
    fig.update_layout(xaxis_title="true class", yaxis_title="predicted class", height=480,
                      yaxis=dict(autorange="reversed"), margin=dict(l=40, r=20, t=30, b=40))
    return fig


def dataset_table():
    rows = []
    for split, s in DATASET_STATS["splits"].items():
        rows.append({"split": split, "images": s["images"], **s["instances"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ layout

with gr.Blocks(title="Rotten fruit sorter") as demo:
    gr.Markdown(
        "# Rotten fruit sorter: packing-line decision support\n"
        f"Positive class = **rotten**. Costs: false negative (rotten packed) €{costs.COST_FN:.2f}, "
        f"false positive (good discarded) €{costs.COST_FP:.2f}, review €{costs.COST_REVIEW:.3f}/fruit "
        f"({costs.REVIEW_SECONDS:.0f} s, reviewer wrong {costs.REVIEWER_ERROR:.0%} of the time). "
        f"Base rate {costs.BASE_RATE:.0%}. Chosen single threshold **{CHOSEN_T:.2f}**; "
        f"chosen review band **{CHOSEN_BAND['t_low']:.2f}–{CHOSEN_BAND['t_high']:.2f}**. "
        f"Thresholds are picked by the lowest worst-case cost within ±{SELECTION['window']:.2f}, "
        f"not the literal minimum ({SELECTION['literal_minimum']:.3f}), which hinges on a single fruit."
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=CHOSEN_BAND["t_low"], step=0.002,
                          label="Lower threshold: below this, PASS (pack)")
        t_high = gr.Slider(0, 1, value=CHOSEN_BAND["t_high"], step=0.002,
                           label="Upper threshold: at or above this, FAIL (discard)")

    with gr.Tab("Operating point"):
        with gr.Row():
            base_rate = gr.Slider(0.01, 0.20, value=costs.BASE_RATE, step=0.01, label="Base rate (share rotten)")
            cost_fn = gr.Slider(0.5, 30, value=costs.COST_FN, step=0.5, label="False-negative cost (€)")
            reviewer_error = gr.Slider(0, 0.20, value=costs.REVIEWER_ERROR, step=0.01,
                                       label="Reviewer error rate (share of reviewed fruit decided wrongly)")
        summary = gr.Markdown()
        with gr.Row():
            outcomes = gr.Dataframe(label="Confusion matrix with review, per 1,000 fruits at this base rate")
            alternatives = gr.Dataframe(label="Against the alternatives")
        curve = gr.Plot(label="Cost against threshold")

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Upload a fruit photo")
            image_out = gr.Image(label="Decision per fruit")
        detections = gr.Dataframe(label="Fruits found")
        image_in.change(predict_image, [image_in, t_low, t_high], [image_out, detections])

    with gr.Tab("Per-class results"):
        gr.Markdown("Finetuned model on the test split (Ultralytics validation).")
        gr.Dataframe(per_class_table(), label="Per class: precision, recall, AP50, AP50-95")
        gr.Dataframe(baseline_table(), label="Against the baselines, same test split")
        gr.Dataframe(per_fruit_table(), label=f"Rotten vs fresh per fruit type at threshold {CHOSEN_T:.2f}")
        gr.Plot(confusion_figure(), label="Confusion matrix (counts)")

    with gr.Tab("Dataset"):
        gr.Markdown(f"Roboflow `{DATASET_STATS['source']}`: instances per class and split.")
        gr.Dataframe(dataset_table())

    operating_inputs = [t_low, t_high, base_rate, cost_fn, reviewer_error]
    operating_outputs = [summary, outcomes, alternatives, curve]
    for control in operating_inputs:
        control.change(update_operating, operating_inputs, operating_outputs)
    for control in (t_low, t_high):
        control.change(predict_image, [image_in, t_low, t_high], [image_out, detections])
    demo.load(update_operating, operating_inputs, operating_outputs)


if __name__ == "__main__":
    demo.launch()
