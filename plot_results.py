"""Draw the results chart used in the README (docs/results.svg).

Usage:
    python plot_results.py

Left: the model's false-positive probability for every Kepler object, split by
its true status, using out-of-fold predictions (each object scored by a model
that never saw it, same as train.py). Right: the ROC curve.
Plain SVG, no plotting library needed.
"""

import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import exoclf

W, H = 900, 380
TOP, BOTTOM = 96, 46
BINS = 20

STYLE = """
  .bg { fill: #fcfcfb; } .grid { stroke: #e4e3df; stroke-width: 1; } .base { stroke: #b9b8b2; stroke-width: 1; }
  .axis { fill: #52514e; font: 12px system-ui, -apple-system, sans-serif; }
  .title { fill: #0b0b0b; font: 600 15px system-ui, -apple-system, sans-serif; }
  .sub { fill: #0b0b0b; font: 600 13px system-ui, -apple-system, sans-serif; }
  .planet { fill: #2a78d6; } .fp { fill: #eb6834; } .roc { fill: none; stroke: #2a78d6; stroke-width: 2.5; }
  .chance { stroke: #b9b8b2; stroke-width: 1.5; stroke-dasharray: 5 4; }
  .cut { stroke: #52514e; stroke-width: 1.5; stroke-dasharray: 3 3; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; } .grid { stroke: #34332f; } .base { stroke: #5b5a55; } .axis { fill: #c3c2b7; }
    .title, .sub { fill: #f0efec; } .planet { fill: #3987e5; } .fp { fill: #d95926; } .roc { stroke: #3987e5; }
    .chance { stroke: #5b5a55; } .cut { stroke: #c3c2b7; }
  }
"""


def out_of_fold_probabilities():
    df, _ = exoclf.load_clean_data()
    X = df[exoclf.MODEL_FEATURES]
    y = (df[exoclf.LABEL] == exoclf.POSITIVE).astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=exoclf.RANDOM_STATE)
    proba = cross_val_predict(exoclf.build_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    return y, proba


def histogram_panel(y, proba, x0, x1):
    edges = np.linspace(0, 1, BINS + 1)
    share = {c: 100 * np.histogram(proba[y == c], edges)[0] / (y == c).sum() for c in (0, 1)}
    top = np.ceil(max(share[0].max(), share[1].max()) / 10) * 10
    y0, y1 = H - BOTTOM, TOP

    def py(v):
        return y0 - v * (y0 - y1) / top

    step = (x1 - x0) / BINS
    bar = (step - 4) / 2
    out = [f'<text class="sub" x="{x0}" y="{TOP - 14}">Share of each group, by model score (out-of-fold)</text>']
    for v in range(0, int(top) + 1, 10):
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{py(v):.0f}" y2="{py(v):.0f}"/>'
                   f'<text class="axis" x="{x0 - 6}" y="{py(v) + 4:.0f}" text-anchor="end">{v}%</text>')
    for i in range(BINS):
        for k, (c, cls) in enumerate(((0, "planet"), (1, "fp"))):
            h = share[c][i]
            if h > 0:
                x = x0 + i * step + 1 + k * (bar + 2)
                out.append(f'<rect class="{cls}" x="{x:.1f}" y="{py(h):.1f}" width="{bar:.1f}" '
                           f'height="{y0 - py(h):.1f}" rx="1.5"/>')
    cut = x0 + 0.5 * (x1 - x0)
    out.append(f'<line class="cut" x1="{cut:.0f}" x2="{cut:.0f}" y1="{y1 - 4}" y2="{y0}"/>'
               f'<text class="axis" x="{cut + 6:.0f}" y="{y1 + 8}">decision line</text>')
    for v in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<text class="axis" x="{x0 + v * (x1 - x0):.0f}" y="{y0 + 18}" text-anchor="middle">{v:g}</text>')
    out.append(f'<text class="axis" x="{(x0 + x1) / 2:.0f}" y="{y0 + 36}" text-anchor="middle">'
               f'False-positive probability</text>')
    return out


def roc_panel(y, proba, x0, x1):
    fpr, tpr, _ = roc_curve(y, proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))
    auc = roc_auc_score(y, proba)
    y0, y1 = H - BOTTOM, TOP

    def px(v):
        return x0 + v * (x1 - x0)

    def py(v):
        return y0 - v * (y0 - y1)

    out = [f'<text class="sub" x="{x0}" y="{TOP - 14}">ROC curve (AUC {auc:.2f})</text>']
    for v in (0, 0.5, 1):
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{py(v):.0f}" y2="{py(v):.0f}"/>'
                   f'<text class="axis" x="{x0 - 6}" y="{py(v) + 4:.0f}" text-anchor="end">{v:g}</text>'
                   f'<text class="axis" x="{px(v):.0f}" y="{y0 + 18}" text-anchor="middle">{v:g}</text>')
    out.append(f'<line class="chance" x1="{px(0):.0f}" y1="{py(0):.0f}" x2="{px(1):.0f}" y2="{py(1):.0f}"/>')
    pts = " ".join(f"{px(a):.0f},{py(b):.0f}" for a, b in zip(fpr[keep], tpr[keep]))
    out.append(f'<polyline class="roc" points="{pts}"/>')
    out.append(f'<text class="axis" x="{(x0 + x1) / 2:.0f}" y="{y0 + 36}" text-anchor="middle">'
               f'Planets wrongly flagged</text>')
    out.append(f'<text class="axis" x="{px(0.42):.0f}" y="{py(0.32):.0f}">random guessing</text>')
    return out, auc


def main(out="docs/results.svg"):
    y, proba = out_of_fold_probabilities()
    left = histogram_panel(y, proba, 58, 560)
    right, auc = roc_panel(y, proba, 640, 880)
    caught = 100 * (proba[y == 1] >= 0.5).mean()
    kept = 100 * (proba[y == 0] < 0.5).mean()
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
  aria-label="Model scores for confirmed planets and false positives, and the ROC curve">
<style>{STYLE}</style>
<rect class="bg" width="{W}" height="{H}" rx="8"/>
<text class="title" x="58" y="24">Real planets score low, false alarms score high</text>
<rect class="planet" x="58" y="40" width="14" height="10" rx="2"/>
<text class="axis" x="78" y="50">Confirmed planets ({kept:.0f}% kept)</text>
<rect class="fp" x="290" y="40" width="14" height="10" rx="2"/>
<text class="axis" x="310" y="50">False positives ({caught:.0f}% caught)</text>
{"".join(left)}
{"".join(right)}
<text class="axis" x="604" y="{(TOP + H - BOTTOM) / 2:.0f}" text-anchor="middle"
  transform="rotate(-90 604 {(TOP + H - BOTTOM) / 2:.0f})">False positives caught</text>
</svg>
"""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(svg)
    print(f"Wrote {out}: AUC {auc:.3f}, {kept:.1f}% planets kept, {caught:.1f}% false positives caught")


if __name__ == "__main__":
    main()
