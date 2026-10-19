import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import exoclf

st.set_page_config(page_title="Exoplanet False-Positive Classifier", page_icon="🔭", layout="wide")

PLANET_COLOR = "#2a9d8f"
FP_COLOR = "#e76f51"
ARCHIVE_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative"


# ---------- data and model (cached, built once per server) ----------

@st.cache_data
def load_data():
    return exoclf.load_clean_data()


@st.cache_resource
def load_model():
    df, _ = load_data()
    return exoclf.train(df)


@st.cache_data
def load_metrics():
    if not os.path.exists("metrics.json"):
        return None
    with open("metrics.json") as f:
        return json.load(f)


df, data_stats = load_data()
model = load_model()
metrics = load_metrics()

# Real Kepler objects from the training data, picked to show typical cases.
EXAMPLES = {
    "Small confirmed planet (10-day orbit)": dict(
        koi_period=9.9315, koi_duration=5.678, koi_depth=398.6, koi_impact=0.037, koi_ror=0.01809,
        koi_dor=13.607, koi_srad=1.22, koi_smass=1.035, koi_slogg=4.28, truth="Confirmed planet"),
    "Hot Jupiter (2.8-day orbit)": dict(
        koi_period=2.8165, koi_duration=2.506, koi_depth=16814.0, koi_impact=0.105, koi_ror=0.11706,
        koi_dor=9.571, koi_srad=0.955, koi_smass=0.941, koi_slogg=4.452, truth="Confirmed planet"),
    "Grazing eclipsing binary": dict(
        koi_period=8.4604, koi_duration=4.735, koi_depth=76413.0, koi_impact=1.061, koi_ror=0.47764,
        koi_dor=14.087, koi_srad=0.702, koi_smass=0.796, koi_slogg=4.638, truth="False positive"),
    "Tricky case: tiny, shallow signal": dict(
        koi_period=6.1260, koi_duration=1.733, koi_depth=40.2, koi_impact=0.422, koi_ror=0.00589,
        koi_dor=24.7, koi_srad=1.286, koi_smass=0.844, koi_slogg=4.146, truth="False positive"),
}
CUSTOM = "Enter my own values"

INPUT_FORMAT = {
    "koi_period": (0.01, "%.4f"), "koi_duration": (0.1, "%.3f"), "koi_depth": (10.0, "%.1f"),
    "koi_impact": (0.01, "%.3f"), "koi_ror": (0.001, "%.5f"), "koi_dor": (0.1, "%.3f"),
    "koi_srad": (0.01, "%.3f"), "koi_smass": (0.01, "%.3f"), "koi_slogg": (0.01, "%.3f"),
}
INPUT_GROUPS = {
    "🌑 Transit signal": ["koi_period", "koi_duration", "koi_depth"],
    "🪐 Geometry": ["koi_impact", "koi_ror", "koi_dor"],
    "☀️ Host star": ["koi_srad", "koi_smass", "koi_slogg"],
}


def label_for(feature):
    name, unit = exoclf.FEATURE_INFO[feature]
    return f"{name} [{unit}]" if unit else name


def verdict(p_fp):
    """Turn a false-positive probability into a short, honest verdict."""
    if p_fp >= 0.65:
        return "error", "Likely a false positive", "The signal looks more like an eclipsing binary or noise than a planet."
    if p_fp <= 0.35:
        return "success", "Likely a real planet", "The signal looks like a planet passing in front of its star."
    return "warning", "Uncertain", "The model can't call this one. It needs follow-up observations."


# ---------- header ----------

st.title("🔭 Is it a planet, or a false alarm?")
st.markdown(
    "NASA's Kepler telescope found thousands of dips in starlight that *could* be planets. "
    "Many turn out to be **false positives**: eclipsing binary stars, background stars or noise. "
    "This app uses machine learning to tell them apart from the transit and star measurements alone."
)

if metrics:
    m = metrics["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.1%}", f"{m['accuracy'] - m['baseline_accuracy']:+.1%} vs. guessing",
              help="Share of objects classified correctly, measured on objects the model never saw "
                   "(5-fold cross-validation). Guessing 'false positive' every time would score "
                   f"{m['baseline_accuracy']:.1%}.")
    c2.metric("Balanced accuracy", f"{m['balanced_accuracy']:.1%}",
              help="Average of the accuracy on planets and on false positives, so the bigger class can't hide mistakes.")
    c3.metric("ROC AUC", f"{m['roc_auc']:.3f}",
              help="Chance that a random false positive gets a higher score than a random planet (1.0 = perfect).")
    c4.metric("Kepler objects", f"{data_stats['clean_rows']:,}", help="Labelled objects the model is trained on.")

tab_check, tab_batch, tab_perf, tab_about = st.tabs(
    ["🔍 Check a signal", "📂 Batch check", "📊 Model performance", "🌌 About the data"])

# ---------- tab 1: single prediction ----------

with tab_check:
    choice = st.selectbox("Start from a real Kepler object, or enter your own values",
                          list(EXAMPLES) + [CUSTOM])
    start = EXAMPLES.get(choice, EXAMPLES["Small confirmed planet (10-day orbit)"])

    values = {}
    cols = st.columns(len(INPUT_GROUPS))
    for col, (group, feats) in zip(cols, INPUT_GROUPS.items()):
        with col:
            st.markdown(f"**{group}**")
            for f in feats:
                step, fmt = INPUT_FORMAT[f]
                values[f] = st.number_input(label_for(f), min_value=0.0, value=float(start[f]),
                                            step=step, format=fmt, key=f"{choice}-{f}")

    row = pd.DataFrame([values])
    p_fp = float(exoclf.predict_fp_probability(model, row)[0])
    kind, title, explanation = verdict(p_fp)

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        getattr(st, kind)(f"**{title}**: {explanation}")
        st.markdown(f"False-positive probability: **{p_fp:.0%}**")
        st.progress(p_fp)
        st.caption("0% = certainly a planet · 100% = certainly a false positive")
        if choice in EXAMPLES:
            st.caption(f"Kepler's final verdict for this object: **{start['truth']}**. "
                       "This object is part of the training data, so the model has seen it before.")
    with right:
        flags = exoclf.red_flags(values)
        st.markdown("**Physical red flags**")
        if flags:
            for flag in flags:
                st.markdown(f"- ⚠️ {flag}")
        else:
            st.markdown("- ✅ None of the classic warning signs (grazing, oversized, very deep).")
        st.caption("Rules of thumb an astronomer checks by eye. The model learns its own, subtler patterns.")

# ---------- tab 2: batch prediction ----------

with tab_batch:
    st.markdown(
        "Score many objects at once. Upload a CSV with the columns below. A table downloaded from the "
        f"[NASA Exoplanet Archive KOI table]({ARCHIVE_URL}) works as it is, comment lines included."
    )
    st.code(", ".join(exoclf.INPUT_FEATURES), language=None)
    st.markdown("💡 **Try this:** download the KOI table, upload it, and sort the *CANDIDATE* objects by "
                "false-positive probability to see which candidates are most likely real.")

    sample = df.sample(20, random_state=1)[exoclf.INPUT_FEATURES + [exoclf.LABEL]]
    st.download_button("Download a sample file (20 Kepler objects)", sample.to_csv(index=False),
                       file_name="sample_kepler_objects.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded, comment="#")
        except Exception as e:
            st.error(f"Could not read this file as CSV: {e}")
            data = None

        if data is not None:
            missing = [c for c in exoclf.INPUT_FEATURES if c not in data.columns]
            if missing:
                st.error("These required columns are missing: " + ", ".join(missing))
            else:
                complete = data[exoclf.INPUT_FEATURES].notna().all(axis=1)
                scored = data[complete].copy()
                if len(scored) == 0:
                    st.error("No rows have all the required values.")
                else:
                    scored.insert(0, "fp_probability", exoclf.predict_fp_probability(model, scored).round(3))
                    scored.insert(1, "prediction", np.where(scored["fp_probability"] >= 0.5,
                                                            "False positive", "Planet"))
                    scored = scored.sort_values("fp_probability")

                    a, b, c = st.columns(3)
                    a.metric("Objects scored", f"{len(scored):,}")
                    b.metric("Likely planets", f"{(scored['prediction'] == 'Planet').sum():,}")
                    c.metric("Likely false positives", f"{(scored['prediction'] == 'False positive').sum():,}")
                    if (~complete).sum():
                        st.caption(f"Skipped {(~complete).sum():,} rows with missing values.")

                    if "koi_disposition" in scored.columns:
                        st.markdown("**Model vs. NASA disposition**")
                        st.dataframe(pd.crosstab(scored["koi_disposition"], scored["prediction"]))

                    id_cols = [c for c in ["kepoi_name", "kepler_name", "kepid", "koi_disposition", exoclf.LABEL]
                               if c in scored.columns]
                    st.dataframe(scored[["fp_probability", "prediction"] + id_cols + exoclf.INPUT_FEATURES],
                                 hide_index=True)
                    st.download_button("Download results as CSV", scored.to_csv(index=False),
                                       file_name="fp_predictions.csv", mime="text/csv")

# ---------- tab 3: model performance ----------

with tab_perf:
    if metrics is None:
        st.info("Run `python train.py` to compute the performance metrics.")
    else:
        m = metrics["metrics"]
        st.markdown(f"Measured with **{metrics['validation']}**: every object is scored by a model "
                    "that never saw it during training.")

        st.dataframe(pd.DataFrame({
            "Metric": ["Accuracy", "Balanced accuracy", "ROC AUC", "Precision (false positives)",
                       "Recall (false positives)", "Recall (planets)", "Baseline: always guess FP"],
            "Value": [f"{v:.3f}" for v in (m["accuracy"], m["balanced_accuracy"], m["roc_auc"],
                                           m["precision_fp"], m["recall_fp"], m["recall_confirmed"],
                                           m["baseline_accuracy"])],
        }), hide_index=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4, 3.6))
            ax.imshow(cm, cmap="Blues")
            names = ["Planet", "False pos."]
            ax.set_xticks([0, 1], names)
            ax.set_yticks([0, 1], names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
            ax.set_title("Confusion matrix")
            fig.tight_layout()
            st.pyplot(fig)

        with c2:
            roc = metrics["roc_curve"]
            fig, ax = plt.subplots(figsize=(4, 3.6))
            ax.plot(roc["fpr"], roc["tpr"], color=FP_COLOR, lw=2, label=f"AUC = {m['roc_auc']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random guess")
            ax.set_xlabel("Planets wrongly flagged")
            ax.set_ylabel("False positives caught")
            ax.set_title("ROC curve")
            ax.legend(loc="lower right", frameon=False)
            fig.tight_layout()
            st.pyplot(fig)

        with c3:
            imp = pd.Series(metrics["feature_importance"]).sort_values()
            fig, ax = plt.subplots(figsize=(4, 3.6))
            ax.barh([exoclf.FEATURE_INFO[f][0] for f in imp.index], imp.values, color=PLANET_COLOR)
            ax.set_xlabel("Importance")
            ax.set_title("What the model relies on")
            ax.tick_params(axis="y", labelsize=8)
            fig.tight_layout()
            st.pyplot(fig)

        d = metrics["data"]
        st.caption(f"Data cleaning: started from {d['raw_rows']:,} rows, removed {d['removed_duplicates']:,} "
                   f"exact duplicates and {d['removed_conflicting']:,} rows with identical measurements but "
                   f"different labels, leaving {d['clean_rows']:,} objects. Duplicates would otherwise "
                   "appear in both training and test data and inflate the scores.")

# ---------- tab 4: about the data ----------

with tab_about:
    st.markdown("""
#### How planets are found, and how they're faked
When a planet passes in front of its star, the star dims slightly. Kepler watched about 150,000 stars
for four years looking for these dips, called **transits**. But other things make similar dips:

- **Eclipsing binaries**: two stars orbiting each other. Their eclipses are often deep, or *grazing*,
  where one star only clips the edge of the other.
- **Background binaries**: an eclipsing pair behind the target star, diluted to look planet-sized.
- **Noise and instrument effects**: tiny, shallow signals that are not real.

Astronomers vet each signal by hand. This model learns the same patterns from about 6,000 objects whose
status is already known, so it can rank new candidates in seconds.
""")

    st.markdown("#### Features the model uses")
    st.dataframe(pd.DataFrame([
        {"Column": f, "Meaning": exoclf.FEATURE_INFO[f][0], "Unit": exoclf.FEATURE_INFO[f][1] or "–"}
        for f in exoclf.MODEL_FEATURES
    ]), hide_index=True)
    st.caption("All columns except the grazing indicator come straight from the NASA Exoplanet Archive "
               "KOI table. The grazing indicator is computed as impact parameter + radius ratio.")

    st.markdown("#### How planets and false positives differ")
    show = ["koi_period", "koi_depth", "koi_ror", "transit_shape"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, f in zip(axes, show):
        values_all = df[f][df[f] > 0]
        bins = np.logspace(np.log10(values_all.quantile(0.005)), np.log10(values_all.quantile(0.995)), 40)
        for label, color, name in [("Confirmed", PLANET_COLOR, "Planets"), ("FP", FP_COLOR, "False positives")]:
            vals = df.loc[df[exoclf.LABEL] == label, f]
            # share of each class per bin, so the two classes are comparable despite different sizes
            ax.hist(vals, bins=bins, color=color, alpha=0.6, label=name,
                    weights=np.full(len(vals), 1 / len(vals)))
        ax.set_xscale("log")
        ax.set_title(exoclf.FEATURE_INFO[f][0], fontsize=11)
        ax.set_yticks([])
    axes[0].legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig)
    is_fp = df[exoclf.LABEL] == "FP"
    grazing = df["transit_shape"] > 1
    st.caption("Share of each class per bin. Many false positives form a second group of deep, oversized signals "
               f"from eclipsing binaries. {grazing[is_fp].mean():.0%} of false positives are grazing "
               f"(indicator above 1), against {grazing[~is_fp].mean():.1%} of planets.")

    counts = df[exoclf.LABEL].value_counts()
    st.caption(f"Training data: {counts.get('Confirmed', 0):,} confirmed planets and "
               f"{counts.get('FP', 0):,} false positives from the Kepler mission.")

st.divider()
st.caption("Built by [Saeed Hojjatpanah](https://github.com/saeedm31) · "
           "[Source code](https://github.com/saeedm31/AI-Classifies-FP-Transit-Exoplanets) · "
           "Data: NASA Exoplanet Archive, Kepler Objects of Interest")
