# 🔭 Exoplanet False-Positive Classifier

**Is that dip in starlight a planet, or a false alarm? A machine learning app that vets Kepler planet candidates in seconds.**

[![Live demo](https://img.shields.io/badge/Live%20demo-Streamlit-FF4B4B?logo=streamlit)](https://ai-classifies-fp-transit-exoplanets.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Data-NASA%20Exoplanet%20Archive-0B3D91)](https://exoplanetarchive.ipac.caltech.edu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

👉 **[Try the live app](https://ai-classifies-fp-transit-exoplanets.streamlit.app/)**

---

## 💡 The problem

When a planet passes in front of its star, the star dims a little. NASA's Kepler telescope found thousands of these dips, but many are **false positives**: two stars eclipsing each other, a background star, or plain noise. Astronomers vet each candidate by hand, which is slow.

This app learns from about 6,000 Kepler objects whose status is already known, and gives any new signal a **false-positive probability** in under a second.

## ✨ What you can do

- **🔍 Check a signal**: start from a real Kepler object (a hot Jupiter, a grazing eclipsing binary…) or type your own values, and get an instant verdict with a probability.
- **⚠️ See the red flags**: the app also lists the classic physical warning signs an astronomer would check, such as a grazing transit or an oversized companion.
- **📂 Batch check**: upload a table downloaded from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) as it is, and download it back with a probability for every object. Useful for ranking unvetted *CANDIDATE* objects.
- **📊 Check the model**: honest cross-validated metrics, confusion matrix, ROC curve and feature importance.

## 📈 Results

Measured with 5-fold stratified cross-validation, so every object is scored by a model that never saw it:

| Metric | Score |
|---|---|
| Accuracy | **90.1%** (baseline of always guessing "false positive": 72.5%) |
| Balanced accuracy | **89.8%** |
| ROC AUC | **0.961** |
| Planets correctly kept | 89.0% |
| False positives correctly caught | 90.6% |

**Why the data cleaning matters:** the raw file has 791 exact duplicate rows, mostly confirmed planets, plus 58 rows with identical measurements but different labels. Left in, copies of the same object end up in both the training and test sets and make the model look better than it is. They are removed before training and evaluation.

## 🧠 How it works

```
NASA KOI table ──► clean (drop duplicates & conflicting labels)
                        │
                        ▼
   9 transit + star measurements  ──►  + grazing indicator (b + Rp/R*)
                        │
                        ▼
        Random forest (300 trees, class-balanced)
                        │
                        ▼
          False-positive probability (0 to 1)
```

**Features**: all are standard columns in the NASA Exoplanet Archive KOI table, so archive data can be scored directly.

| Column | Meaning |
|---|---|
| `koi_period` | Orbital period [days] |
| `koi_duration` | Transit duration [hours] |
| `koi_depth` | Transit depth [ppm] |
| `koi_impact` | Impact parameter *b* |
| `koi_ror` | Planet / star radius ratio |
| `koi_dor` | Orbit distance / star radius |
| `koi_srad`, `koi_smass`, `koi_slogg` | Stellar radius, mass, surface gravity |
| `transit_shape` *(derived)* | *b* + Rp/R\*. Above 1 the transit is **grazing**, a classic sign of an eclipsing binary. 42% of false positives are grazing, against 0.6% of planets. |

The most useful signals are the radius ratio, the orbit distance and the grazing indicator. The model essentially learns what an astronomer looks for: companions too big to be planets, and geometry that doesn't fit a planet orbiting that star.

## 🚀 Run it locally

```bash
git clone https://github.com/saeedm31/AI-Classifies-FP-Transit-Exoplanets.git
cd AI-Classifies-FP-Transit-Exoplanets
pip install -r requirements.txt

python train.py          # optional: re-run the evaluation, writes metrics.json
streamlit run app.py     # opens http://localhost:8501
```

## 📁 Project structure

```
├── app.py          # Streamlit app: single check, batch check, performance, data
├── exoclf.py       # Data cleaning, features and model (shared by app and training)
├── train.py        # Cross-validated evaluation → metrics.json
├── metrics.json    # Saved evaluation results shown in the app
├── FP_new.csv      # Labelled Kepler objects (Confirmed / FP)
└── requirements.txt
```

## ⚠️ Limitations

- Trained only on Kepler data. Other missions (TESS) have different noise and may need retraining.
- Uses the fitted transit parameters, not the light curve itself, so it can't spot every type of false positive (for example, a shift in the star's position during transit).
- A screening tool to rank candidates, not a replacement for follow-up observations.

## 🗺️ Next steps

- Add uncertainty-aware features and the KOI false-positive flags for comparison
- Test on TESS Objects of Interest
- Explain single predictions with SHAP values

---

Built by [Saeed Hojjatpanah](https://github.com/saeedm31): former astronomer, now an ML engineer. Data: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/), Kepler Objects of Interest.
