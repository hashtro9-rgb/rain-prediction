"""
Predicting Rain from Weather Conditions - a classification study.
=================================================================

Reproducible, top-to-bottom analysis script.

Run from the project root:
    python analysis/rain_analysis.py

Pipeline:
    1. Load + profile the data (2,500 rows, 5 numeric features, imbalanced target)
    2. EDA: class balance, feature distributions by class, correlation, decision rule
    3. Models: Logistic Regression (scaled) + Random Forest, both class_weight="balanced"
    4. Evaluation on a stratified held-out test set (ROC-AUC, precision, recall, F1, CM)
    5. Save every chart to charts/, dump aggregates to analysis/dashboard_data.json,
       and write analysis/FINDINGS.md

The central challenge is CLASS IMBALANCE: only ~12.6% of rows are "rain", so a model
that always says "no rain" scores 87.4% accuracy yet is useless. We therefore evaluate
on ROC-AUC, precision, recall and F1 for the positive (rain) class - never accuracy alone.

Everything uses random_state=42 for reproducibility.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
)

# --------------------------------------------------------------------------------------
# Config & house style
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
SEED = RANDOM_STATE
np.random.seed(SEED)

# Resolve project root relative to this file so the script is location-independent.
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "weather_forecast_data.csv"
CHARTS_DIR = ROOT / "charts"
ANALYSIS_DIR = ROOT / "analysis"
CHARTS_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

SOURCE_NOTE = "Data: weather_forecast_data.csv"

# Rainfall-blues palette
C_PURPLE = "#0F3A5F"   # navy (darkest / negative direction)
C_PURPLE2 = "#1E6FA8"  # ocean blue
C_PINK = "#2E9BD6"     # sky blue (primary accent / rain)
C_PINK2 = "#38C6E0"    # cyan (bright accent)
C_LILAC = "#8FC7EA"    # pale sky (light fill)
C_CANVAS = "#EFF6FB"
C_INK = "#103049"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "Segoe UI, DejaVu Sans, sans-serif",
        "axes.edgecolor": "#C7DBEA",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#E1EDF6",
        "grid.linewidth": 0.8,
        "axes.titleweight": "bold",
        "axes.titlecolor": C_INK,
        "text.color": C_INK,
        "axes.labelcolor": C_INK,
        "xtick.color": "#5B7A93",
        "ytick.color": "#5B7A93",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

PURPLE_PINK_CMAP = LinearSegmentedColormap.from_list(
    "purple_pink", [C_PURPLE, C_PURPLE2, C_LILAC, C_PINK2]
)


def _source(ax):
    """Add the standard source note under an axes."""
    ax.annotate(
        SOURCE_NOTE,
        xy=(0, 0),
        xycoords="figure fraction",
        xytext=(0.008, 0.012),
        textcoords="figure fraction",
        fontsize=7.5,
        color="#5B7A93",
        style="italic",
    )


def save(fig, name):
    path = CHARTS_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved chart -> {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------------------
# 1. Load + profile
# --------------------------------------------------------------------------------------
print("[1/6] Loading + profiling data ...")
df = pd.read_csv(DATA_PATH)
FEATURES = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]
UNITS = {
    "Temperature": "°C",
    "Humidity": "%",
    "Wind_Speed": "m/s",
    "Cloud_Cover": "%",
    "Pressure": "hPa",
}

n_rows = len(df)
assert df.isnull().sum().sum() == 0, "unexpected missing values"

# Binary target: 1 = rain (positive), 0 = no rain
df["y"] = (df["Rain"].str.strip().str.lower() == "rain").astype(int)
n_rain = int(df["y"].sum())
n_norain = int((df["y"] == 0).sum())
pct_rain = n_rain / n_rows * 100
naive_acc = n_norain / n_rows * 100  # always-predict-"no rain" accuracy trap

print(f"  rows={n_rows}  rain={n_rain}  no_rain={n_norain}  %rain={pct_rain:.1f}")
print(f"  naive 'always no-rain' accuracy trap = {naive_acc:.1f}%")

# --------------------------------------------------------------------------------------
# 2. EDA
# --------------------------------------------------------------------------------------
print("[2/6] Running EDA ...")

# --- 2a. Class balance (donut) ---
fig, ax = plt.subplots(figsize=(6.2, 5.0))
wedges, _ = ax.pie(
    [n_norain, n_rain],
    colors=[C_LILAC, C_PINK],
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
)
ax.text(0, 0.12, f"{pct_rain:.1f}%", ha="center", va="center",
        fontsize=30, fontweight="bold", color=C_PINK)
ax.text(0, -0.22, "of days rain", ha="center", va="center", fontsize=12, color="#5B7A93")
ax.legend(
    wedges,
    [f"No rain  ({n_norain:,})", f"Rain  ({n_rain:,})"],
    loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False,
)
ax.set_title("Class balance: rain is the rare event\n"
             f"A model that always says 'no rain' scores {naive_acc:.1f}% accuracy - and catches zero storms",
             fontsize=12.5, pad=16)
ax.set_aspect("equal")
_source(ax)
save(fig, "01_class_balance.png")

# --- 2b. Feature distributions split by class + group means ---
group_means = df.groupby("y")[FEATURES].mean()  # rows: 0 no-rain, 1 rain
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()
for i, feat in enumerate(FEATURES):
    ax = axes[i]
    lo = df[feat].min()
    hi = df[feat].max()
    bins = np.linspace(lo, hi, 30)
    ax.hist(df.loc[df.y == 0, feat], bins=bins, density=True, color=C_LILAC,
            alpha=0.65, label="No rain")
    ax.hist(df.loc[df.y == 1, feat], bins=bins, density=True, color=C_PINK,
            alpha=0.60, label="Rain")
    m0, m1 = group_means.loc[0, feat], group_means.loc[1, feat]
    ax.axvline(m0, color=C_PURPLE, lw=1.6, ls="--")
    ax.axvline(m1, color=C_PINK2, lw=1.8, ls="--")
    ax.set_title(f"{feat}  ({UNITS[feat]})\n"
                 f"mean: no-rain {m0:.1f} vs rain {m1:.1f}", fontsize=10.5)
    ax.set_ylabel("density", fontsize=9)
    if i == 0:
        ax.legend(frameon=False, fontsize=9)
# hide the empty 6th panel and use it for a takeaway
axes[5].axis("off")
axes[5].text(
    0.02, 0.95,
    "How rainy days differ\n(rain mean minus no-rain mean):",
    fontsize=11, fontweight="bold", color=C_INK, va="top",
)
lines = []
for feat in FEATURES:
    d = group_means.loc[1, feat] - group_means.loc[0, feat]
    arrow = "▲" if d > 0 else "▼"
    lines.append(f"{arrow} {feat:<12} {d:+.1f} {UNITS[feat]}")
axes[5].text(0.02, 0.72, "\n".join(lines), fontsize=11, family="monospace",
             color="#103049", va="top")
fig.suptitle("What does a rainy day look like? Feature distributions by outcome",
             fontsize=14, fontweight="bold", color=C_INK, y=0.99)
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
_source(axes[3])
save(fig, "02_feature_distributions.png")

# --- 2c. Correlation heatmap (features + rain 0/1) ---
corr = df[FEATURES + ["y"]].rename(columns={"y": "Rain(0/1)"}).corr()
fig, ax = plt.subplots(figsize=(7.6, 6.6))
im = ax.imshow(corr.values, cmap=PURPLE_PINK_CMAP, vmin=-1, vmax=1)
labels = corr.columns.tolist()
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.5 else C_INK, fontsize=9)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r", fontsize=9)
ax.set_title("Feature correlations - what moves with rain?\n"
             "Humidity & Cloud Cover lead; features are largely independent (low multicollinearity)",
             fontsize=11.5, pad=12)
ax.grid(False)
_source(ax)
save(fig, "03_correlation_heatmap.png")

# rain correlations (drop self)
rain_corr = corr["Rain(0/1)"].drop("Rain(0/1)").sort_values(ascending=False)

# --- 2d. Decision-rule 2D bucket table: rain rate by Humidity x Cloud_Cover ---
def bucket(s, edges):
    return pd.cut(s, bins=edges, include_lowest=True)

hum_edges = [30, 55, 75, 100]
cld_edges = [0, 33, 66, 100]
hum_labels = ["Low (30-55)", "Med (55-75)", "High (75-100)"]
cld_labels = ["Low (0-33)", "Med (33-66)", "High (66-100)"]
df["_hum_b"] = pd.cut(df["Humidity"], bins=hum_edges, labels=hum_labels, include_lowest=True)
df["_cld_b"] = pd.cut(df["Cloud_Cover"], bins=cld_edges, labels=cld_labels, include_lowest=True)
rain_rate = (
    df.pivot_table(index="_hum_b", columns="_cld_b", values="y", aggfunc="mean", observed=False) * 100
)
rain_count = df.pivot_table(index="_hum_b", columns="_cld_b", values="y", aggfunc="size", observed=False)
rain_rate = rain_rate.reindex(index=hum_labels, columns=cld_labels)
rain_count = rain_count.reindex(index=hum_labels, columns=cld_labels)

fig, ax = plt.subplots(figsize=(8.2, 6.2))
im = ax.imshow(rain_rate.values, cmap=PURPLE_PINK_CMAP, vmin=0, vmax=np.nanmax(rain_rate.values))
ax.set_xticks(range(len(cld_labels)))
ax.set_yticks(range(len(hum_labels)))
ax.set_xticklabels(cld_labels, fontsize=9.5)
ax.set_yticklabels(hum_labels, fontsize=9.5)
ax.set_xlabel("Cloud Cover", fontsize=10, fontweight="bold")
ax.set_ylabel("Humidity", fontsize=10, fontweight="bold")
for i in range(len(hum_labels)):
    for j in range(len(cld_labels)):
        rr = rain_rate.values[i, j]
        nn = rain_count.values[i, j]
        ax.text(j, i, f"{rr:.0f}%\n(n={int(nn)})", ha="center", va="center",
                color="white" if rr > np.nanmax(rain_rate.values) * 0.5 else C_INK,
                fontsize=10.5, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Rain rate (%)", fontsize=9)
lo_lo = rain_rate.loc["Low (30-55)", "Low (0-33)"]
hi_hi = rain_rate.loc["High (75-100)", "High (66-100)"]
ax.set_title("Simple rule: rain probability by Humidity x Cloud Cover\n"
             f"Dry+clear = {lo_lo:.0f}% rain  vs  humid+cloudy = {hi_hi:.0f}% rain",
             fontsize=11.5, pad=12)
ax.grid(False)
_source(ax)
save(fig, "04_rainrate_humidity_cloud.png")

# --------------------------------------------------------------------------------------
# 3. Modeling
# --------------------------------------------------------------------------------------
print("[3/6] Training models ...")
X = df[FEATURES].values
y = df["y"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print(f"  train={len(y_train)} (rain={int(y_train.sum())})  "
      f"test={len(y_test)} (rain={int(y_test.sum())})")

# Logistic Regression - scaled via Pipeline, interpretable baseline
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=RANDOM_STATE)),
])
logreg.fit(X_train, y_train)

# Random Forest - main model
rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train, y_train)

models = {"Logistic Regression": logreg, "Random Forest": rf}

# --------------------------------------------------------------------------------------
# 4. Evaluation
# --------------------------------------------------------------------------------------
print("[4/6] Evaluating on held-out test set ...")
results = {}
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)  # [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    results[name] = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "proba": proba,
    }
    r = results[name]
    print(f"  {name:<20} AUC={r['roc_auc']:.3f}  P={r['precision']:.3f}  "
          f"R={r['recall']:.3f}  F1={r['f1']:.3f}  acc={r['accuracy']:.3f}")

# Winner by ROC-AUC
winner = max(results, key=lambda k: results[k]["roc_auc"])
print(f"  winner by ROC-AUC: {winner}")

# --- 4a. ROC curves overlaid ---
fig, ax = plt.subplots(figsize=(7.4, 6.4))
roc_series = {}
colors = {"Logistic Regression": C_PURPLE, "Random Forest": C_PINK}
for name in models:
    fpr, tpr, _ = roc_curve(y_test, results[name]["proba"])
    roc_series[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    ax.plot(fpr, tpr, color=colors[name], lw=2.4,
            label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], ls="--", color="#B7CFDE", lw=1.3, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate (Recall on rain)", fontsize=10)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.02)
ax.legend(loc="lower right", frameon=True, fontsize=9.5)
ax.set_title("ROC curves - both models rank rainy days far above chance\n"
             f"Winner by AUC: {winner} ({results[winner]['roc_auc']:.3f})",
             fontsize=11.5, pad=12)
_source(ax)
save(fig, "05_roc_curves.png")

# --- 4b. Confusion matrix for the best model ---
cm_best = results[winner]["confusion_matrix"]
cm_arr = np.array([[cm_best["tn"], cm_best["fp"]], [cm_best["fn"], cm_best["tp"]]])
fig, ax = plt.subplots(figsize=(6.6, 5.8))
im = ax.imshow(cm_arr, cmap=PURPLE_PINK_CMAP)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred: No rain", "Pred: Rain"], fontsize=10)
ax.set_yticklabels(["True: No rain", "True: Rain"], fontsize=10)
cell_labels = [["True negatives", "False alarms"], ["Missed storms", "Caught storms"]]
mx = cm_arr.max()
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm_arr[i, j]}\n{cell_labels[i][j]}", ha="center", va="center",
                color="white" if cm_arr[i, j] > mx * 0.5 else C_INK,
                fontsize=11, fontweight="bold")
ax.set_title(f"Confusion matrix - {winner} (threshold 0.50)\n"
             f"Catches {cm_best['tp']} of {cm_best['tp'] + cm_best['fn']} real rain days; "
             f"{cm_best['fp']} false alarms",
             fontsize=11.5, pad=12)
ax.grid(False)
_source(ax)
save(fig, "06_confusion_matrix.png")

# --- 4c. Precision-Recall curve + threshold table (winner) ---
prec, rec, thr = precision_recall_curve(y_test, results[winner]["proba"])
fig, ax = plt.subplots(figsize=(7.4, 6.2))
ax.plot(rec, prec, color=C_PINK, lw=2.4)
base_rate = y_test.mean()
ax.axhline(base_rate, ls="--", color="#B7CFDE", lw=1.3,
           label=f"No-skill baseline ({base_rate:.3f})")
# mark the default 0.5 threshold operating point
p50 = results[winner]["precision"]; r50 = results[winner]["recall"]
ax.scatter([r50], [p50], color=C_PURPLE, s=90, zorder=5,
           label=f"Threshold 0.50 (P={p50:.2f}, R={r50:.2f})")
ax.set_xlabel("Recall (rain caught)", fontsize=10)
ax.set_ylabel("Precision (alarms that were right)", fontsize=10)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.legend(loc="upper right", frameon=True, fontsize=9)
ax.set_title(f"Precision-Recall tradeoff - {winner}\n"
             "Lowering the threshold catches more storms but raises false alarms",
             fontsize=11.5, pad=12)
_source(ax)
save(fig, "07_precision_recall.png")

# Threshold sweep table on the winner
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_table = []
pw = results[winner]["proba"]
for t in thresholds:
    pr = (pw >= t).astype(int)
    threshold_table.append({
        "threshold": t,
        "precision": float(precision_score(y_test, pr, zero_division=0)),
        "recall": float(recall_score(y_test, pr, zero_division=0)),
        "f1": float(f1_score(y_test, pr, zero_division=0)),
        "predicted_rain": int(pr.sum()),
    })

# --------------------------------------------------------------------------------------
# 5. Drivers: RF importances + logistic coefficients
# --------------------------------------------------------------------------------------
print("[5/6] Extracting drivers ...")
rf_import = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.barh(rf_import.index, rf_import.values, color=C_PINK)
for i, v in enumerate(rf_import.values):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9.5, color=C_INK)
ax.set_xlabel("Importance (mean decrease in impurity)", fontsize=10)
ax.set_xlim(0, rf_import.max() * 1.18)
top_feat = rf_import.index[-1]
ax.set_title(f"Random Forest feature importance\n"
             f"{top_feat} dominates the model's decisions",
             fontsize=11.5, pad=12)
_source(ax)
save(fig, "08_rf_importance.png")

# Logistic coefficients -> odds ratios (on standardized features)
log_clf = logreg.named_steps["clf"]
coefs = pd.Series(log_clf.coef_[0], index=FEATURES).sort_values()
odds = np.exp(coefs)
fig, ax = plt.subplots(figsize=(7.8, 5.2))
bar_colors = [C_PINK if c > 0 else C_PURPLE for c in coefs.values]
ax.barh(coefs.index, coefs.values, color=bar_colors)
for i, (c, o) in enumerate(zip(coefs.values, odds.values)):
    ax.text(c + (0.05 if c >= 0 else -0.05), i, f"{c:+.2f}  (OR {o:.2f})",
            va="center", ha="left" if c >= 0 else "right", fontsize=9, color=C_INK)
ax.axvline(0, color="#5B7A93", lw=1)
ax.set_xlabel("Logistic coefficient (standardized features)  ->  pushes toward RAIN", fontsize=9.5)
pad = max(abs(coefs.min()), abs(coefs.max())) * 0.55
ax.set_xlim(coefs.min() - pad, coefs.max() + pad)
ax.set_title("Logistic Regression coefficients & odds ratios\n"
             "Pink pushes toward rain, purple toward no-rain (per +1 SD)",
             fontsize=11.5, pad=12)
_source(ax)
save(fig, "09_logistic_coefficients.png")

# --------------------------------------------------------------------------------------
# 6. Dump dashboard_data.json + write FINDINGS.md
# --------------------------------------------------------------------------------------
print("[6/6] Writing dashboard_data.json + FINDINGS.md ...")

# distributions for the dashboard control (histogram per feature, per class)
dist_payload = {}
for feat in FEATURES:
    lo, hi = float(df[feat].min()), float(df[feat].max())
    bins = np.linspace(lo, hi, 26)
    centers = ((bins[:-1] + bins[1:]) / 2).round(2).tolist()
    h0, _ = np.histogram(df.loc[df.y == 0, feat], bins=bins, density=True)
    h1, _ = np.histogram(df.loc[df.y == 1, feat], bins=bins, density=True)
    dist_payload[feat] = {
        "unit": UNITS[feat],
        "centers": centers,
        "no_rain": h0.round(5).tolist(),
        "rain": h1.round(5).tolist(),
        "mean_no_rain": float(group_means.loc[0, feat]),
        "mean_rain": float(group_means.loc[1, feat]),
    }

dashboard = {
    "meta": {
        "n_rows": n_rows,
        "n_features": len(FEATURES),
        "n_rain": n_rain,
        "n_no_rain": n_norain,
        "pct_rain": round(pct_rain, 1),
        "naive_accuracy": round(naive_acc, 1),
        "random_state": RANDOM_STATE,
        "test_size": 0.25,
        "features": FEATURES,
        "units": UNITS,
    },
    "class_balance": {"no_rain": n_norain, "rain": n_rain},
    "group_means": {
        feat: {
            "no_rain": float(group_means.loc[0, feat]),
            "rain": float(group_means.loc[1, feat]),
            "delta": float(group_means.loc[1, feat] - group_means.loc[0, feat]),
            "unit": UNITS[feat],
        }
        for feat in FEATURES
    },
    "correlation": {
        "labels": corr.columns.tolist(),
        "matrix": corr.round(3).values.tolist(),
        "rain_corr": {k: round(float(v), 3) for k, v in rain_corr.items()},
    },
    "distributions": dist_payload,
    "rain_rate_table": {
        "humidity_labels": hum_labels,
        "cloud_labels": cld_labels,
        "rate": np.nan_to_num(rain_rate.values).round(1).tolist(),
        "count": np.nan_to_num(rain_count.values).astype(int).tolist(),
    },
    "models": {
        name: {
            "roc_auc": round(results[name]["roc_auc"], 3),
            "precision": round(results[name]["precision"], 3),
            "recall": round(results[name]["recall"], 3),
            "f1": round(results[name]["f1"], 3),
            "accuracy": round(results[name]["accuracy"], 3),
            "confusion_matrix": results[name]["confusion_matrix"],
        }
        for name in models
    },
    "winner": winner,
    "roc_curves": {
        name: {
            "fpr": [round(x, 4) for x in roc_series[name]["fpr"]],
            "tpr": [round(x, 4) for x in roc_series[name]["tpr"]],
            "auc": round(results[name]["roc_auc"], 3),
        }
        for name in models
    },
    "pr_curve": {
        "recall": [round(float(x), 4) for x in rec],
        "precision": [round(float(x), 4) for x in prec],
        "baseline": round(float(base_rate), 4),
        "op_recall": round(r50, 3),
        "op_precision": round(p50, 3),
    },
    "threshold_table": threshold_table,
    "rf_importance": {k: round(float(v), 4) for k, v in
                      rf_import.sort_values(ascending=False).items()},
    "logistic": {
        "coef": {k: round(float(v), 4) for k, v in coefs.sort_values(ascending=False).items()},
        "odds": {k: round(float(v), 4) for k, v in
                 odds.reindex(coefs.sort_values(ascending=False).index).items()},
    },
}

with open(ANALYSIS_DIR / "dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(dashboard, f, indent=2)
print(f"  saved -> {(ANALYSIS_DIR / 'dashboard_data.json').relative_to(ROOT)}")

# ---- FINDINGS.md (generated by the script, per instructions) ----
w = results[winner]
l = results["Logistic Regression"]
r = results["Random Forest"]
top3_imp = list(dashboard["rf_importance"].items())[:3]
top_push = coefs.sort_values(ascending=False)

def pm(feat):
    g = dashboard["group_means"][feat]
    return f"{g['no_rain']:.1f} -> {g['rain']:.1f} {g['unit']} ({g['delta']:+.1f})"

findings = f"""# Predicting Rain from Weather Conditions - Findings

*A classification study on 2,500 daily weather observations. Reproducible via
`python analysis/rain_analysis.py` (random_state = 42). {SOURCE_NOTE}.*

---

## The headline

**A Random Forest predicts rain with a ROC-AUC of {r['roc_auc']:.3f}, catching
{w['confusion_matrix']['tp']} of {w['confusion_matrix']['tp'] + w['confusion_matrix']['fn']}
real rain days on unseen data - despite rain being only {pct_rain:.1f}% of all days.**
Humidity and Cloud Cover are the two conditions that matter most.

---

## Why accuracy is a trap here

**The data is imbalanced: {n_rain:,} rainy days vs {n_norain:,} dry days ({pct_rain:.1f}% rain).**
A lazy model that *always* predicts "no rain" is right {naive_acc:.1f}% of the time - yet it
never warns you about a single storm. **So this study never judges a model on accuracy alone.**
Instead we use ROC-AUC, plus precision, recall and F1 for the rain (positive) class, and the
confusion matrix.

*So what:* headline accuracy would make a useless model look excellent. The metrics that
matter are whether we actually catch the rain.

---

## What does a rainy day look like?

**Rainy days are markedly more humid and cloudier, with slightly lower pressure.** Comparing
group means (no-rain -> rain):

| Feature | No-rain -> Rain (delta) |
|---|---|
| Humidity | {pm('Humidity')} |
| Cloud Cover | {pm('Cloud_Cover')} |
| Pressure | {pm('Pressure')} |
| Temperature | {pm('Temperature')} |
| Wind Speed | {pm('Wind_Speed')} |

*So what:* the physical intuition holds - humid, cloudy, low-pressure air brings rain. These
are the levers a model should lean on.

---

## A simple rule already separates most of the signal

**When humidity is low and skies are clear, the rain rate is
{dashboard['rain_rate_table']['rate'][0][0]:.0f}%; when both humidity and cloud cover are high,
it jumps to {dashboard['rain_rate_table']['rate'][2][2]:.0f}%.** (See
`charts/04_rainrate_humidity_cloud.png`.)

*So what:* even before machine learning, two features carve the days into low- and high-risk
buckets - which is exactly what the models then formalize.

---

## The models

Both models handle the imbalance with `class_weight="balanced"` and are evaluated on a
stratified 25% held-out test set ({len(y_test)} days, {int(y_test.sum())} of them rain).

| Model | ROC-AUC | Precision (rain) | Recall (rain) | F1 (rain) | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | {l['roc_auc']:.3f} | {l['precision']:.3f} | {l['recall']:.3f} | {l['f1']:.3f} | {l['accuracy']:.3f} |
| Random Forest | {r['roc_auc']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['accuracy']:.3f} |

**Winner by ROC-AUC: {winner} ({w['roc_auc']:.3f}).** The interpretable Logistic Regression is
a strong, close baseline ({l['roc_auc']:.3f}) - reassuring, because it means the signal is
largely learnable with a simple linear rule.

*So what:* both models rank a random rainy day above a random dry day almost every time - far
above the 50% of a coin flip.

### An honest caveat on that near-perfect Random Forest

The Random Forest scores ROC-AUC {r['roc_auc']:.3f} with only {w['confusion_matrix']['fp']} false
alarm and {w['confusion_matrix']['fn']} missed storms on the test set. **A score this clean is a
signal in itself: it almost certainly means this dataset was generated from a near-deterministic
rule** on humidity, cloud cover and temperature, rather than sampled from messy real weather.
Real operational data is noisier, and I would expect performance closer to the Logistic
Regression's on live observations. The result is reported honestly as-is, but I would not promise
100% recall in production without validating on real, independently collected data.

---

## Which conditions drive the prediction?

**Random Forest importance ranks {top3_imp[0][0]} ({top3_imp[0][1]:.3f}),
{top3_imp[1][0]} ({top3_imp[1][1]:.3f}) and {top3_imp[2][0]} ({top3_imp[2][1]:.3f}) on top.**
The Logistic Regression agrees on direction: the strongest push *toward* rain comes from
**{top_push.index[0]}** (coef {top_push.iloc[0]:+.2f}, odds ratio {np.exp(top_push.iloc[0]):.2f}
per +1 SD), while higher **{top_push.index[-1]}** pushes toward no-rain
(coef {top_push.iloc[-1]:+.2f}).

*So what:* the two very different model families agree on the same physical story, which is a
good sign the finding is real rather than an artifact.

---

## The precision / recall tradeoff

At the default 0.50 threshold the winner runs at precision {w['precision']:.3f} /
recall {w['recall']:.3f}. Moving the threshold trades one for the other:

| Threshold | Precision | Recall | F1 | Days flagged rain |
|---|---|---|---|---|
""" + "\n".join(
    f"| {row['threshold']:.1f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['predicted_rain']} |"
    for row in threshold_table
) + f"""

*So what:* if missing a storm is costly, lower the threshold to raise recall (at the price of
more false alarms); if false alarms are costly, raise it. The right operating point is a
business decision, not a modeling one.

---

## Future work

A gradient-boosted model (XGBoost or scikit-learn's `HistGradientBoostingClassifier`) would
likely squeeze out marginal gains over the Random Forest, and calibrated probabilities plus
cost-based threshold tuning would sharpen the operating point.

---

### Charts
- `charts/01_class_balance.png` - the imbalance and the accuracy trap
- `charts/02_feature_distributions.png` - conditions by outcome
- `charts/03_correlation_heatmap.png` - feature correlations
- `charts/04_rainrate_humidity_cloud.png` - rain rate by humidity x cloud
- `charts/05_roc_curves.png` - ROC curves, both models
- `charts/06_confusion_matrix.png` - confusion matrix, {winner}
- `charts/07_precision_recall.png` - precision-recall tradeoff
- `charts/08_rf_importance.png` - Random Forest feature importance
- `charts/09_logistic_coefficients.png` - logistic coefficients & odds ratios
"""

with open(ANALYSIS_DIR / "FINDINGS.md", "w", encoding="utf-8") as f:
    f.write(findings)
print(f"  saved -> {(ANALYSIS_DIR / 'FINDINGS.md').relative_to(ROOT)}")

print("\nDone. Winner:", winner,
      f"(ROC-AUC {results[winner]['roc_auc']:.3f})")
