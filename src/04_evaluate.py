"""
Step 4 - Evaluate the six models on the test years (2025-26).

Metrics (all computed at the real damage rate - the test set is NOT rebalanced):
  PR-AUC (primary)   area under the precision-recall curve; a useless model
                     scores about the damage rate (~0.055), a perfect one 1.0
  ROC-AUC            probability a damaged strike is ranked above an undamaged one
  Brier score        mean squared error of the probabilities (lower = better
                     calibrated and sharper)
  Recall, precision, F1, balanced accuracy - at the threshold chosen on 2024
  Calibration-in-the-large: mean predicted probability vs actual damage rate

Uncertainty: 1,000 bootstrap resamples of the test rows give 95% percentile
confidence intervals. The same resamples are used for every model, so the
difference in PR-AUC between each model and the best one also gets a CI
(paired bootstrap): if that CI excludes 0, the gap is unlikely to be chance.

Input : outputs/predictions/predictions_A.csv.gz, data/processed/models/*_A.joblib
Output: outputs/tables/04_test_metrics.csv, 04_confusion_matrices.csv,
        04_permutation_importance.csv
        outputs/figures/fig4_pr_curves.png, fig5_calibration.png,
        fig6_permutation_importance.png

Run from the repository root:  python src/04_evaluate.py
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

import config as C
import pipeline as P

N_BOOT = 1000
MODELS = ["LR", "GNB", "KNN", "SVM", "DT", "RF"]
NAMES = {"LR": "Logistic regression", "GNB": "Gaussian naive Bayes", "KNN": "k-nearest neighbours",
         "SVM": "SVM (RBF)", "DT": "Decision tree", "RF": "Random forest"}
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK2 = "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

pred = pd.read_csv(C.ROOT / "outputs" / "predictions" / "predictions_A.csv.gz")
test = pred[pred["split"] == "test"].reset_index(drop=True)
y = test["y"].to_numpy()
thresholds = {m: joblib.load(C.PROCESSED_DIR / "models" / f"{m}_A.joblib")["threshold"]
              for m in MODELS}


def metric_set(y_true, prob, thr):
    hard = (prob >= thr).astype(int)
    return {"pr_auc": average_precision_score(y_true, prob),
            "roc_auc": roc_auc_score(y_true, prob),
            "brier": brier_score_loss(y_true, prob),
            "recall": recall_score(y_true, hard, zero_division=0),
            "precision": precision_score(y_true, hard, zero_division=0),
            "f1": f1_score(y_true, hard, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_true, hard)}


# ------------------------------------------------------------ point estimates + bootstrap
rng = np.random.default_rng(C.RANDOM_STATE)
boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]

point = {m: metric_set(y, test[m].to_numpy(), thresholds[m]) for m in MODELS}
boot = {m: pd.DataFrame([metric_set(y[i], test[m].to_numpy()[i], thresholds[m]) for i in boot_idx])
        for m in MODELS}
best = max(MODELS, key=lambda m: point[m]["pr_auc"])

rows = []
for m in MODELS:
    r = {"model": m}
    for k, v in point[m].items():
        lo, hi = np.percentile(boot[m][k], [2.5, 97.5])
        r[k] = round(v, 3)
        r[f"{k}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    diff = boot[m]["pr_auc"] - boot[best]["pr_auc"]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    r["pr_auc_diff_vs_best_ci"] = "-" if m == best else f"[{lo:.3f}, {hi:.3f}]"
    r["threshold"] = round(thresholds[m], 3)
    r["mean_predicted_prob"] = round(test[m].mean(), 4)
    r["observed_damage_rate"] = round(y.mean(), 4)
    rows.append(r)
metrics = pd.DataFrame(rows)
metrics.to_csv(C.TABLE_DIR / "04_test_metrics.csv", index=False)
print(metrics[["model", "pr_auc", "pr_auc_ci", "roc_auc", "brier", "recall", "precision",
               "f1", "balanced_acc", "mean_predicted_prob"]].to_string(index=False))
print(f"No-skill PR-AUC (= test damage rate): {y.mean():.3f}.  Best model: {best}")

# ------------------------------------------------------------ confusion matrices
cm_rows = []
for m in MODELS:
    tn, fp, fn, tp = confusion_matrix(y, (test[m] >= thresholds[m]).astype(int)).ravel()
    cm_rows.append({"model": m, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "flagged_per_1000_strikes": round(1000 * (tp + fp) / len(y), 1)})
pd.DataFrame(cm_rows).to_csv(C.TABLE_DIR / "04_confusion_matrices.csv", index=False)

# ------------------------------------------------------------ Figure 4: PR curves
fig, ax = plt.subplots(figsize=(6.3, 4.2))
for m in MODELS:
    prec, rec, _ = precision_recall_curve(y, test[m])
    ax.plot(rec, prec, color=COLORS[m], lw=2 if m == best else 1.4,
            label=f"{NAMES[m]} (PR-AUC {point[m]['pr_auc']:.2f})")
ax.axhline(y.mean(), color=INK2, ls="--", lw=1)
ax.text(0.02, y.mean() + 0.015, f"No skill = damage rate ({y.mean():.3f})",
        ha="left", fontsize=7, color=INK2)
ax.set_xlabel("Recall (share of damaging strikes caught)")
ax.set_ylabel("Precision (share of alerts that are real)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Precision-recall curves, test years 2025-26")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.savefig(C.FIG_DIR / "fig4_pr_curves.png")
plt.close(fig)

# ------------------------------------------------------------ Figure 5: calibration
fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.4), sharex=True, sharey=True)
for ax, m in zip(axes.ravel(), MODELS):
    frac, mean_p = calibration_curve(y, test[m], n_bins=10, strategy="quantile")
    ax.plot([0, 0.6], [0, 0.6], color=INK2, ls="--", lw=1)
    ax.plot(mean_p, frac, marker="o", ms=4, color=COLORS[m], lw=1.6, clip_on=True)
    ax.set_title(f"{m}  (Brier {point[m]['brier']:.3f})", fontsize=9)
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6)
for ax in axes[1]:
    ax.set_xlabel("Predicted probability")
for ax in axes[:, 0]:
    ax.set_ylabel("Observed damage rate")
fig.suptitle("Calibration on the test years: points on the dashed line = honest probabilities",
             x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(C.FIG_DIR / "fig5_calibration.png")
plt.close(fig)

# ------------------------------------------------------------ Figure 6: permutation importance
# Shuffle one input column at a time in the test data and measure how much
# PR-AUC drops. A big drop = the model relies on that column.
df = P.load_data()
X_te, y_te = P.get_xy(df, "test")
model = joblib.load(C.PROCESSED_DIR / "models" / f"{best}_A.joblib")["model"]
pi = permutation_importance(model, X_te, y_te, scoring="average_precision", n_repeats=5,
                            random_state=C.RANDOM_STATE, n_jobs=1)
imp = (pd.DataFrame({"feature": X_te.columns, "mean_drop": pi.importances_mean,
                     "sd": pi.importances_std})
         .sort_values("mean_drop", ascending=False))
imp.round(4).to_csv(C.TABLE_DIR / "04_permutation_importance.csv", index=False)

top = imp.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.3, 4.2))
ax.barh(top["feature"], top["mean_drop"], xerr=top["sd"], color=COLORS[best], height=0.65,
        error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0, labelsize=8)
ax.set_xlabel("Drop in test PR-AUC when the column is shuffled")
ax.set_title(f"What the {NAMES[best].lower()} relies on (top 15 inputs)")
fig.savefig(C.FIG_DIR / "fig6_permutation_importance.png")
plt.close(fig)
print("Saved tables and figures 4-6.")
