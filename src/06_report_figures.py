"""
Step 6 - Report figure and the balanced-accuracy comparison with prior work.

Uses only the saved predictions from step 3, so it runs in seconds and needs no
refitting.

1) Figure 2 of the report (fig2_report_pr_calibration.png):
   (a) precision-recall curves for the six models on the test years 2025-26;
   (b) reliability curves for three contrasting models (RF, LR, GNB).

2) Balanced accuracy (06_balanced_accuracy.csv).
   The best prior NWSD damage model (Misra, Toppo and Mendonca, 2022) is reported
   as 78.81% *balanced* accuracy (as cited by Kim, Starkute and Mendonca, 2026).
   Balanced accuracy = (recall + specificity) / 2, so unlike plain accuracy it
   does not depend on the damage rate, which makes a like-for-like comparison
   possible. It is still threshold-dependent, so we pick the threshold that
   maximises balanced accuracy (Youden's J = recall + specificity - 1) on the
   2024 validation year and apply it unchanged to 2025-26. 95% CIs come from
   1,000 bootstrap resamples of the test rows (same seed as step 4).
   For reference only, the "oracle" row shows the best balanced accuracy
   achievable if the threshold were tuned on the test set itself (optimistic,
   not a valid estimate).

Input : outputs/predictions/predictions_A.csv.gz
Output: outputs/figures/fig2_report_pr_calibration.png
        outputs/tables/06_balanced_accuracy.csv

Run from the repository root:  python src/06_report_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             precision_recall_curve, roc_curve)

import config as C

N_BOOT = 1000
MODELS = ["LR", "GNB", "KNN", "SVM", "DT", "RF"]
NAMES = {"LR": "Logistic regression", "GNB": "Gaussian naive Bayes", "KNN": "k-nearest neighbours",
         "SVM": "SVM (RBF)", "DT": "Decision tree", "RF": "Random forest"}
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK2 = "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

pred = pd.read_csv(C.ROOT / "outputs" / "predictions" / "predictions_A.csv.gz")
valid = pred[pred["split"] == "valid"].reset_index(drop=True)
test = pred[pred["split"] == "test"].reset_index(drop=True)
y_va, y = valid["y"].to_numpy(), test["y"].to_numpy()
pr_auc = {m: average_precision_score(y, test[m]) for m in MODELS}
best = max(MODELS, key=pr_auc.get)

# ------------------------------------------------------------ Figure 2 (report)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
for m in MODELS:
    prec, rec, _ = precision_recall_curve(y, test[m])
    ax1.plot(rec, prec, color=COLORS[m], lw=1.9 if m == best else 1.1,
             label=f"{m} ({pr_auc[m]:.3f})")
ax1.axhline(y.mean(), color=INK2, ls="--", lw=0.9)
ax1.text(0.02, y.mean() + 0.02, f"no skill = {y.mean():.3f}", ha="left", fontsize=6.5, color=INK2)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
ax1.set_title("(a) Precision-recall, test 2025-26")
ax1.legend(title="Model (PR-AUC)", title_fontsize=6.5, frameon=False, fontsize=6.5, loc="upper right")

for m in ["RF", "LR", "GNB"]:
    frac, mean_p = calibration_curve(y, test[m], n_bins=10, strategy="quantile")
    ax2.plot(mean_p, frac, marker="o", ms=3, lw=1.3, color=COLORS[m], label=m)
ax2.plot([0, 1], [0, 1], color=INK2, ls="--", lw=0.9)
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.set_xlabel("Mean predicted probability"); ax2.set_ylabel("Observed damage rate")
ax2.set_title("(b) Reliability (10 quantile bins)")
ax2.legend(frameon=False, fontsize=6.5, loc="upper left")
ax2.text(0.98, 0.04, "below the line = over-predicts", ha="right", fontsize=6.5, color=INK2)
fig.tight_layout()
fig.savefig(C.FIG_DIR / "fig2_report_pr_calibration.png")
plt.close(fig)


# ------------------------------------------------------------ balanced accuracy
def youden_threshold(y_true, prob):
    fpr, tpr, thr = roc_curve(y_true, prob)
    return thr[np.argmax(tpr - fpr)]


def rates(y_true, prob, thr):
    hard = prob >= thr
    tp = np.sum(hard & (y_true == 1)); fn = np.sum(~hard & (y_true == 1))
    tn = np.sum(~hard & (y_true == 0)); fp = np.sum(hard & (y_true == 0))
    rec, spec = tp / (tp + fn), tn / (tn + fp)
    return {"balanced_acc": (rec + spec) / 2, "recall": rec, "specificity": spec,
            "precision": tp / max(tp + fp, 1), "accuracy": (tp + tn) / len(y_true),
            "flagged_per_1000": 1000 * (tp + fp) / len(y_true)}


rng = np.random.default_rng(C.RANDOM_STATE)
boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
rows = []
for m in MODELS:
    p = test[m].to_numpy()
    thr = youden_threshold(y_va, valid[m].to_numpy())
    r = {"model": m, "threshold_rule": "Youden J on 2024", "threshold": round(float(thr), 4)}
    r.update({k: round(v, 4) for k, v in rates(y, p, thr).items()})
    ba = [balanced_accuracy_score(y[i], p[i] >= thr) for i in boot_idx]
    lo, hi = np.percentile(ba, [2.5, 97.5])
    r["balanced_acc_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    rows.append(r)
    thr_o = youden_threshold(y, p)            # optimistic reference only
    o = {"model": m, "threshold_rule": "oracle: Youden J on test (reference only)",
         "threshold": round(float(thr_o), 4)}
    o.update({k: round(v, 4) for k, v in rates(y, p, thr_o).items()})
    o["balanced_acc_ci"] = "-"
    rows.append(o)
rows.append({"model": "All 'no damage'", "threshold_rule": "trivial rule",
             "balanced_acc": 0.5, "recall": 0.0, "specificity": 1.0,
             "accuracy": round(1 - y.mean(), 4), "flagged_per_1000": 0.0})
out = pd.DataFrame(rows)
out.to_csv(C.TABLE_DIR / "06_balanced_accuracy.csv", index=False)
print(out[["model", "threshold_rule", "threshold", "balanced_acc", "balanced_acc_ci", "recall",
           "specificity", "precision", "accuracy"]].to_string(index=False))
print("Saved fig2_report_pr_calibration.png and 06_balanced_accuracy.csv")
