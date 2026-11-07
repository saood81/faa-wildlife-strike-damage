"""
Each experiment changes one factor and refits six models using step-3 hyper-parameters,
without retuning.
E1: missingness indicators versus plain imputation, plus a flags-only model.
E2: verified labels versus adding 12,079 unverified no-damage rows.
E3: temporal versus random splitting across all years.
E4: pre-strike set A versus set B, adding strike location and ingestion.
E1, E2 and E4 use paired 95% CIs; E3 uses unpaired CIs because its test rows differ.
Input : data/processed/analytic.csv.gz, data/processed/models/*_A.joblib
Output: outputs/tables/05_experiments.csv, outputs/tables/05_calibration_by_year.csv,
        outputs/figures/fig7_experiments.png
Run from the repository root (after 03_models.py): python src/05_experiments.py
"""
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import pipeline as P

warnings.filterwarnings("ignore", category=UserWarning)
MODELS = ["LR", "GNB", "KNN", "SVM", "DT", "RF"]
N_BOOT = 1000


# helpers
def tuned_estimator(name):
    fitted = joblib.load(C.PROCESSED_DIR / "models" / f"{name}_A.joblib")["model"]
    # A calibration wrapper supplies SVM probabilities.
    if isinstance(fitted, CalibratedClassifierCV):
        fitted = fitted.estimator
    return clone(fitted.named_steps["model"])


def fit_predict(name, X_tr, y_tr, X_te, feature_set="A", missing="indicator", pipe=None):
    if pipe is None:
        pipe = P.make_pipeline(tuned_estimator(name), feature_set, missing)
    if name == "SVM":
        X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=P.SVM_SUBSAMPLE,
                                            stratify=y_tr, random_state=C.RANDOM_STATE)
        pipe = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def summarise(y, prob, boot_idx):
    pr = [average_precision_score(y[i], prob[i]) for i in boot_idx]
    return {"pr_auc": average_precision_score(y, prob),
            "pr_auc_lo": np.percentile(pr, 2.5), "pr_auc_hi": np.percentile(pr, 97.5),
            "roc_auc": roc_auc_score(y, prob), "brier": brier_score_loss(y, prob),
            "mean_pred": prob.mean(), "damage_rate": y.mean(), "_boot_pr": np.array(pr)}


df = P.load_data()
X_tr, y_tr = P.get_xy(df, "train")
X_te, y_te = P.get_xy(df, "test")
y = y_te.to_numpy()
rng = np.random.default_rng(C.RANDOM_STATE)
boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]

results = []
main = {}


def record(exp, arm, name, res, paired=True):
    row = {"experiment": exp, "arm": arm, "model": name,
           **{k: v for k, v in res.items() if not k.startswith("_")}}
    if paired and arm != "main":
        d = res["_boot_pr"] - main[name]["_boot_pr"]
        row.update(pr_auc_change=res["pr_auc"] - main[name]["pr_auc"],
                   change_lo=np.percentile(d, 2.5), change_hi=np.percentile(d, 97.5))
    results.append(row)
    print(f"{exp:3s} {arm:28s} {name:4s} PR-AUC {res['pr_auc']:.3f} "
          f"ROC {res['roc_auc']:.3f} mean p {res['mean_pred']:.3f}")


t0 = time.time()
# main run (reference)
main_probs = {}
for m in MODELS:
    main_probs[m] = fit_predict(m, X_tr, y_tr, X_te)
    main[m] = summarise(y, main_probs[m], boot_idx)
    record("ALL", "main", m, main[m])

# E1 missingness
for m in MODELS:
    record("E1", "imputation only (no flags)", m,
           summarise(y, fit_predict(m, X_tr, y_tr, X_te, missing="impute"), boot_idx))

# The flags-only model uses blank-field indicators to measure report completeness.
flag_cols = [c for c in X_tr.columns if X_tr[c].isna().any()]
F_tr = X_tr[flag_cols].isna().astype(int)
F_te = X_te[flag_cols].isna().astype(int)
for m in MODELS:
    pipe = Pipeline([("scale", StandardScaler()), ("model", tuned_estimator(m))])
    record("E1", "blank-field flags ONLY", m,
           summarise(y, fit_predict(m, F_tr, y_tr, F_te, pipe=pipe), boot_idx))

# E2 label noise
X_tr_all, y_tr_all = P.get_xy(df, "train", verified_only=False)
for m in MODELS:
    record("E2", "+ unverified negatives", m,
           summarise(y, fit_predict(m, X_tr_all, y_tr_all, X_te), boot_idx))

# E3 split design
# Randomly split verified 2016-26 rows, matching temporal split sizes and damage rate.
ver = df[df["LABEL_VERIFIED"] == 1]
num, ordn, cat, high = C.feature_lists("A")
X_all, y_all = ver[num + ordn + cat + high], ver[C.TARGET].astype(int)
Xr_tr, Xr_rest, yr_tr, yr_rest = train_test_split(
    X_all, y_all, train_size=len(y_tr), stratify=y_all, random_state=C.RANDOM_STATE)
Xr_te, _, yr_te, _ = train_test_split(
    Xr_rest, yr_rest, train_size=len(y_te), stratify=yr_rest, random_state=C.RANDOM_STATE)
yr = yr_te.to_numpy()
boot_r = [rng.integers(0, len(yr), len(yr)) for _ in range(N_BOOT)]
for m in MODELS:
    record("E3", "random split (all years mixed)", m,
           summarise(yr, fit_predict(m, Xr_tr, yr_tr, Xr_te), boot_r), paired=False)

years = df.loc[(df["SPLIT"] == "test") & (df["LABEL_VERIFIED"] == 1), "INCIDENT_YEAR"].to_numpy()
cal = [{"model": m, "year": yr_, "n": int((years == yr_).sum()),
        "actual_damage_rate": round(y[years == yr_].mean(), 4),
        "mean_predicted": round(main_probs[m][years == yr_].mean(), 4)}
       for m in MODELS for yr_ in sorted(set(years))]
pd.DataFrame(cal).to_csv(C.TABLE_DIR / "05_calibration_by_year.csv", index=False)

# E4 feature set B
X_trB, _ = P.get_xy(df, "train", feature_set="B")
X_teB, _ = P.get_xy(df, "test", feature_set="B")
for m in MODELS:
    record("E4", "set B (+ strike location)", m,
           summarise(y, fit_predict(m, X_trB, y_tr, X_teB, feature_set="B"), boot_idx))

# save
out = pd.DataFrame(results).round(4)
out.to_csv(C.TABLE_DIR / "05_experiments.csv", index=False)
print(f"\nTotal {round((time.time() - t0) / 60, 1)} min")

# Figure 7
BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
panels = [("E1", "imputation only (no flags)", "E1  Missingness flags vs plain imputation"),
          ("E2", "+ unverified negatives", "E2  Adding unverified 'no damage' labels"),
          ("E3", "random split (all years mixed)", "E3  Random split vs temporal split"),
          ("E4", "set B (+ strike location)", "E4  Adding strike location (set B)")]
ref = out[out["arm"] == "main"].set_index("model")
fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.0), sharex=True)
for ax, (exp, arm, title) in zip(axes.ravel(), panels):
    alt = out[(out["experiment"] == exp) & (out["arm"] == arm)].set_index("model")
    for i, m in enumerate(MODELS):
        ax.plot([ref.loc[m, "pr_auc"], alt.loc[m, "pr_auc"]], [i, i], color="#d6d5d0", lw=2, zorder=1)
    ax.scatter(ref.loc[MODELS, "pr_auc"], range(6), color=BLUE, s=30, zorder=2,
               edgecolor="white", linewidth=1, label="Main design")
    ax.scatter(alt.loc[MODELS, "pr_auc"], range(6), color=ORANGE, s=30, zorder=2,
               edgecolor="white", linewidth=1, label="Changed design")
    ax.set_yticks(range(6)); ax.set_yticklabels(MODELS)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("Test PR-AUC")
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.text(0.01, -0.01, "E3: the random-split test set has a 6.8% damage rate vs 5.5% for 2025-26, "
         "which by itself raises PR-AUC; ROC-AUC (table 05) is not affected by the damage rate.",
         fontsize=6.5, color=INK2)
fig.tight_layout()
fig.savefig(C.FIG_DIR / "fig7_experiments.png")
print("Saved tables and figure 7.")
