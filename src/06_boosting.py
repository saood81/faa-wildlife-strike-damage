"""
Add HistGradientBoostingClassifier as model seven: boosted trees perform strongly on
medium-sized tables (Grinsztajn et al., 2022).
Follow steps 3-4: same preprocessing, stratified 5-fold PR-AUC CV, 2024 max-F1 threshold
and bootstrap seed.
Early stopping uses only an internal 10% split of each training fold.
Input : data/processed/analytic.csv.gz, outputs/predictions/predictions_A.csv.gz
Output: outputs/predictions/predictions_HGB.csv.gz;
        outputs/tables/06_boosting_tuning.csv, 06_boosting_test_metrics.csv
Run from the repository root: python src/06_boosting.py (~15-25 min)
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config as C
import pipeline as P

warnings.filterwarnings("ignore", category=UserWarning)
N_BOOT = 1000

df = P.load_data()
X_tr, y_tr = P.get_xy(df, "train")
X_va, y_va = P.get_xy(df, "valid")
X_te, y_te = P.get_xy(df, "test")

hgb = HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.1,
                                     n_iter_no_change=20, random_state=C.RANDOM_STATE)
# The initial grid selected values on three boundaries, so widen the search towards
# smaller, more regularised trees (decision M7).
grid = {"model__learning_rate": [0.02, 0.05, 0.1],
        "model__max_leaf_nodes": [7, 15, 31],
        "model__min_samples_leaf": [100, 300],
        "model__class_weight": [None, "balanced"]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)

t0 = time.time()
search = GridSearchCV(P.make_pipeline(hgb), grid, scoring="average_precision", cv=cv,
                      n_jobs=1, refit=True)
search.fit(X_tr, y_tr)
model = search.best_estimator_
p_va, p_te = P.scores(model, X_va), P.scores(model, X_te)

prec, rec, thr = precision_recall_curve(y_va, p_va)
f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
threshold = float(thr[int(np.nanargmax(f1[:-1]))])

i = search.best_index_
best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
pd.DataFrame([{
    "model": "HGB", "best_params": best,
    "cv_pr_auc_mean": round(search.cv_results_["mean_test_score"][i], 4),
    "cv_pr_auc_sd": round(search.cv_results_["std_test_score"][i], 4),
    "valid_pr_auc": round(average_precision_score(y_va, p_va), 4),
    "threshold_from_valid": round(threshold, 4),
    "boosting_iterations": int(model.named_steps["model"].n_iter_),
    "train_rows": len(y_tr), "grid_size": len(search.cv_results_["params"]),
    "minutes": round((time.time() - t0) / 60, 1)}]).to_csv(
    C.TABLE_DIR / "06_boosting_tuning.csv", index=False)
print(f"HGB CV PR-AUC {search.cv_results_['mean_test_score'][i]:.3f} | "
      f"valid {average_precision_score(y_va, p_va):.3f} | best {best}")
edge = [k for k, v in search.best_params_.items() if v in (grid[k][0], grid[k][-1])
        and k != "model__class_weight" and len(grid[k]) > 1]
print("Selected values on a grid edge:", edge or "none")
res = pd.DataFrame(search.cv_results_)
for k in grid:
    print(res.groupby(f"param_{k}", dropna=False)["mean_test_score"].max().round(4).to_dict())

pd.DataFrame({"split": ["valid"] * len(y_va) + ["test"] * len(y_te),
              "y": np.r_[y_va.values, y_te.values], "HGB": np.r_[p_va, p_te]}).to_csv(
    C.ROOT / "outputs" / "predictions" / "predictions_HGB.csv.gz", index=False, compression="gzip")

# test metrics, same bootstrap as step 4
pred = pd.read_csv(C.ROOT / "outputs" / "predictions" / "predictions_A.csv.gz")
test = pred[pred["split"] == "test"].reset_index(drop=True)
y = test["y"].to_numpy()
assert np.array_equal(y, y_te.to_numpy()), "row order differs from step 3"
p_rf = test["RF"].to_numpy()


def metric_set(y_true, prob, t):
    hard = (prob >= t).astype(int)
    return {"pr_auc": average_precision_score(y_true, prob), "roc_auc": roc_auc_score(y_true, prob),
            "brier": brier_score_loss(y_true, prob), "recall": recall_score(y_true, hard),
            "precision": precision_score(y_true, hard, zero_division=0),
            "f1": f1_score(y_true, hard), "balanced_acc": balanced_accuracy_score(y_true, hard)}


rng = np.random.default_rng(C.RANDOM_STATE)
boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
point = metric_set(y, p_te, threshold)
boot = pd.DataFrame([metric_set(y[b], p_te[b], threshold) for b in boot_idx])
diff = np.array([average_precision_score(y[b], p_te[b]) - average_precision_score(y[b], p_rf[b])
                 for b in boot_idx])

# Select the Youden threshold using the 2024 validation data.
fpr, tpr, thr_roc = roc_curve(y_va, p_va)
t_y = thr_roc[np.argmax(tpr - fpr)]
ba_y = balanced_accuracy_score(y, p_te >= t_y)
ba_boot = [balanced_accuracy_score(y[b], p_te[b] >= t_y) for b in boot_idx]

row = {"model": "HGB"}
for k, v in point.items():
    lo, hi = np.percentile(boot[k], [2.5, 97.5])
    row[k], row[f"{k}_ci"] = round(v, 3), f"[{lo:.3f}, {hi:.3f}]"
lo, hi = np.percentile(diff, [2.5, 97.5])
row.update({"pr_auc_diff_vs_RF": round(point["pr_auc"] - average_precision_score(y, p_rf), 3),
            "pr_auc_diff_vs_RF_ci": f"[{lo:.3f}, {hi:.3f}]",
            "threshold": round(threshold, 3), "mean_predicted_prob": round(p_te.mean(), 4),
            "observed_damage_rate": round(y.mean(), 4),
            "balanced_acc_youden": round(ba_y, 3),
            "balanced_acc_youden_ci": "[{:.3f}, {:.3f}]".format(*np.percentile(ba_boot, [2.5, 97.5]))})
pd.DataFrame([row]).to_csv(C.TABLE_DIR / "06_boosting_test_metrics.csv", index=False)
print(pd.Series(row).to_string())
