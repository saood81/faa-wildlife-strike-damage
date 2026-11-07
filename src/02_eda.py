"""
Create report figures using training years 2016-23; Figure 1 alone uses all years to
show drift.
Input : data/processed/analytic.csv.gz (from 01_prepare.py)
Output: outputs/figures/fig1_volume_prevalence_by_year.png,
        outputs/figures/fig2_damage_rate_by_feature.png,
        outputs/figures/fig3_missingness_vs_damage.png
Run from the repository root: python src/02_eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

import config as C

C.FIG_DIR.mkdir(parents=True, exist_ok=True)

# one consistent, print-friendly style for every figure
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

df = pd.read_csv(C.ANALYTIC_FILE, low_memory=False)
ver = df[df["LABEL_VERIFIED"] == 1]
train = ver[ver["SPLIT"] == "train"]
y = C.TARGET

# Figure 1
by_year = ver.groupby("INCIDENT_YEAR")[y].agg(n="size", rate="mean")
split_of = ver.groupby("INCIDENT_YEAR")["SPLIT"].first()
colors = split_of.map({"train": BLUE, "valid": ORANGE, "test": INK2})

fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.25})
a1.bar(by_year.index, by_year["n"], color=colors, width=0.7, edgecolor="white", linewidth=1)
a1.set_ylabel("Verified strikes")
a1.set_title("a) Strike reports per year (2026 = Jan-Jun only)")
a1.grid(axis="x", visible=False)
handles = [Patch(color=c, label=l) for l, c in
           [("Train 2016-23", BLUE), ("Validate 2024", ORANGE), ("Test 2025-26", INK2)]]
a1.legend(handles=handles, frameon=False, ncol=3, loc="upper left", fontsize=8)

a2.plot(by_year.index, 100 * by_year["rate"], color=INK2, lw=2, zorder=1)
a2.scatter(by_year.index, 100 * by_year["rate"], c=colors, s=36, zorder=2,
           edgecolor="white", linewidth=1.2)
a2.set_ylabel("Damaged (%)")
a2.set_ylim(0, 10)
a2.set_title("b) Share of strikes that damaged the aircraft")
a2.set_xticks(by_year.index)
a2.grid(axis="x", visible=False)
fig.savefig(C.FIG_DIR / "fig1_volume_prevalence_by_year.png")
plt.close(fig)

# Figure 2
panels = [
    ("SIZE_ORD", "Bird size (pilot estimate)", {1: "Small", 2: "Medium", 3: "Large"}),
    ("AC_MASS", "Aircraft mass class", {1: "1 (<=2.25 t)", 2: "2", 3: "3", 4: "4", 5: "5 (>272 t)"}),
    ("TYPE_ENG", "Engine type", {"A": "Piston", "B": "Turbojet", "C": "Turboprop",
                                 "D": "Turbofan", "F": "Turboshaft"}),
    ("PHASE_OF_FLIGHT", "Phase of flight", None),
]
fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.0))
overall = 100 * train[y].mean()
for ax, (col, title, labels) in zip(axes.ravel(), panels):
    g = train.groupby(train[col].astype("string").fillna("Not reported"))[y].agg(n="size", rate="mean")
    g = g[g["n"] >= 100]  # Omit categories with too few records for reliable estimates.
    if labels:
        g.index = [labels.get(int(float(i)) if col != "TYPE_ENG" and i != "Not reported" else i, i)
                   for i in g.index]
    g = g.sort_values("rate")
    bar_cols = [GREY if i == "Not reported" else BLUE for i in g.index]
    ax.barh(g.index, 100 * g["rate"], color=bar_cols, height=0.65)
    ax.axvline(overall, color=INK2, lw=1, ls="--", zorder=0)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelsize=8)
    for i, (r, n) in enumerate(zip(g["rate"], g["n"])):
        ax.text(100 * r + 0.4, i, f"{100*r:.1f}%", va="center", fontsize=7, color=INK2,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.set_xlim(0, max(100 * g["rate"].max() * 1.25, overall * 1.5))
for ax in axes[1]:
    ax.set_xlabel("Damaged (%)")
fig.text(0.01, -0.01, f"Training years 2016-23, verified labels. Dashed line = overall rate "
         f"({overall:.1f}%). Grey = field left blank. Categories with <100 strikes hidden.",
         fontsize=7, color=INK2)
fig.tight_layout()
fig.savefig(C.FIG_DIR / "fig2_damage_rate_by_feature.png")
plt.close(fig)

# Figure 3
rows = []
pretty = {"SPEED": "Speed", "NUM_SEEN_ORD": "Number of birds seen", "WARNED": "Pilot warned",
          "HEIGHT": "Height", "SKY": "Sky condition", "TIME_OF_DAY": "Light conditions",
          "SIZE_ORD": "Bird size", "PHASE_OF_FLIGHT": "Phase of flight",
          "DISTANCE": "Distance from airport", "LATITUDE": "Airport location",
          "HOUR_SIN": "Time of strike"}
for col, name in pretty.items():
    m = train[col].isna()
    rows.append({"feature": f"{name} ({100*m.mean():.0f}% blank)",
                 "missing": 100 * train.loc[m, y].mean(),
                 "present": 100 * train.loc[~m, y].mean()})
r = pd.DataFrame(rows).sort_values("present")
fig, ax = plt.subplots(figsize=(6.3, 3.6))
for i, row in enumerate(r.itertuples()):
    ax.plot([row.missing, row.present], [i, i], color="#d6d5d0", lw=2, zorder=1)
ax.scatter(r["missing"], range(len(r)), color=GREY, s=40, zorder=2, label="Field left blank",
           edgecolor="white", linewidth=1.2)
ax.scatter(r["present"], range(len(r)), color=BLUE, s=40, zorder=2, label="Field filled in",
           edgecolor="white", linewidth=1.2)
ax.set_yticks(range(len(r)))
ax.set_yticklabels(r["feature"], fontsize=8)
ax.tick_params(axis="y", length=0)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Damaged (%)")
ax.set_xlim(0, 12)
ax.set_title("Damage is usually more common when a field is filled in")
ax.legend(frameon=False, loc="lower left", fontsize=8)
fig.savefig(C.FIG_DIR / "fig3_missingness_vs_damage.png")
plt.close(fig)

print("Saved 3 figures to", C.FIG_DIR.relative_to(C.ROOT))
