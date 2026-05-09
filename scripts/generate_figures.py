"""
Generate ablation figures from results/fold_breakdown.csv.

Produces:
  results/figures/ablation_mean_val_auroc_per_category.png
  results/figures/ablation_per_fold_val_auroc_by_category.png

Run from the project root:
  python scripts/generate_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "fold_breakdown.csv"
FIG_DIR  = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

assert CSV_PATH.exists(), f"Missing: {CSV_PATH}"

# ── load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["fold"] = df["fold"].astype(int)

# prefer val_auroc; fall back to test_auroc
score_col = "val_auroc" if "val_auroc" in df.columns else "test_auroc"
f1_col    = "val_f1"    if "val_f1"    in df.columns else "test_f1"

print(f"Loaded {len(df)} rows from {CSV_PATH}")
print(f"Categories : {sorted(df['category'].unique())}")
print(f"Folds      : {sorted(df['fold'].unique())}")
print(f"Score col  : {score_col}")

# ── styling ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"savefig.dpi": 200, "figure.dpi": 140})

PALETTE = sns.color_palette("viridis", 6)

# ── Figure 1 — Mean AUROC ± std per category ────────────────────────────────
cat_agg = (
    df.groupby("category", as_index=False)
      .agg(
          mean_auroc=(score_col, "mean"),
          std_auroc =(score_col, "std"),
          mean_f1   =(f1_col,    "mean"),
      )
      .sort_values("mean_auroc", ascending=False)
      .reset_index(drop=True)
)
cat_agg["std_auroc"] = cat_agg["std_auroc"].fillna(0.0)
overall_mean = cat_agg["mean_auroc"].mean()

fig1, ax1 = plt.subplots(figsize=(11, 6))
ax1.bar(
    cat_agg["category"],
    cat_agg["mean_auroc"],
    yerr=cat_agg["std_auroc"],
    capsize=6,
    color=PALETTE,
    edgecolor="white",
    linewidth=0.8,
    error_kw={"linewidth": 2, "ecolor": "#333333"},
    zorder=3,
)
ax1.axhline(
    overall_mean,
    color="#e63946",
    linestyle="--",
    linewidth=2,
    label=f"Overall mean AUROC = {overall_mean:.4f}",
    zorder=4,
)
# Annotate each bar with its mean value
for i, row in cat_agg.iterrows():
    ax1.text(
        i, row["mean_auroc"] + row["std_auroc"] + 0.003,
        f"{row['mean_auroc']:.4f}",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

ax1.set_title(
    "Mean Validation AUROC per Defect Category\n(100% Labels, 3-Fold Cross-Validation)",
    fontsize=14, fontweight="bold", pad=12,
)
ax1.set_xlabel("Defect Category", fontsize=13)
ax1.set_ylabel("Validation AUROC (0 to 1)", fontsize=13)
ax1.set_ylim(0.80, 1.06)
ax1.legend(frameon=True, fontsize=11)
ax1.grid(axis="y", alpha=0.4, zorder=0)
fig1.tight_layout()

out1 = FIG_DIR / "ablation_mean_val_auroc_per_category.png"
fig1.savefig(out1, dpi=200)
print(f"\n[OK] Saved → {out1}")

# ── Figure 2 — Per-fold AUROC per category ──────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 6))
category_palette = sns.color_palette("tab10", df["category"].nunique())

for (category, grp), color in zip(df.groupby("category"), category_palette):
    grp = grp.sort_values("fold")
    ax2.plot(
        grp["fold"],
        grp[score_col],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=category,
        color=color,
    )
    # Annotate each point
    for _, r in grp.iterrows():
        ax2.annotate(
            f"{r[score_col]:.3f}",
            xy=(r["fold"], r[score_col]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
        )

ax2.set_title(
    "Validation AUROC per Fold by Defect Category\n(100% Labels)",
    fontsize=14, fontweight="bold", pad=12,
)
ax2.set_xlabel("Fold", fontsize=13)
ax2.set_ylabel("Validation AUROC (0 to 1)", fontsize=13)
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(["Fold 1", "Fold 2", "Fold 3"], fontsize=12)
ax2.set_ylim(0.80, 1.05)
ax2.legend(title="Defect Category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=11)
ax2.grid(axis="y", alpha=0.4)
fig2.tight_layout()

out2 = FIG_DIR / "ablation_per_fold_val_auroc_by_category.png"
fig2.savefig(out2, dpi=200)
print(f"[OK] Saved → {out2}")

print("\nDone. Both figures saved to results/figures/")
