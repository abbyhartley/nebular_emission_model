"""Re-plot the in-survey Balmer-conditioning scatter figure from the committed
CSV with LaTeX-math legend labels. Numbers are untouched (no retraining)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model")
LABELS7 = [r"H$\gamma$", r"[N II]", r"[S II]$_{6717}$", r"[S II]$_{6731}$",
           r"[O II]$_{3726}$", r"[O II]$_{3729}$", r"[O III]"]

tab = pd.read_csv(REPO / "docs" / "balmer_conditioning_results.csv")
per = tab[tab.line != "ALL-7 pooled"].reset_index(drop=True)
assert len(per) == len(LABELS7), f"expected 7 lines, got {len(per)}"

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 11, "ytick.labelsize": 12, "legend.fontsize": 12})
x = np.arange(len(per)); w = 0.27
fig, ax = plt.subplots(figsize=(9.5, 5.0))
ax.bar(x - w, per.base_scatter, w, label=r"baseline ($M_\star$, $L_{\mathrm{H}\alpha}$)", color="#0072B2")
ax.bar(x, per.twostage_scatter, w, label=r"+$R$ predicted (two-stage)", color="#009E73")
ax.bar(x + w, per.oracle_scatter, w, label=r"+$R$ true (oracle)", color="#CC79A7")
ax.set_xticks(x); ax.set_xticklabels(LABELS7, rotation=30, ha="right")
ax.set_ylabel(r"scatter of $\log_{10}(L_{\rm line}/L_{\mathrm{H}\alpha})$  [dex]")
ax.set_title("Effect of conditioning on the Balmer decrement (SDSS in-survey)")
ax.legend(frameon=True, framealpha=0.9)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(REPO / "figs" / f"balmer_conditioning_scatter.{e}", dpi=180, bbox_inches="tight")
print("Rewrote figs/balmer_conditioning_scatter.{png,pdf} with math labels")
