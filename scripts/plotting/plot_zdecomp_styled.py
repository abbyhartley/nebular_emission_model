"""
Restyle the z-decomposition figure in the house style (SciencePlots + Okabe-Ito,
matching the M* sweep figure), from the committed docs/zdecomp_ratio_vs_z.csv.
Two panels: median log([NII]/Ha) and log([OIII]/Hb) vs redshift, at fixed
(M*, L_Ha) control box, for SDSS vs DESI. Shaded band = z-overlap [0.07, 0.10].
"""
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model")
df = pd.read_csv(REPO / "docs" / "zdecomp_ratio_vs_z.csv")

C = {"sdss": "#0072B2", "desi": "#E69F00"}      # Okabe-Ito, matches the M* sweep figure
LAB = {"sdss": "SDSS", "desi": "DESI"}
ZB = (0.07, 0.10)                                # z-overlap band

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 16, "axes.titlesize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})

fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)
panels = [("med_NII_Ha", r"median $\log(\mathrm{[N\,II]}/\mathrm{H}\alpha)$"),
          ("med_OIII_Hb", r"median $\log(\mathrm{[O\,III]}/\mathrm{H}\beta)$")]

for ax, (col, ylab) in zip(axes, panels):
    for s in ("sdss", "desi"):
        g = df[df.survey == s].sort_values("z_mid")
        ax.plot(g.z_mid, g[col], "-o", color=C[s], lw=2.6, ms=6,
                label=LAB[s] if col == "med_NII_Ha" else None)
    ax.axvspan(*ZB, color="0.5", alpha=0.10, lw=0)
    ax.set_xlabel("redshift")
    ax.set_ylabel(ylab)
axes[0].legend(frameon=True, framealpha=0.9, title=r"at fixed $(M_\star,\,L_{\mathrm{H}\alpha})$",
               title_fontsize=12, loc="upper right")
from matplotlib.transforms import blended_transform_factory
for ax in axes:
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(np.mean(ZB), 0.50, "z-overlap", transform=tr, rotation=90,
            ha="center", va="center", fontsize=10, color="0.4")

for ext in ("png", "pdf"):
    (REPO / "figs").mkdir(exist_ok=True)
    out = REPO / "figs" / f"zdecomp_ratio_vs_z.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Wrote:", out)
