"""
Phase 1: from the cross-matched (same-galaxy) SDSS/DESI sample, compare (a) per-line
fluxes and (b) per-line RATIOS to Ha -- the quantity the flow actually models.
Key comparison: the same-galaxy ratio offset vs the POPULATION cross-survey shift
(from the z-decomposition: SDSS-DESI [NII]/Ha = -0.174, [OIII]/Hb = +0.126 dex).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model")
df = pd.read_csv(REPO / "docs" / "crossmatch_sdss_desi_fluxes.csv")
LINES = ["Hbeta", "Hgamma", "NII6584", "SII6717", "SII6731", "OII3726", "OII3729", "OIII5007"]
LAB = [r"H$\beta$", r"H$\gamma$", r"[N II]", r"[S II]$_a$", r"[S II]$_b$", r"[O II]$_a$", r"[O II]$_b$", r"[O III]"]


def med_nmad(v):
    v = v[np.isfinite(v)]
    return np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))


def ratio_off(line):
    a = np.log10(df[f"desi_{line}"] / df[f"desi_Halpha"])
    b = np.log10(df[f"sdss_{line}"] / df[f"sdss_Halpha"])
    g = np.isfinite(a) & np.isfinite(b)
    return med_nmad((a - b)[g].to_numpy())


print("=== same-galaxy offsets (DESI - SDSS), matched sample ===")
flux_off = {}
rat_off = {}
for ln in LINES + ["Halpha"]:
    r = np.log10(df[f"desi_{ln}"] / df[f"sdss_{ln}"])
    r = r[np.isfinite(r)]
    flux_off[ln] = np.median(r)
for ln in LINES:
    rat_off[ln] = ratio_off(ln)[0]
    print(f"  {ln:9s}  flux off={flux_off[ln]:+.3f}   ratio(line/Ha) off={rat_off[ln]:+.3f}")

# direct comparison with population shift for the two headline ratios
niiha = ratio_off("NII6584")[0]
oiii_hb_a = np.log10(df["desi_OIII5007"] / df["desi_Hbeta"])
oiii_hb_b = np.log10(df["sdss_OIII5007"] / df["sdss_Hbeta"])
g = np.isfinite(oiii_hb_a) & np.isfinite(oiii_hb_b)
oiiihb = np.median((oiii_hb_a - oiii_hb_b)[g])
print("\n=== SAME-GALAXY vs POPULATION (DESI - SDSS) ===")
print(f"  [NII]/Ha  : same-galaxy {niiha:+.3f} dex   vs population {+0.174:+.3f} dex")
print(f"  [OIII]/Hb : same-galaxy {oiiihb:+.3f} dex   vs population {-0.126:+.3f} dex")

# figure
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 14, "axes.titlesize": 13, "xtick.labelsize": 10, "ytick.labelsize": 11})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
allln = LINES + ["Halpha"]
alllab = LAB + [r"H$\alpha$"]
axes[0].bar(range(len(allln)), [flux_off[l] for l in allln], color="#0072B2")
axes[0].axhline(0, color="0.4", lw=0.8)
axes[0].set_xticks(range(len(allln))); axes[0].set_xticklabels(alllab, rotation=30, ha="right")
axes[0].set_ylabel(r"median $\log_{10}(F_{\rm DESI}/F_{\rm SDSS})$ [dex]")
axes[0].set_title("Per-line FLUX offset (same galaxies)")
axes[1].bar(range(len(LINES)), [rat_off[l] for l in LINES], color="#CC79A7")
axes[1].axhline(0, color="0.4", lw=0.8)
axes[1].set_xticks(range(len(LINES))); axes[1].set_xticklabels(LAB, rotation=30, ha="right")
axes[1].set_ylabel(r"median offset in $\log_{10}(L_{\rm line}/L_{\mathrm{H}\alpha})$ [dex]")
axes[1].set_title("Per-line RATIO-to-H$\\alpha$ offset (what the flow sees)")
fig.suptitle("Same-galaxy SDSS vs DESI reporting (cross-matched, N=%d)" % len(df), fontsize=13)
for e in ("png", "pdf"):
    fig.savefig(REPO / "figs" / f"crossmatch_flux_ratio_offsets.{e}", dpi=170, bbox_inches="tight")
print("Wrote figs/crossmatch_flux_ratio_offsets.png")
