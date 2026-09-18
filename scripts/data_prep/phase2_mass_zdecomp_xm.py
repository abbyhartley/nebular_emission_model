"""
Phase 2 (cross-matched sample):
 (A) DIRECT measurement of the M* zero-point: DESI color-mass vs SDSS color-mass
     for the SAME galaxies (the real offset, vs the degenerate +-0.6 dex sweep).
 (B) z vs [NII]/Ha and [OIII]/Hb for the SAME galaxies, SDSS vs DESI -- should
     nearly overlap (offset = same-galaxy reporting), unlike the population plot.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model")
df = pd.read_csv(REPO / "docs" / "crossmatch_sdss_desi_fluxes.csv")
C = {"sdss": "#0072B2", "desi": "#E69F00"}

# (A) mass zero-point
dm = (df["logm_desi"] - df["logm_sdss"]).to_numpy()
dm = dm[np.isfinite(dm)]
dm_med = np.median(dm); dm_nmad = 1.4826 * np.median(np.abs(dm - dm_med))
lha_off = np.median(np.log10(df["desi_Halpha"] / df["sdss_Halpha"]).replace([np.inf, -np.inf], np.nan).dropna())
print(f"(A) M* zero-point  median(logM_desi - logM_sdss) = {dm_med:+.3f} dex  (NMAD {dm_nmad:.3f})")
print(f"    L_Ha zero-point median(log Ha_desi/Ha_sdss)   = {lha_off:+.3f} dex")

# (B) ratios per survey, same galaxies
def logratio(num_s, num_d, den_s, den_d):
    rs = np.log10(df[num_s] / df[den_s]); rd = np.log10(df[num_d] / df[den_d])
    return rs.to_numpy(), rd.to_numpy()

nii_s, nii_d = logratio("sdss_NII6584", "desi_NII6584", "sdss_Halpha", "desi_Halpha")
o3_s, o3_d = logratio("sdss_OIII5007", "desi_OIII5007", "sdss_Hbeta", "desi_Hbeta")
z = df["z_sdss"].to_numpy()
zb = np.linspace(0.04, 0.12, 6)


def binmed(x, y):
    g = np.isfinite(x) & np.isfinite(y) & (x > zb[0]) & (x < zb[-1])
    m, e, _ = binned_statistic(x[g], y[g], "median", bins=zb)
    c, _, _ = binned_statistic(x[g], y[g], "count", bins=zb)
    m[c < 40] = np.nan
    return 0.5 * (e[:-1] + e[1:]), m


print(f"(B) same-galaxy median offsets: [NII]/Ha (D-S) = {np.nanmedian(nii_d-nii_s):+.3f} "
      f"(pop +0.174) ; [OIII]/Hb = {np.nanmedian(o3_d-o3_s):+.3f} (pop -0.126)")

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 14, "axes.titlesize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 12})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

ax[0].hist(dm, bins=np.linspace(-0.6, 0.6, 61), color="0.5", alpha=0.8)
ax[0].axvline(0, color="k", ls=":", lw=1)
ax[0].axvline(dm_med, color="#CC79A7", lw=2, label=f"median {dm_med:+.3f} dex")
ax[0].set_xlabel(r"$\log M_\star^{\rm DESI} - \log M_\star^{\rm SDSS}$ [dex]")
ax[0].set_ylabel("galaxies"); ax[0].set_title("(A) Measured M$_\\star$ zero-point (same galaxies)")
ax[0].legend()

for a, (ss, dd, lab) in zip(ax[1:], [(nii_s, nii_d, r"$\log$([N II]/H$\alpha$)"),
                                     (o3_s, o3_d, r"$\log$([O III]/H$\beta$)")]):
    xs, ms = binmed(z, ss); xd, md = binmed(z, dd)
    a.plot(xs, ms, "-o", color=C["sdss"], lw=2.4, ms=6, label="SDSS")
    a.plot(xd, md, "-o", color=C["desi"], lw=2.4, ms=6, label="DESI")
    a.set_xlabel("redshift"); a.set_ylabel("median " + lab)
    a.legend(title="same galaxies")
ax[1].set_title("(B) [N II]/H$\\alpha$ vs z"); ax[2].set_title("(B) [O III]/H$\\beta$ vs z")
fig.suptitle("Cross-matched (same-galaxy) SDSS vs DESI: mass zero-point and line ratios", fontsize=13)
for e in ("png", "pdf"):
    fig.savefig(REPO / "figs" / f"crossmatch_mass_zdecomp.{e}", dpi=170, bbox_inches="tight")
print("Wrote figs/crossmatch_mass_zdecomp.png")
