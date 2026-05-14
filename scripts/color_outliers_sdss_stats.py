from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

# your calibrated relation
from normflow.stellar_mass import log10_ml_r_from_gmr_sdss


# -----------------------
# Load SDSS (MPA-JHU+RCSED2) table
# -----------------------
infile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo_selected_test.fits")
t = Table.read(infile, hdu=1)

# -----------------------
# Pull columns
# -----------------------
z  = np.asarray(t["Z_1"], dtype=float)
mg = np.asarray(t["corrmag_g"], dtype=float)
mr = np.asarray(t["corrmag_r"], dtype=float)
kg = np.asarray(t["kcorr_g"], dtype=float)
kr = np.asarray(t["kcorr_r"], dtype=float)

# Distance modulus + absolute mags
DM = np.asarray(cosmo.distmod(z).value, dtype=float)
Mg = mg - DM - kg
Mr = mr - DM - kr

# Color + stellar mass
gmr = Mg - Mr
Msun_r = 4.64
logLr = -0.4 * (Mr - Msun_r)
logMLr = np.asarray(log10_ml_r_from_gmr_sdss(gmr), dtype=float)
logM_color = logMLr + logLr

# -----------------------
# Master finite mask
# -----------------------
good = (
    np.isfinite(z) & np.isfinite(mg) & np.isfinite(mr) &
    np.isfinite(kg) & np.isfinite(kr) &
    np.isfinite(Mg) & np.isfinite(Mr) &
    np.isfinite(gmr) & np.isfinite(logM_color)
)

z, mg, mr, kg, kr, Mg, Mr, gmr, logM_color = (
    z[good], mg[good], mr[good], kg[good], kr[good],
    Mg[good], Mr[good], gmr[good], logM_color[good]
)

print("N finite all ingredients:", len(z))

# -----------------------
# Define outliers (tune if you want)
# -----------------------
# Color validity range for your ML relation (and generally physical SDSS colors)
color_out = (gmr < -0.5) | (gmr > 1.6)

# Absolute magnitude sanity (very loose; main galaxies usually -23 < Mr < -17)
Mr_out = (Mr < -25) | (Mr > -14)

# Mass sanity (very loose)
mass_out = (logM_color < 7.0) | (logM_color > 12.5)

print("\nOutlier counts:")
print(" color_out:", int(color_out.sum()))
print(" Mr_out:   ", int(Mr_out.sum()))
print(" mass_out: ", int(mass_out.sum()))

# -----------------------
# (1) Overlap checks: are extreme masses the same sources as extreme colors/Mr?
# -----------------------
def overlap_report(a, b, name_a, name_b):
    ab = np.sum(a & b)
    na = np.sum(a)
    nb = np.sum(b)
    print(f"\nOverlap {name_a} vs {name_b}:")
    print(" N(a):", int(na), " N(b):", int(nb), " N(a&b):", int(ab))
    print(" frac of a explained by b:", (ab / na) if na else np.nan)
    print(" frac of b explained by a:", (ab / nb) if nb else np.nan)

overlap_report(mass_out, color_out, "mass_out", "color_out")
overlap_report(mass_out, Mr_out,   "mass_out", "Mr_out")

# Also check: do mass outliers have extreme g-r even if not beyond [-0.5,1.6]?
# Use a robust z-score based on median/MAD
def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad

gmr_rz = robust_z(gmr)
Mr_rz  = robust_z(Mr)
mass_rz = robust_z(logM_color)

print("\nRobust-z summary for mass_out subset:")
if np.sum(mass_out) > 0:
    print(" |gmr_rz| median among mass_out:", float(np.median(np.abs(gmr_rz[mass_out]))))
    print(" |Mr_rz|  median among mass_out:", float(np.median(np.abs(Mr_rz[mass_out]))))

# -----------------------
# (2) Verify root cause: inspect ingredients for the extreme mass objects
# -----------------------
# Grab the worst N objects by mass (both tails)
Nshow = 50
idx_hi = np.argsort(logM_color)[-Nshow:]
idx_lo = np.argsort(logM_color)[:Nshow]
idx = np.unique(np.concatenate([idx_lo, idx_hi]))

df_ext = pd.DataFrame({
    "z": z[idx],
    "corrmag_g": mg[idx],
    "corrmag_r": mr[idx],
    "kcorr_g": kg[idx],
    "kcorr_r": kr[idx],
    "DM": DM[good][idx],      # DM recomputed already but DM[good] is aligned; use this to inspect too
    "Mg": Mg[idx],
    "Mr": Mr[idx],
    "gmr": gmr[idx],
    "logML_r": logMLr[good][idx],
    "logLr": logLr[good][idx],
    "logM_color": logM_color[idx],
})

# Sort by mass and print
df_ext_sorted = df_ext.sort_values("logM_color")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("\nExtreme mass objects (lowest then highest):")
print(df_ext_sorted.to_string(index=False))

# Simple “broken ingredient” flags (tune thresholds after you look once)
flag_bad_k = (np.abs(df_ext["kcorr_g"]) > 5) | (np.abs(df_ext["kcorr_r"]) > 5)
flag_bad_z = (df_ext["z"] <= 0) | (df_ext["z"] > 1.0)
flag_bad_m = (df_ext["corrmag_g"] <= 0) | (df_ext["corrmag_r"] <= 0) | (df_ext["corrmag_g"] > 40) | (df_ext["corrmag_r"] > 40)

print("\nAmong displayed extremes:")
print(" bad k-corr count:", int(flag_bad_k.sum()))
print(" bad z count:     ", int(flag_bad_z.sum()))
print(" bad mag count:   ", int(flag_bad_m.sum()))

# -----------------------
# Visuals: where do mass outliers live in (g-r, Mr, z) and k-corr space?
# -----------------------
# A) g-r vs z, outliers highlighted, colored by kcorr_g
plt.figure(figsize=(7.5, 5), constrained_layout=True)
plt.scatter(z[~mass_out], gmr[~mass_out], s=3, alpha=0.15, color="0.5", rasterized=True, label="not mass_out")
sc = plt.scatter(z[mass_out], gmr[mass_out], c=kg[mass_out], s=25, alpha=0.9,
                 cmap="viridis", edgecolor="k", linewidth=0.2, rasterized=True, label="mass_out")
plt.axhline(-0.5, color="k", ls="--", lw=1)
plt.axhline(1.6, color="k", ls="--", lw=1)
plt.xlabel("z"); plt.ylabel(r"$(g-r)_\mathrm{rest}$")
plt.title("Mass outliers in color–redshift space (colored by kcorr_g)")
cb = plt.colorbar(sc); cb.set_label("kcorr_g")
plt.legend()
plt.show()

# B) kcorr_g vs kcorr_r, outliers highlighted
plt.figure(figsize=(6.5, 5), constrained_layout=True)
plt.scatter(kg[~mass_out], kr[~mass_out], s=3, alpha=0.15, color="0.6", rasterized=True, label="not mass_out")
plt.scatter(kg[mass_out], kr[mass_out], s=30, alpha=0.9, color="crimson",
            edgecolor="k", linewidth=0.2, rasterized=True, label="mass_out")
plt.xlabel("kcorr_g"); plt.ylabel("kcorr_r")
plt.title("k-corrections: mass outliers highlighted")
plt.axvline(0, color="k", lw=0.5); plt.axhline(0, color="k", lw=0.5)
plt.legend()
plt.show()

# C) Mr vs z, outliers highlighted
plt.figure(figsize=(7.5, 5), constrained_layout=True)
plt.scatter(z[~mass_out], Mr[~mass_out], s=3, alpha=0.15, color="0.5", rasterized=True)
plt.scatter(z[mass_out], Mr[mass_out], s=25, alpha=0.9, color="crimson",
            edgecolor="k", linewidth=0.2, rasterized=True)
plt.gca().invert_yaxis()
plt.xlabel("z"); plt.ylabel(r"$M_r$")
plt.title("Mr vs z (mass outliers highlighted)")
plt.show()
