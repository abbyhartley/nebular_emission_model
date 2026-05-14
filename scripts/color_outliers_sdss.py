from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

# -----------------------
# Load SDSS (MPA-JHU+RCSED2) table
# -----------------------
infile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo_selected_test.fits")
t = Table.read(infile, hdu=1)

# -----------------------
# Columns (based on what you listed)
# -----------------------
z = np.asarray(t["Z_1"], dtype=float)

mg = np.asarray(t["corrmag_g"], dtype=float)
mr = np.asarray(t["corrmag_r"], dtype=float)

kg = np.asarray(t["kcorr_g"], dtype=float)
kr = np.asarray(t["kcorr_r"], dtype=float)

# Distance modulus and rest-frame absolute mags
DM = np.asarray(cosmo.distmod(z).value, dtype=float)
Mg = mg - DM - kg
Mr = mr - DM - kr

gmr = Mg - Mr

# Basic finite mask
good = np.isfinite(z) & np.isfinite(gmr) & np.isfinite(kg) & np.isfinite(kr)
z, gmr, kg, kr = z[good], gmr[good], kg[good], kr[good]

# -----------------------
# Define "extreme" outliers (adjust thresholds if desired)
# -----------------------
# A: astrophysically plausible-ish range for SDSS rest g-r
plausible = (gmr > -0.5) & (gmr < 1.6)
outliers = ~plausible

print(f"N total finite: {len(gmr)}")
print(f"N plausible (-0.5<g-r<1.6): {plausible.sum()} ({plausible.mean():.3%})")
print(f"N outliers: {outliers.sum()} ({outliers.mean():.3%})")

# -----------------------
# 1) Histogram (zoom + full)
# -----------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# zoomed histogram
ax[0].hist(gmr, bins=120, range=(-0.5, 1.6), histtype="step", color="k", linewidth=1.5)
ax[0].set_xlabel(r"$(g-r)_\mathrm{rest}$")
ax[0].set_ylabel("N")
ax[0].set_title("SDSS rest-frame g-r (zoomed)")

# full histogram to show extreme tails (use log y so you can see them)
gmin, gmax = np.nanpercentile(gmr, [0.1, 99.9])
ax[1].hist(gmr, bins=200, range=(gmin, gmax), color="steelblue", alpha=0.8)
ax[1].set_yscale("log")
ax[1].set_xlabel(r"$(g-r)_\mathrm{rest}$")
ax[1].set_ylabel("N (log)")
ax[1].set_title(f"g-r (0.1–99.9 percentile), log y\n[{gmin:.2f}, {gmax:.2f}]")

plt.savefig('sdss_g-r_hist.png')

# -----------------------
# 2) g-r vs redshift, colored by kcorr_g and kcorr_r
# -----------------------
# Use small markers + rasterize so it’s fast with ~2e5 points
def scatter_gr_z_colored(z, gmr, color, label, vmin=None, vmax=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    sc = ax.scatter(
        z, gmr,
        c=color, s=3, alpha=0.5,
        cmap=cmap, vmin=vmin, vmax=vmax,
        rasterized=True
    )
    ax.axhline(-0.5, color="k", ls="--", lw=1)
    ax.axhline(1.6, color="k", ls="--", lw=1)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$(g-r)_\mathrm{rest}$")
    ax.set_title(f"SDSS: g-r vs z colored by {label}")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(label)
    plt.savefig('sdss_g-r_{label}_colored.png')

# Pick robust colorbar limits so a few extreme k-corr values don’t wash everything out
kg_lo, kg_hi = np.nanpercentile(kg, [1, 99])
kr_lo, kr_hi = np.nanpercentile(kr, [1, 99])

scatter_gr_z_colored(z, gmr, kg, "kcorr_g", vmin=kg_lo, vmax=kg_hi)
scatter_gr_z_colored(z, gmr, kr, "kcorr_r", vmin=kr_lo, vmax=kr_hi)

# -----------------------
# Optional: highlight outliers explicitly
# -----------------------
fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
ax.scatter(z[plausible], gmr[plausible], s=3, alpha=0.25, color="0.4", rasterized=True, label="plausible")
ax.scatter(z[outliers], gmr[outliers], s=8, alpha=0.8, color="crimson", rasterized=True, label="outlier")
ax.set_xlabel("z")
ax.set_ylabel(r"$(g-r)_\mathrm{rest}$")
ax.set_title("SDSS: g-r vs z with outliers highlighted")
ax.legend(markerscale=2)
plt.savefig('sdss_g-r_outliers.png')
