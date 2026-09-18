# build_alt_desi_samples.py
#
# Build two ALTERNATIVE DESI training FITS from the parent, faithfully replicating the
# add_colorSM_to_desifits_v2 recipe (LOGM_COLOR + per-line flux calibration + mass trim),
# but with LOOSER selection cuts (no per-line S/N>5). Subsample each to N_TRAIN so the
# flow comparison isolates the SELECTION, not sample size.
#
#   ALT-A: base & z>0.05 & cont S/N>3 & Halpha S/N>7 & all-8 lines detected (IVAR>0,flux>0)
#   ALT-B: base & z>0.05 & cont S/N>3 & per-line S/N>3 on all 9 lines
# base = SURVEY=main & PROGRAM=bright & ZWARN=0 & finite z>0.
# S/N uses RAW flux/ivar (as in selection.py); flux calibration applied AFTER (as in recipe).
# Output cols match train_nf_desi.py expectations: Z, LOGM_COLOR, <LINE>_FLUX(+_IVAR), HALPHA*.

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table

BASE = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs")
DESI_FULL = BASE / "fastspec_zall_combined.fits"
OUT_A = BASE / "DESI_ALT_A.fits"
OUT_B = BASE / "DESI_ALT_B.fits"

N_TRAIN = 130000
SEED = 0
Z_MIN, CONT_MIN, HA_MIN_A, LINE_MIN_B = 0.05, 3.0, 7.0, 3.0
Msun_r, ZP = 4.64, 0.271

# order matches DESI_FLUX below; index 4 = Halpha
DESI_FLUX = ["OII_3726_FLUX", "OII_3729_FLUX", "HGAMMA_FLUX", "HBETA_FLUX", "HALPHA_FLUX",
             "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX"]
HA_I = 4
NONHA = [0, 1, 2, 3, 5, 6, 7, 8]
FLUX_CAL_DEX = {"OII_3726_FLUX": 0.0650, "OII_3729_FLUX": 0.0075, "HGAMMA_FLUX": 0.0676,
                "HBETA_FLUX": 0.0628, "HALPHA_FLUX": 0.0752, "OIII_5007_FLUX": 0.0502,
                "NII_6584_FLUX": 0.0449, "SII_6716_FLUX": 0.0635, "SII_6731_FLUX": 0.0545}


def stream_pool():
    """Collect raw flux(9), ivar(9), g01, r01, z for galaxies in (base & z>0.05 & cont>3)
    that fall in ALT-A OR ALT-B, plus their membership flags."""
    keep_flux, keep_ivar, keep_g, keep_r, keep_z, keep_A, keep_B = ([] for _ in range(7))
    need = DESI_FLUX + [c + "_IVAR" for c in DESI_FLUX] + \
           ["Z", "ZWARN", "SNR_R", "SURVEY", "PROGRAM", "ABSMAG01_SDSS_G", "ABSMAG01_SDSS_R"]
    with fits.open(DESI_FULL, memmap=True) as hdul:
        for hi in range(1, len(hdul)):
            h = hdul[hi]
            if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                continue
            d = h.data
            if d is None or any(c not in d.names for c in need):
                continue
            n = len(d["Z"])
            flux = np.zeros((n, 9), np.float32); ivar = np.zeros((n, 9), np.float32)
            for i, c in enumerate(DESI_FLUX):
                flux[:, i] = np.asarray(d[c], np.float32)
                ivar[:, i] = np.asarray(d[c + "_IVAR"], np.float32)
            snr = np.where((flux > 0) & (ivar > 0) & np.isfinite(flux) & np.isfinite(ivar),
                           flux * np.sqrt(np.clip(ivar, 0, None)), 0.0)
            z = np.asarray(d["Z"], float); zwarn = np.asarray(d["ZWARN"], float)
            cont = np.asarray(d["SNR_R"], float)
            sv = np.char.lower(np.char.strip(np.asarray(d["SURVEY"]).astype(str)))
            pg = np.char.lower(np.char.strip(np.asarray(d["PROGRAM"]).astype(str)))
            base = (sv == "main") & (pg == "bright") & np.isfinite(z) & (z > Z_MIN) & \
                   np.isfinite(zwarn) & (zwarn == 0) & (cont > CONT_MIN)
            det8 = (snr[:, NONHA] > 0).all(1)
            A = base & (snr[:, HA_I] > HA_MIN_A) & det8
            B = base & (snr > LINE_MIN_B).all(1)
            u = A | B
            if not u.any():
                continue
            keep_flux.append(flux[u]); keep_ivar.append(ivar[u])
            keep_g.append(np.asarray(d["ABSMAG01_SDSS_G"], np.float32)[u])
            keep_r.append(np.asarray(d["ABSMAG01_SDSS_R"], np.float32)[u])
            keep_z.append(z[u].astype(np.float32)); keep_A.append(A[u]); keep_B.append(B[u])
    return (np.concatenate(keep_flux), np.concatenate(keep_ivar), np.concatenate(keep_g),
            np.concatenate(keep_r), np.concatenate(keep_z), np.concatenate(keep_A), np.concatenate(keep_B))


def build_table(flux, ivar, g01, r01, z):
    """Compute LOGM_COLOR, trim, calibrate fluxes; return astropy Table (train-ready)."""
    logM = (1.062 * (g01 - r01) - 0.555) + (-0.4 * (r01 - Msun_r)) + ZP
    finite = np.isfinite(logM)
    lo, hi = np.percentile(logM[finite], [0.1, 99.9])
    keep = finite & (logM >= lo) & (logM <= hi)
    flux, ivar, z, logM = flux[keep], ivar[keep], z[keep], logM[keep]
    cols = {"Z": z.astype(np.float32), "LOGM_COLOR": logM.astype(np.float32)}
    for i, c in enumerate(DESI_FLUX):
        cols[c] = (flux[:, i] * 10.0 ** FLUX_CAL_DEX[c]).astype(np.float32)   # calibrated
        cols[c + "_IVAR"] = ivar[:, i]                                         # raw ivar (validity)
    return Table(cols)


def subsample(tab, n, seed):
    if len(tab) <= n:
        return tab
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tab), size=n, replace=False)
    return tab[np.sort(idx)]


def main():
    print("streaming parent...", flush=True)
    flux, ivar, g01, r01, z, A, B = stream_pool()
    print(f"pool={len(z):,}  ALT-A pre-trim={int(A.sum()):,}  ALT-B pre-trim={int(B.sum()):,}", flush=True)
    for tag, mask, out in [("ALT-A", A, OUT_A), ("ALT-B", B, OUT_B)]:
        tab = build_table(flux[mask], ivar[mask], g01[mask], r01[mask], z[mask])
        print(f"{tag}: after mass-trim {len(tab):,}", flush=True)
        tab = subsample(tab, N_TRAIN, SEED)
        tab.write(out, format="fits", overwrite=True)
        print(f"{tag}: wrote {len(tab):,} rows -> {out}  "
              f"(LOGM_COLOR {np.min(tab['LOGM_COLOR']):.2f}-{np.max(tab['LOGM_COLOR']):.2f}, "
              f"z {np.min(tab['Z']):.3f}-{np.max(tab['Z']):.3f})", flush=True)


if __name__ == "__main__":
    main()
