# corr_noise_diagnosis.py
# Test WHY the SDSS [NII] anticorrelations weakened going per-line S/N 5 -> 3.
# Hypothesis: measurement-noise dilution from the faint (S/N 3-5) galaxies added.
# Two decisive tests, data only (no flow):
#   (1) correlation vs line-S/N bin  (should climb with S/N if noise-driven)
#   (2) noise disattenuation: corr_true = corr_obs / sqrt((1-fx)(1-fy)), f = noise_var/obs_var;
#       if disattenuated full-S/N>3 corr ~= strict S/N>5 corr -> noise dilution confirmed.
from pathlib import Path
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 14, "axes.titlesize": 13, "legend.fontsize": 10,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})
LN10 = np.log(10.0)

CFG = {
    "SDSS": dict(fits=BASE + "SDSS_main_training_data_ALTB.fits", kind="err", ha="H_ALPHA_FLUX",
                 lines=["H_BETA_FLUX", "H_GAMMA_FLUX", "NII_6584_FLUX", "SII_6717_FLUX", "SII_6731_FLUX",
                        "OII_3726_FLUX", "OII_3729_FLUX", "OIII_5007_FLUX"],
                 nii="NII_6584_FLUX", oiii="OIII_5007_FLUX", hb="H_BETA_FLUX", oii="OII_3726_FLUX"),
    "DESI": dict(fits=BASE + "DESI_BGS_training_data_ALTB.fits", kind="ivar", ha="HALPHA_FLUX",
                 lines=["HBETA_FLUX", "HGAMMA_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX",
                        "OII_3726_FLUX", "OII_3729_FLUX", "OIII_5007_FLUX"],
                 nii="NII_6584_FLUX", oiii="OIII_5007_FLUX", hb="HBETA_FLUX", oii="OII_3726_FLUX"),
}
PAIRS = [("[NII]-[OIII]", "oiii"), ("[NII]-Hβ", "hb"), ("[NII]-[OII]3726", "oii")]


def snr(flux, unc, kind):
    flux = np.asarray(flux, float); unc = np.asarray(unc, float)
    ok = np.isfinite(flux) & np.isfinite(unc) & (flux > 0) & (unc > 0)
    s = np.full(flux.shape, np.nan)
    s[ok] = flux[ok] / unc[ok] if kind == "err" else flux[ok] * np.sqrt(unc[ok])
    return s


def logratio(line, ha):
    line = np.asarray(line, float); ha = np.asarray(ha, float)
    out = np.full(line.shape, np.nan); m = (line > 0) & (ha > 0)
    out[m] = np.log10(line[m]) - np.log10(ha[m])
    return out


def corr(x, y, mask):
    m = mask & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1]


def load(cfg):
    t = Table.read(cfg["fits"], hdu=1)
    df = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
    suf = "_ERR" if cfg["kind"] == "err" else "_IVAR"
    fha = df[cfg["ha"]].to_numpy(float); sha = snr(fha, df[cfg["ha"] + suf].to_numpy(float), cfg["kind"])
    d = {"fha": fha, "sha": sha}
    snrs = np.column_stack([snr(df[c].to_numpy(float), df[c + suf].to_numpy(float), cfg["kind"]) for c in cfg["lines"]])
    d["min8"] = np.nanmin(snrs, axis=1)
    for key in ["nii", "oiii", "hb", "oii"]:
        col = cfg[key]
        d["r_" + key] = logratio(df[col].to_numpy(float), fha)
        d["sn_" + key] = snr(df[col].to_numpy(float), df[col + suf].to_numpy(float), cfg["kind"])
    return d


def noise_var_logratio(sn_line, sn_ha):
    # var of log10(line/Ha) from photon noise: (1/ln10)^2 (1/SNR_line^2 + 1/SNR_Ha^2)
    return (1.0 / LN10**2) * (1.0 / sn_line**2 + 1.0 / sn_ha**2)


def disattenuate(rx, ry, snx, sny, sha, mask):
    m = mask & np.isfinite(rx) & np.isfinite(ry) & np.isfinite(snx) & np.isfinite(sny) & np.isfinite(sha)
    cobs = np.corrcoef(rx[m], ry[m])[0, 1]
    fx = np.mean(noise_var_logratio(snx[m], sha[m])) / np.var(rx[m])
    fy = np.mean(noise_var_logratio(sny[m], sha[m])) / np.var(ry[m])
    fx = min(fx, 0.95); fy = min(fy, 0.95)
    return cobs, cobs / np.sqrt((1 - fx) * (1 - fy)), fx, fy


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    sn_edges = np.array([3, 4, 5, 6, 8, 11, 16, 25, 45])
    for ax, tag in zip(axes, ["SDSS", "DESI"]):
        d = load(CFG[tag])
        strict = d["min8"] > 5
        full = d["min8"] > 3
        print(f"\n===== {tag}  (N full S/N>3 = {int(np.isfinite(d['min8'])[full].sum() if False else full.sum()):,}, "
              f"strict S/N>5 = {int(strict.sum()):,}) =====", flush=True)
        for label, ykey in PAIRS:
            c5 = corr(d["r_nii"], d["r_" + ykey], strict)
            c3 = corr(d["r_nii"], d["r_" + ykey], full)
            cobs, cdis, fx, fy = disattenuate(d["r_nii"], d["r_" + ykey], d["sn_nii"], d["sn_" + ykey], d["sha"], full)
            print(f"  {label:14s}  strict(S/N>5)={c5:+.3f}   full(S/N>3)={c3:+.3f}   "
                  f"full disattenuated={cdis:+.3f}   (noise frac NII={fx:.2f}, {ykey}={fy:.2f})", flush=True)
            # correlation vs min-line-S/N bin
            xc, yc = [], []
            for lo, hi in zip(sn_edges[:-1], sn_edges[1:]):
                b = full & (d["min8"] >= lo) & (d["min8"] < hi)
                cc = corr(d["r_nii"], d["r_" + ykey], b)
                if np.isfinite(cc):
                    xc.append(0.5 * (lo + hi)); yc.append(cc)
            ax.plot(xc, yc, "o-", label=label)
            ax.axhline(c5, ls=":", lw=1, color=ax.lines[-1].get_color(), alpha=0.6)
        ax.set_xscale("log"); ax.set_xlabel("min per-line S/N (bin center)")
        ax.set_ylabel("correlation with [NII]/H$\\alpha$"); ax.set_title(tag)
        ax.axhline(0, color="0.7", lw=0.8); ax.legend()
    fig.suptitle("Correlation with [NII] vs S/N (dotted = strict S/N>5 value): rising with S/N = noise dilution",
                 fontsize=13)
    out = REPO + "figs_ALTB/corr_noise_vs_snr.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("\nSaved:", out, flush=True)


if __name__ == "__main__":
    main()
