"""
Verify the sample-size discrepancy (review 2.1 / sample reconciliation).
Reproduces the selection funnel on the PARENT catalogs, contrasting a looser GLOBAL
S/N cut (paper-like) against the current STRICT per-line S/N>5 on all 9 lines.
"""
import numpy as np
from astropy.io import fits

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"

DESI_LINES = ["OII_3726_FLUX","OII_3729_FLUX","HGAMMA_FLUX","HBETA_FLUX","HALPHA_FLUX",
              "OIII_5007_FLUX","NII_6584_FLUX","SII_6716_FLUX","SII_6731_FLUX"]
SDSS_LINES = ["OII_3726_FLUX","OII_3729_FLUX","H_GAMMA_FLUX","H_BETA_FLUX","H_ALPHA_FLUX",
              "OIII_5007_FLUX","NII_6584_FLUX","SII_6717_FLUX","SII_6731_FLUX"]


def count_desi():
    fn = BASE + "fastspec_zall_combined.fits"
    c = dict(total=0, base=0, globalSNR=0, paperlike=0, perline=0)
    spec_present = None
    with fits.open(fn, memmap=True) as h:
        for i in range(1, len(h)):
            if not hasattr(h[i], "columns"):
                continue
            d = h[i].data
            cols = set(h[i].columns.names)
            if spec_present is None:
                spec_present = "SPECTYPE" in cols
            c["total"] += len(d)
            z = np.asarray(d["Z"], float); zw = np.asarray(d["ZWARN"], float)
            snr_r = np.asarray(d["SNR_R"], float)
            base = np.isfinite(z) & (z > 0.05) & np.isfinite(zw) & (zw == 0)
            if "SPECTYPE" in cols:
                base &= (np.asarray(d["SPECTYPE"]).astype(str) == "GALAXY")
            c["base"] += int(base.sum())
            gsnr = base & np.isfinite(snr_r) & (snr_r > 5.0)
            c["globalSNR"] += int(gsnr.sum())
            ha_iv = np.asarray(d["HALPHA_FLUX_IVAR"], float)
            paper = gsnr & (ha_iv > 0)
            for lf in DESI_LINES:
                f = np.asarray(d[lf], float)
                paper &= np.isfinite(f) & (f > 0)
            c["paperlike"] += int(paper.sum())
            strict = gsnr.copy()
            for lf in DESI_LINES:
                f = np.asarray(d[lf], float); iv = np.asarray(d[lf + "_IVAR"], float)
                m = np.isfinite(f) & (f > 0) & np.isfinite(iv) & (iv > 0)
                snr = np.where(m, f * np.sqrt(np.where(iv > 0, iv, np.nan)), -1.0)
                strict &= m & (snr > 5.0)
            c["perline"] += int(strict.sum())
    c["spectype_applied"] = bool(spec_present)
    return c


def count_sdss():
    fn = BASE + "mpa_rcsed2_combo.fits"
    with fits.open(fn, memmap=True) as h:
        d = h[1].data
        cols = set(h[1].columns.names)
        n = len(d)
        z = np.asarray(d["Z_1"], float); zw = np.asarray(d["Z_WARNING"], float)
        snm = np.asarray(d["SN_MEDIAN"], float)
        base = np.isfinite(z) & (z > 0.05) & np.isfinite(zw) & (zw == 0)
        spec_applied = "SPECTROTYPE" in cols
        if spec_applied:
            base &= (np.asarray(d["SPECTROTYPE"]).astype(str) == "GALAXY")
        gsnr = base & np.isfinite(snm) & (snm > 5.0)
        ha = np.asarray(d["H_ALPHA_FLUX"], float)
        paper = gsnr & np.isfinite(ha) & (ha > 0)
        for lf in SDSS_LINES:
            f = np.asarray(d[lf], float)
            paper &= np.isfinite(f) & (f > 0)
        strict = gsnr.copy()
        for lf in SDSS_LINES:
            f = np.asarray(d[lf], float); er = np.asarray(d[lf + "_ERR"], float)
            m = np.isfinite(f) & (f > 0) & np.isfinite(er) & (er > 0)
            snr = np.where(m, f / np.where(er > 0, er, np.nan), -1.0)
            strict &= m & (snr > 5.0)
    return dict(total=n, base=int(base.sum()), globalSNR=int(gsnr.sum()),
                paperlike=int(paper.sum()), perline=int(strict.sum()),
                spectype_applied=spec_applied)


def report(name, c, paper_final):
    t=c['total']; b=c['base']; g=c['globalSNR']; p=c['paperlike']; pl=c['perline']; sp=c['spectype_applied']
    print(f"\n===== {name} funnel =====")
    print(f"  total parent                              : {t:,}")
    print(f"  + type/z>0.05/ZWARN=0  (spectype={sp})     : {b:,}")
    print(f"  + global S/N>5                            : {g:,}")
    print(f"  + Ha ivar>0 & all line flux>0 (LOOSE)     : {p:,}")
    print(f"  + per-line S/N>5 on all 9 (CURRENT CODE)  : {pl:,}")
    print(f"  -> paper quotes final                     : {paper_final:,}")


if __name__ == "__main__":
    report("DESI BGS", count_desi(), 733630)
    report("SDSS main", count_sdss(), 149913)
