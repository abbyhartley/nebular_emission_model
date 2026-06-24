# compare_datasets_noNF.py
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.stats import ks_2samp, wasserstein_distance

FLUX_SCALE = 1e-17

LINE_ORDER = [
    ("Hbeta",    ["HBETA_FLUX", "H_BETA_FLUX"]),
    ("Hgamma",   ["HGAMMA_FLUX", "H_GAMMA_FLUX"]),
    ("NII6584",  ["NII_6584_FLUX"]),
    ("SII671x",  ["SII_6716_FLUX", "SII_6717_FLUX"]),
    ("SII6731",  ["SII_6731_FLUX"]),
    ("OII3726",  ["OII_3726_FLUX"]),
    ("OII3729",  ["OII_3729_FLUX"]),
    ("OIII5007", ["OIII_5007_FLUX"]),
]

def resolve_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not resolve any of: {candidates}")

def load_scalar_df(fits_path):
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()

def compute_log_flux_ratios(df, *, survey):
    """
    Returns:
      X: (N,8) array of log10(F_line/F_Ha)
      colmap: resolved raw column names for ha and each line
    Masking:
      - requires z finite (only for consistency; not used here)
      - requires ha>0 and line>0 to take log10
      - DESI: additionally requires *_FLUX_IVAR>0 when present
    """
    survey = survey.lower()
    if survey == "sdss":
        z_col = "Z_1"
        ha_col = "H_ALPHA_FLUX"
        require_ivar = False
        ha_ivar = None
        line_ivars = None
    elif survey == "desi":
        z_col = "Z"
        ha_col = "HALPHA_FLUX"
        require_ivar = True
        ha_ivar = "HALPHA_FLUX_IVAR"
        # try to use these if present
        line_ivars = {
            "HBETA_FLUX": "HBETA_FLUX_IVAR",
            "HGAMMA_FLUX": "HGAMMA_FLUX_IVAR",
            "NII_6584_FLUX": "NII_6584_FLUX_IVAR",
            "SII_6716_FLUX": "SII_6716_FLUX_IVAR",
            "SII_6731_FLUX": "SII_6731_FLUX_IVAR",
            "OII_3726_FLUX": "OII_3726_FLUX_IVAR",
            "OII_3729_FLUX": "OII_3729_FLUX_IVAR",
            "OIII_5007_FLUX": "OIII_5007_FLUX_IVAR",
        }
    else:
        raise ValueError("survey must be 'sdss' or 'desi'")

    # resolve line columns
    line_cols = []
    for _, aliases in LINE_ORDER:
        line_cols.append(resolve_col(df, aliases))

    colmap = dict(z=z_col, ha=ha_col, lines=line_cols)

    z = df[z_col].to_numpy(dtype=float) if z_col in df.columns else np.full(len(df), np.nan)
    ha = df[ha_col].to_numpy(dtype=float)

    mask = np.isfinite(ha) & np.isfinite(z)
    if require_ivar and (ha_ivar in df.columns):
        mask &= (df[ha_ivar].to_numpy(dtype=float) > 0)

    # require positive to take log
    ha_cgs = ha * FLUX_SCALE
    mask &= (ha_cgs > 0)

    logs = {}
    logs["ha"] = np.full(len(df), np.nan, dtype=float)
    logs["ha"][mask] = np.log10(ha_cgs[mask])

    for c in line_cols:
        x = df[c].to_numpy(dtype=float) * FLUX_SCALE
        m = mask & np.isfinite(x) & (x > 0)

        # DESI ivar if available for the specific column name
        if require_ivar and line_ivars is not None:
            # map by canonical DESI name if possible
            # if we resolved to SDSS-style names (unlikely for DESI), just skip ivar
            if c in line_ivars and (line_ivars[c] in df.columns):
                m &= (df[line_ivars[c]].to_numpy(dtype=float) > 0)

        logs[c] = np.full(len(df), np.nan, dtype=float)
        logs[c][m] = np.log10(x[m])

    # final mask: need ha log and all 8 line logs finite
    fin = np.isfinite(logs["ha"])
    for c in line_cols:
        fin &= np.isfinite(logs[c])

    # build X ratios
    X = np.vstack([logs[c][fin] - logs["ha"][fin] for c in line_cols]).T  # (N,8)
    return X, colmap

def summarize_X(X, label):
    print(f"\n=== {label} ===")
    print("N:", X.shape[0])
    for j, (lname, _) in enumerate(LINE_ORDER):
        p1, p50, p99 = np.percentile(X[:, j], [1, 50, 99])
        print(f"{lname:8s}  ratio p1/p50/p99 = {p1: .3f}  {p50: .3f}  {p99: .3f}")

def compare_A_vs_B(XA, XB, labelA, labelB, seed=0, n_match=100000):
    """
    Dataset-level comparison per line:
      - KS statistic + p-value
      - Wasserstein-1 distance (dex)
      - median shift (B-A) after random size-matched sampling
      - robust scatter of (B-A): 0.5*(p84-p16)
    """
    rng = np.random.default_rng(seed)
    n = min(len(XA), len(XB), n_match)
    ia = rng.choice(len(XA), size=n, replace=False)
    ib = rng.choice(len(XB), size=n, replace=False)
    A = XA[ia]
    B = XB[ib]

    rows = []
    for j, (lname, _) in enumerate(LINE_ORDER):
        a = A[:, j]; b = B[:, j]
        ks = ks_2samp(a, b)
        w1 = wasserstein_distance(a, b)
        diff = b - a
        med = np.median(diff)
        p16, p84 = np.percentile(diff, [16, 84])
        scat = 0.5 * (p84 - p16)
        rows.append([lname, ks.statistic, ks.pvalue, w1, med, scat])

    out = pd.DataFrame(rows, columns=[
        "line", "KS_stat", "KS_pvalue", "Wasserstein1_dex", "median_shift_(B-A)_dex", "scatter_diff_dex"
    ])

    print(f"\n=== Compare {labelA} vs {labelB} (n={n} matched) ===")
    print(out.to_string(index=False))
    return out

def main():
    sdss_path = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")
    desi_path = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")
    desi_sdsslike_path = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/desi_bgs_sdss_like_rlt17p77_with_LOGM_COLOR.fits")

    # load
    df_sdss = load_scalar_df(sdss_path)
    df_desi = load_scalar_df(desi_path)
    df_desi_sl = load_scalar_df(desi_sdsslike_path)

    # compute ratios
    X_sdss, _ = compute_log_flux_ratios(df_sdss, survey="sdss")
    X_desi, _ = compute_log_flux_ratios(df_desi, survey="desi")
    X_desi_sl, _ = compute_log_flux_ratios(df_desi_sl, survey="desi")

    # summaries
    summarize_X(X_sdss, "SDSS training subset")
    summarize_X(X_desi, "DESI training subset")
    summarize_X(X_desi_sl, "DESI SDSS-like subset")

    # comparisons
    out1 = compare_A_vs_B(X_sdss, X_desi, "SDSS", "DESI", seed=0)
    out2 = compare_A_vs_B(X_sdss, X_desi_sl, "SDSS", "DESI_SDSSLIKE", seed=1)
    out3 = compare_A_vs_B(X_desi, X_desi_sl, "DESI", "DESI_SDSSLIKE", seed=2)

    outcsv = Path("dataset_comparison_noNF_ratio_space.csv")
    pd.concat(
        [out1.assign(comp="SDSS_vs_DESI"),
         out2.assign(comp="SDSS_vs_DESI_SDSSLIKE"),
         out3.assign(comp="DESI_vs_DESI_SDSSLIKE")],
        ignore_index=True
    ).to_csv(outcsv, index=False)
    print("\nWrote:", outcsv)

if __name__ == "__main__":
    main()
