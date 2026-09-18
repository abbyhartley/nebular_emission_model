# train_alt.py <infile.fits> <out_flow.eqx> <out_meta.pkl>
# Parametrized copy of train_nf_desi.py (identical config/logic) for alternative-selection
# DESI training samples. Columns match DESI_BGS_training_data.fits.
import sys
from pathlib import Path
import pickle
import numpy as np
import jax.numpy as jnp  # noqa
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
sys.path.insert(0, str(REPO / "src"))
from normflow.train_NF import train_line_ratio_flow

infile = Path(sys.argv[1]); out_flow = Path(sys.argv[2]); out_meta = Path(sys.argv[3])

z_col = "Z"; logm_col = "LOGM_COLOR"; FLUX_SCALE = 1e-17
ha_flux_col = "HALPHA_FLUX"; ha_ivar_col = "HALPHA_FLUX_IVAR"
line_flux_cols = ["HBETA_FLUX", "HGAMMA_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX",
                  "OII_3726_FLUX", "OII_3729_FLUX", "OIII_5007_FLUX"]
line_ivar_cols = [c + "_IVAR" for c in line_flux_cols]
SEED, EPOCHS, BATCH, LR, CLIP = 0, 200, 2048, 3e-4, 1.0


def log10_luminosity_from_flux(z, flux_cgs):
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    return np.log10(flux_cgs) + np.log10(4.0 * np.pi) + 2.0 * np.log10(dl_cm)


def main():
    t = Table.read(infile, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    df = t[names].to_pandas()
    base = np.isfinite(df[z_col].astype(float)) & np.isfinite(df[logm_col].astype(float))
    base &= (df[ha_ivar_col].astype(float) > 0)
    for iv in line_ivar_cols:
        base &= (df[iv].astype(float) > 0)
    df["_HA_FLUX_CGS"] = df[ha_flux_col].astype(float) * FLUX_SCALE
    for c in line_flux_cols:
        df[f"_{c}_CGS"] = df[c].astype(float) * FLUX_SCALE
    base &= np.isfinite(df["_HA_FLUX_CGS"]) & (df["_HA_FLUX_CGS"] > 0)
    for c in line_flux_cols:
        base &= np.isfinite(df[f"_{c}_CGS"]) & (df[f"_{c}_CGS"] > 0)
    df = df.loc[base].copy().reset_index(drop=True)
    df["LOG10_HA_FLUX"] = np.log10(df["_HA_FLUX_CGS"].to_numpy())
    for c in line_flux_cols:
        df[f"LOG10_{c}"] = np.log10(df[f"_{c}_CGS"].to_numpy())
    df["LOG_LHA"] = log10_luminosity_from_flux(df[z_col].to_numpy(float), df["_HA_FLUX_CGS"].to_numpy(float))
    print(f"[{infile.name}] rows after validity: {len(df)}", flush=True)

    line_aliases_log = {
        "hbeta": ["LOG10_HBETA_FLUX"], "hgamma": ["LOG10_HGAMMA_FLUX"], "nii6584": ["LOG10_NII_6584_FLUX"],
        "sii6716": ["LOG10_SII_6716_FLUX"], "sii6731": ["LOG10_SII_6731_FLUX"],
        "oii3726": ["LOG10_OII_3726_FLUX"], "oii3729": ["LOG10_OII_3729_FLUX"], "oiii5007": ["LOG10_OIII_5007_FLUX"]}
    res = train_line_ratio_flow(df, logmstar_col=logm_col, loglha_col="LOG_LHA", logha_col="LOG10_HA_FLUX",
                                line_aliases=line_aliases_log, use_ratios_to_ha=True,
                                seed=SEED, batch_size=BATCH, epochs=EPOCHS, lr=LR, clip=CLIP)
    flow, meta = res["flow"], res["meta"]
    meta["extra"] = dict(infile=str(infile), n_train=len(res["df_train"]), note="alternative-selection DESI flow")
    eqx.tree_serialise_leaves(out_flow, flow)
    with open(out_meta, "wb") as f:
        pickle.dump(meta, f)
    print("Saved:", out_flow, out_meta, flush=True)


if __name__ == "__main__":
    main()
