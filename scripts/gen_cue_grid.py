"""
Generate a Cue (Li+2025, via Tengri) nebular grid in the dust-insensitive ratio
space, VARYING N/O (gas_logno) in addition to logU, logZ_gas, and density (gas_logn)
that Byler's fixed-abundance stellar grid could not. Ionizing spectrum + C/O held at
fiducial; DIG/escape nuisances zeroed for clean HII-region ratios.

Efficiency: build one model per (N/O, n_H) with logU, logZ_gas FREE, then vmap over
the (logU, logZ_gas) grid -> few JIT compiles instead of thousands.
"""
import jax, jax.numpy as jnp, numpy as np, pandas as pd, tengri
F = tengri.Fixed
OUT = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/docs/cue_grid.csv"

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
LOGU  = np.linspace(-4.0, -1.8, 12)
LOGZ  = np.linspace(-1.5,  0.4, 16)
LOGNO = np.linspace(-2.0, 0.0, 8)      # log(N/O): key new axis (extended to super-solar)
LOGN  = np.array([0.5, 1.5, 2.5])      # gas density
KEY = jax.random.PRNGKey(0)
UU, ZZ = np.meshgrid(LOGU, LOGZ, indexing="ij")
uflat, zflat = jnp.asarray(UU.ravel()), jnp.asarray(ZZ.ravel())

def build(logno, logn):
    return tengri.SEDModel.build(
        ssp,
        sfh={"type": "dpl", "*": tengri.FIXED, "alpha": 1.0, "beta": 2.5, "tau_gyr": 0.05, "log_total_mass": 10.0},
        dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
        neb={"type": "cue", "gas_logno": F(float(logno)), "gas_logn": F(float(logn))},  # logU, logZ_gas FREE
        redshift=F(0.05),
    )

rows = []
for logn in LOGN:
    for logno in LOGNO:
        m = build(logno, logn)
        template = dict(m.spec.sample(KEY))
        for k in ["neb_dig_frac", "neb_fesc", "neb_fesc_lya", "neb_fdust"]:
            if k in template:
                template[k] = jnp.asarray(0.0)
        def pred(u, z, tmpl=template, mod=m):
            p = dict(tmpl); p["neb_logU"] = u; p["neb_logZ_gas"] = z
            L = mod.predict(p).lines
            return jnp.array([L.halpha, L.hbeta, L.nii_6584, L.oiii_5007, L.sii_6717, L.sii_6731])
        out = np.asarray(jax.vmap(pred)(uflat, zflat))     # (N, 6)
        ha, hb, nii, oiii, s16, s31 = out.T
        g = (ha > 0) & (hb > 0) & (nii > 0) & (oiii > 0) & (s16 > 0) & (s31 > 0)
        for i in np.where(g)[0]:
            rows.append(dict(logu=float(UU.ravel()[i]), logz=float(ZZ.ravel()[i]),
                             logno=float(logno), logn=float(logn),
                             nii_ha=float(np.log10(nii[i]/ha[i])),
                             oiii_hb=float(np.log10(oiii[i]/hb[i])),
                             sii_ha=float(np.log10((s16[i]+s31[i])/ha[i]))))
        print(f"  logn={logn} logno={logno:+.2f}: {int(g.sum())}/{len(g)} valid", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nCue grid: {len(df)} nodes -> {OUT}")
for c in ["nii_ha", "oiii_hb", "sii_ha"]:
    print(f"  {c:8s} [{df[c].min():+.2f}, {df[c].max():+.2f}]")
