"""
Map the FULL Cue reachable region, now freeing the 7 ionizing-spectrum shape
parameters (stellar -> AGN-hard) in addition to the gas params. Monte-Carlo sample
the 12-D Cue input space over its valid ranges; ionspec_index1 (HeII segment)
spans the stellar default (~22) down to AGN-like (~2). DIG/escape nuisances zeroed.
"""
import jax, jax.numpy as jnp, numpy as np, pandas as pd, tengri
U, F = tengri.Uniform, tengri.Fixed
OUT = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/docs/cue_agn_grid.csv"
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
N = 30000
neb = {"type": "cue",
       "logU": U(-4.0, -0.5), "logZ_gas": U(-1.5, 0.5),
       "gas_logn": U(0.0, 4.0), "gas_logno": U(-1.0, 0.5), "gas_logco": U(-0.5, 0.5),
       "ionspec_index1": U(0.0, 50.0), "ionspec_index2": U(-1.0, 35.0),
       "ionspec_index3": U(-2.0, 20.0), "ionspec_index4": U(-2.0, 10.0),
       "ionspec_logLratio1": U(-1.0, 12.0), "ionspec_logLratio2": U(-1.0, 3.0),
       "ionspec_logLratio3": U(-1.0, 3.0)}
model = tengri.SEDModel.build(
    ssp,
    sfh={"type": "dpl", "*": tengri.FIXED, "alpha": 1.0, "beta": 2.5, "tau_gyr": 0.05, "log_total_mass": 10.0},
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb=neb, redshift=F(0.05))
NUIS = ["neb_dig_frac", "neb_fesc", "neb_fesc_lya", "neb_fdust"]

def pred_one(p):
    L = model.predict_emission_lines(p)
    return jnp.array([L.halpha, L.hbeta, L.nii_6584, L.oiii_5007, L.sii_6717, L.sii_6731])

keys = jax.random.split(jax.random.PRNGKey(0), N)
outs = []
for c in range(0, N, 1000):
    pb = dict(jax.vmap(model.spec.sample)(keys[c:c+1000]))
    for k in NUIS:
        if k in pb:
            pb[k] = jnp.zeros_like(pb[k])
    outs.append(np.asarray(jax.vmap(pred_one)(pb)))
    print(f"  chunk {c//1000+1}/30 done", flush=True)
out = np.concatenate(outs)
ha, hb, nii, oiii, s16, s31 = out.T
g = (ha > 0) & (hb > 0) & (nii > 0) & (oiii > 0) & (s16 > 0) & (s31 > 0)
df = pd.DataFrame(dict(nii_ha=np.log10(nii[g]/ha[g]), oiii_hb=np.log10(oiii[g]/hb[g]),
                       sii_ha=np.log10((s16[g]+s31[g])/ha[g])))
df.to_csv(OUT, index=False)
print(f"\nCue (ion-flex) grid: {len(df)}/{N} valid -> {OUT}")
for c in ["nii_ha", "oiii_hb", "sii_ha"]:
    print(f"  {c:8s} [{df[c].min():+.2f}, {df[c].max():+.2f}]")
