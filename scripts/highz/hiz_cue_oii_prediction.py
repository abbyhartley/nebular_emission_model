#!/usr/bin/env python3
"""
Cue prediction (colleague's suggestion): how much should log([OII]3727/Hb) change from z~0.1 to z~1
given the *measured cosmic-average* evolution of electron density n_e(z) and gas-phase metallicity Z(z)?
Compare to our observed ~+0.1 dex [OII] residual (the part not captured by the (M*,L_Ha)-conditioned NF).

Canonical first-pass ISM evolution (swap in colleague's exact refs later):
  n_e: ~30 -> ~100 cm^-3  (log n_H 1.5 -> 2.0)   [Sanders+2016 MOSDEF; Kaasinen+2017]
  Z  : -0.15 dex at fixed mass                    [Sanders+2021; Kashino+2017 FMOS-COSMOS z~1.6]  (bracket -0.1..-0.2)
  U, N/O, C/O, ionizing spectrum held fixed (fiducial) unless noted.
Decomposition: density-only, metallicity-only, both, +U, to see which drives the [OII] change,
and [OIII]/Hb as a control (should stay ~flat, matching why it shows no residual trend).
"""
import jax, jax.numpy as jnp, numpy as np, tengri
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import scienceplots; plt.style.use(["science", "no-latex"])
except Exception: pass
plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 13, "legend.fontsize": 12, "axes.titlesize": 14})

OUTFIG = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/figs_ALTB/hiz_cue_oii_prediction.png"
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
F = tengri.Fixed; KEY = jax.random.PRNGKey(0)


def build(logn, logno=-1.0):
    return tengri.SEDModel.build(ssp,
        sfh={"type": "dpl", "*": tengri.FIXED, "alpha": 1.0, "beta": 2.5, "tau_gyr": 0.05, "log_total_mass": 10.0},
        dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
        neb={"type": "cue", "gas_logno": F(float(logno)), "gas_logn": F(float(logn))}, redshift=F(0.05))


_printed = [False]
def ratios(logn, logU, logZ, logno=-1.0):
    m = build(logn, logno)
    tmpl = dict(m.spec.sample(KEY))
    for k in ["neb_dig_frac", "neb_fesc", "neb_fesc_lya", "neb_fdust"]:
        if k in tmpl: tmpl[k] = jnp.asarray(0.0)
    tmpl["neb_logU"] = jnp.asarray(float(logU)); tmpl["neb_logZ_gas"] = jnp.asarray(float(logZ))
    L = m.predict(tmpl).lines
    if not _printed[0]:
        print("line attrs (OII/OIII/Hbeta):", [a for a in dir(L) if not a.startswith("_") and any(k in a.lower() for k in ["oii", "oiii", "hbeta"])], flush=True)
        _printed[0] = True
    return dict(oii_hb=np.log10(float(L.oii) / float(L.hbeta)),           # Cue gives total [OII]3727
                oiii_hb=np.log10(float(L.oiii_5007) / float(L.hbeta)))


# z~0.1 fiducial local ISM
z0 = ratios(1.5, -2.7, 0.0)
cases = {
    "density only (n_e 30->100)": ratios(2.0, -2.7, 0.0),
    "metallicity only (-0.15)":   ratios(1.5, -2.7, -0.15),
    "n_e + Z (both)":             ratios(2.0, -2.7, -0.15),
    "n_e + Z + U(+0.3)":          ratios(2.0, -2.4, -0.15),
}
brackets = {
    "both, Z=-0.10":   ratios(2.0, -2.7, -0.10),
    "both, Z=-0.20":   ratios(2.0, -2.7, -0.20),
    "both, n_e=200":   ratios(2.3, -2.7, -0.15),
}
print("\nlocal (z~0.1): log[OII]/Hb=%.3f  log[OIII]/Hb=%.3f" % (z0["oii_hb"], z0["oiii_hb"]))
print("\nCASE                          d(log[OII]/Hb)  d(log[OIII]/Hb)")
for name, r in {**cases, **brackets}.items():
    print("  %-28s  %+.3f          %+.3f"
          % (name, r["oii_hb"] - z0["oii_hb"], r["oiii_hb"] - z0["oiii_hb"]), flush=True)

# figure: predicted d(log[OII]/Hb) per driver vs observed residual
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
labels = list(cases.keys()); dO2 = [cases[k]["oii_hb"] - z0["oii_hb"] for k in labels]
dO3 = [cases[k]["oiii_hb"] - z0["oiii_hb"] for k in labels]
x = np.arange(len(labels))
ax[0].bar(x, dO2, color="#CC79A7")
ax[0].axhspan(0.08, 0.12, color="0.6", alpha=0.3, label="observed residual (~0.1 dex)")
ax[0].axhline(0, color="k", lw=0.7)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax[0].set_ylabel(r"predicted $\Delta\log(\mathrm{[OII]}/\mathrm{H}\beta)$, z0.1$\to$1")
ax[0].set_title("[OII]/H$\\beta$: Cue prediction vs observed"); ax[0].legend(frameon=False, fontsize=10)
ax[1].bar(x, dO3, color="#0072B2")
ax[1].axhline(0, color="k", lw=0.7)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax[1].set_ylabel(r"predicted $\Delta\log(\mathrm{[OIII]}/\mathrm{H}\beta)$")
ax[1].set_title("[OIII]/H$\\beta$ (control)")
fig.suptitle(r"Cue: [OII]/H$\beta$ change from measured n$_e$(z) + Z(z) evolution (z$\approx$0.1$\to$1)", y=1.0)
fig.tight_layout(); fig.savefig(OUTFIG, bbox_inches="tight", dpi=150); print("\nSaved:", OUTFIG)
print("=== DONE ===")
