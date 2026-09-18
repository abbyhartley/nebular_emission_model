# bench_paper2.py — paper-grade single-core timing, NF vs SPS+Cloudy for 1e6 galaxies.
# Fixes the FSPS "realistic" measurement: FSPS caches per-metallicity SSPs lazily,
# so the first request at each new Z pays a large one-time build. Over 1e6 galaxies
# those ~dozens of builds amortize to ~0, so the honest per-galaxy cost is the
# STEADY-STATE (cache-warm) cost of get_spectrum with varying Z/age/U.
# We therefore (1) warm the Z cache with a full sweep, then (2) time many varying-Z
# models and report the median (robust) as the steady-state per-galaxy cost.

import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["SPS_HOME"] = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/FSPS"

_allowed = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {_allowed[0]})
print(f"=== allocated cores {_allowed} -> pinned to {sorted(os.sched_getaffinity(0))} ===", flush=True)

import time
import pickle
from pathlib import Path
import numpy as np

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model"
FLOW = Path(f"{REPO}/nf_desi_bgs.eqx")
META = Path(f"{REPO}/nf_desi_bgs_meta.pkl")

N = 1_000_000
BATCH = 200_000


def bench_nf():
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    from flowjax.distributions import Normal
    from flowjax.flows import block_neural_autoregressive_flow

    print("[NF] jax devices:", jax.devices(), flush=True)
    with open(META, "rb") as f:
        meta = pickle.load(f)
    xdim = len(meta["resolved"]["out_cols"])
    key0 = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key0, base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    flow = eqx.tree_deserialise_leaves(FLOW, template)

    def sample_batch(k, U):
        keys = jr.split(k, U.shape[0])
        return jax.vmap(lambda kk, uu: flow.sample(kk, sample_shape=(), condition=uu))(keys, U)

    Un_all = jr.normal(jr.key(1), (N, 2))
    t0 = time.perf_counter()
    _ = sample_batch(jr.key(99), Un_all[:BATCH]).block_until_ready()
    print(f"[NF] one-time compile: {time.perf_counter()-t0:.2f} s", flush=True)

    times = []
    for rep in range(3):
        out = np.empty((N, xdim), np.float32)
        key = jr.key(rep)
        t0 = time.perf_counter()
        for lo in range(0, N, BATCH):
            hi = min(N, lo + BATCH)
            key, sub = jr.split(key)
            xb = sample_batch(sub, Un_all[lo:hi]).block_until_ready()
            out[lo:hi] = np.asarray(xb)
        out = out * meta["X_std"] + meta["X_mean"]
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[NF] rep {rep}: 1e6 in {dt:.2f} s ({N/dt:,.0f} gal/s)", flush=True)
    return float(np.median(times))


def bench_fsps():
    import fsps
    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, add_neb_emission=True,
                                add_neb_continuum=False, dust_type=0)
    wl = np.array(sp.emline_wavelengths)

    def li(t):
        return int(np.argmin(np.abs(wl - t)))
    targets = dict(Hb=4862.7, Hg=4341.7, NII=6585.3, SII16=6718.3, SII31=6732.7,
                   OII26=3727.1, OII29=3730.0, OIII=5008.2, Ha=6564.6)
    I = {k: li(t) for k, t in targets.items()}
    rng = np.random.default_rng(0)

    sp.params["logzsol"] = -0.3; sp.params["gas_logz"] = -0.3; sp.params["gas_logu"] = -2.5
    t0 = time.perf_counter(); sp.get_spectrum(tage=3e-3); _ = np.asarray(sp.emline_luminosity)
    print(f"[FSPS] one-time init/first get_spectrum: {time.perf_counter()-t0:.2f} s", flush=True)

    # (1) COLD: document the lazy per-Z build cost (first calls are big & variable)
    cold = []
    for i in range(30):
        z = rng.uniform(-2.0, 0.5); u = rng.uniform(-4.0, -1.0); age = rng.uniform(0.5, 10.0)
        sp.params["logzsol"] = z; sp.params["gas_logz"] = z; sp.params["gas_logu"] = u
        t0 = time.perf_counter(); sp.get_spectrum(tage=age * 1e-3); _ = np.asarray(sp.emline_luminosity)
        cold.append(time.perf_counter() - t0)
    print(f"[FSPS] cold phase (n=30): median {np.median(cold)*1e3:.1f} ms, "
          f"max {np.max(cold)*1e3:.0f} ms, sum {np.sum(cold):.1f} s", flush=True)

    # (2) WARM the metallicity cache with a full Z sweep so all node-SSPs are built
    for z in np.linspace(-2.0, 0.5, 40):
        sp.params["logzsol"] = float(z); sp.params["gas_logz"] = float(z)
        sp.get_spectrum(tage=3e-3)

    # (3) STEADY STATE: many varying-Z/age/U models with cache warm -> per-galaxy cost
    K = 400
    dts = []
    for i in range(K):
        z = rng.uniform(-2.0, 0.5); u = rng.uniform(-4.0, -1.0); age = rng.uniform(0.5, 10.0)
        sp.params["logzsol"] = z; sp.params["gas_logz"] = z; sp.params["gas_logu"] = u
        t0 = time.perf_counter()
        sp.get_spectrum(tage=age * 1e-3)
        L = np.asarray(sp.emline_luminosity, float)
        _ = {k: L[i2] for k, i2 in I.items()}
        dts.append(time.perf_counter() - t0)
    dts = np.array(dts)
    print(f"[FSPS] STEADY-STATE varying Z/age/U (n={K}): "
          f"median {np.median(dts)*1e3:.1f} ms, mean {np.mean(dts)*1e3:.1f} ms, "
          f"std {np.std(dts)*1e3:.1f} ms, min {np.min(dts)*1e3:.1f} ms", flush=True)

    # best case: fixed Z, vary only age/U (absolute floor)
    sp.params["logzsol"] = -0.3; sp.params["gas_logz"] = -0.3; sp.get_spectrum(tage=3e-3)
    best = []
    for i in range(200):
        u = rng.uniform(-4.0, -1.0); age = rng.uniform(0.5, 10.0)
        sp.params["gas_logu"] = u
        t0 = time.perf_counter(); sp.get_spectrum(tage=age * 1e-3); _ = np.asarray(sp.emline_luminosity)
        best.append(time.perf_counter() - t0)
    print(f"[FSPS] best-case fixed-Z (n=200): median {np.median(best)*1e3:.1f} ms", flush=True)

    return float(np.median(dts)), float(np.median(best))


def main():
    aff = sorted(os.sched_getaffinity(0))
    assert len(aff) == 1, f"expected 1 core, got {aff}"
    t_nf = bench_nf()
    ss_med, best_med = bench_fsps()

    t_fsps_ss = ss_med * N
    t_fsps_best = best_med * N
    nf_per = t_nf / N

    print("\n=== SUMMARY (single CPU core; see HARDWARE line for CPU) ===")
    print(f"NF per galaxy          : {nf_per*1e6:8.2f} us")
    print(f"FSPS per galaxy (steady): {ss_med*1e3:8.2f} ms")
    print(f"FSPS per galaxy (best)  : {best_med*1e3:8.2f} ms")
    print(f"NF          1e6: {t_nf:8.2f} s = {t_nf/3600:.4f} core-h")
    print(f"SPS+Cloudy  1e6 (steady-state, realistic): {t_fsps_ss:10.0f} s = {t_fsps_ss/3600:8.1f} core-h")
    print(f"SPS+Cloudy  1e6 (best-case floor)        : {t_fsps_best:10.0f} s = {t_fsps_best/3600:8.2f} core-h")
    print(f"\nSPEED-UP (steady-state realistic): {t_fsps_ss/t_nf:,.0f} x")
    print(f"SPEED-UP (best-case floor)       : {t_fsps_best/t_nf:,.0f} x")


if __name__ == "__main__":
    main()
