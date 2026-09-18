# bench_paper.py — paper-grade, single-core timing of NF vs SPS+Cloudy for 1e6 galaxies.
# Both methods run pinned to ONE physical CPU core so the reported per-core times
# (and their ratio) are hardware-independent: each method is embarrassingly parallel
# across galaxies, so time = per-core-time * N / N_cores, and the RATIO is invariant.

import os
# Force single-threaded numerics BEFORE importing numpy/jax/fsps.
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["SPS_HOME"] = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/FSPS"

# Pin to a SINGLE core from the job's allocated set, BEFORE importing numpy/jax/fsps
# so every thread they spawn inherits the single-CPU mask. This is the real
# single-core enforcement (taskset -c 0 fails when core 0 isn't in the cpuset).
_allowed = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {_allowed[0]})
print(f"=== allocated cores: {_allowed} -> pinned to {{{_allowed[0]}}} "
      f"(now {sorted(os.sched_getaffinity(0))}) ===", flush=True)

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

    print("[NF] jax devices:", jax.devices())
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

    # one-time compile
    t0 = time.perf_counter()
    _ = sample_batch(jr.key(99), Un_all[:BATCH]).block_until_ready()
    t_compile = time.perf_counter() - t0
    print(f"[NF] one-time compile+first batch: {t_compile:.2f} s")

    times = []
    for rep in range(2):
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
        print(f"[NF] rep {rep}: 1e6 galaxies in {dt:.2f} s ({N/dt:,.0f} gal/s, {1e6*dt/N:.3f} us/gal)")
    return float(np.min(times))  # report best (least contended) single-core time


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

    sp.params["logzsol"] = -0.3
    sp.params["gas_logz"] = -0.3
    sp.params["gas_logu"] = -2.5
    t0 = time.perf_counter()
    sp.get_spectrum(tage=3e-3)
    _ = np.asarray(sp.emline_luminosity)
    print(f"[FSPS] one-time init/first get_spectrum: {time.perf_counter()-t0:.2f} s")

    # realistic: each galaxy its own Z (SSP recompute) + age + U
    N_real = 60
    dts = []
    for _ in range(N_real):
        z = rng.uniform(-2.0, 0.5); u = rng.uniform(-4.0, -1.0); age = rng.uniform(0.5, 10.0)
        sp.params["logzsol"] = z; sp.params["gas_logz"] = z; sp.params["gas_logu"] = u
        t0 = time.perf_counter()
        sp.get_spectrum(tage=age * 1e-3)
        L = np.asarray(sp.emline_luminosity, float)
        _ = {k: L[i] for k, i in I.items()}
        dts.append(time.perf_counter() - t0)
    real_mean, real_std = float(np.mean(dts)), float(np.std(dts))
    print(f"[FSPS] realistic per-model (n={N_real}): {real_mean*1000:.1f} +/- {real_std*1000:.1f} ms")

    # best case: fixed Z (SSP reused), vary age/U
    sp.params["logzsol"] = -0.3; sp.params["gas_logz"] = -0.3
    sp.get_spectrum(tage=3e-3)
    M = 200
    dts = []
    for _ in range(M):
        u = rng.uniform(-4.0, -1.0); age = rng.uniform(0.5, 10.0)
        sp.params["gas_logu"] = u
        t0 = time.perf_counter()
        sp.get_spectrum(tage=age * 1e-3)
        _ = np.asarray(sp.emline_luminosity, float)
        dts.append(time.perf_counter() - t0)
    best_mean, best_std = float(np.mean(dts)), float(np.std(dts))
    print(f"[FSPS] best-case per-model (n={M}): {best_mean*1000:.1f} +/- {best_std*1000:.1f} ms")
    return real_mean, best_mean


def main():
    aff = sorted(os.sched_getaffinity(0))
    assert len(aff) == 1, f"expected 1 core, got {aff}"
    print(f"=== affinity: process bound to {len(aff)} core: {aff} ===")

    t_nf = bench_nf()
    real_mean, best_mean = bench_fsps()

    t_fsps_real = real_mean * N   # core-seconds for 1e6
    t_fsps_best = best_mean * N

    print("\n=== SUMMARY (single CPU core, AMD EPYC 8224P) ===")
    print(f"NF          : 1e6 galaxies in {t_nf:8.2f} s  = {t_nf/3600:.4f} core-h")
    print(f"SPS+Cloudy  (realistic): 1e6 in {t_fsps_real:10.0f} s = {t_fsps_real/3600:8.1f} core-h")
    print(f"SPS+Cloudy  (best case): 1e6 in {t_fsps_best:10.0f} s = {t_fsps_best/3600:8.2f} core-h")
    print(f"\nSPEED-UP (realistic): {t_fsps_real/t_nf:,.0f} x")
    print(f"SPEED-UP (best case): {t_fsps_best/t_nf:,.0f} x")


if __name__ == "__main__":
    main()
