# bench_nf.py — time the normalizing flow generating 8 line ratios for 1e6 galaxies.
# Mirrors the exact sampling path used in scripts/figures (vmap over flow.sample),
# i.e. the model's real production pipeline. CPU only (jax sees CpuDevice here).

import os, time, pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model"
FLOW = Path(f"{REPO}/nf_desi_bgs.eqx")
META = Path(f"{REPO}/nf_desi_bgs_meta.pkl")

N = 1_000_000
BATCH = 200_000   # same batch size as production sample_ratios()


def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    key = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key, base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    return eqx.tree_deserialise_leaves(flow_path, template)


def sample_batch(flow, k, U):
    keys = jr.split(k, U.shape[0])
    return jax.vmap(lambda kk, uu: flow.sample(kk, sample_shape=(), condition=uu))(keys, U)


def main():
    print("jax devices:", jax.devices())
    print("XLA intra-op threads (cpu count):", os.cpu_count())
    with open(META, "rb") as f:
        meta = pickle.load(f)
    flow = load_flow(FLOW, meta)
    xdim = len(meta["resolved"]["out_cols"])
    print("xdim (line ratios per galaxy):", xdim)

    # Conditioning vectors in normalized space; timing is independent of the values.
    Un_all = jr.normal(jr.key(1), (N, 2))

    # Warm-up: compile the sampling program on one batch (one-time cost).
    t0 = time.perf_counter()
    _ = sample_batch(flow, jr.key(99), Un_all[:BATCH]).block_until_ready()
    t_compile = time.perf_counter() - t0
    print(f"[one-time] compile + first {BATCH:,} batch: {t_compile:.2f} s")

    # Timed steady-state generation of the full 1e6 (single draw per galaxy).
    out = np.empty((N, xdim), np.float32)
    key = jr.key(0)
    t0 = time.perf_counter()
    for lo in range(0, N, BATCH):
        hi = min(N, lo + BATCH)
        key, sub = jr.split(key)
        xb = sample_batch(flow, sub, Un_all[lo:hi]).block_until_ready()
        out[lo:hi] = np.asarray(xb)
    # de-normalize to physical log-ratios (part of the real pipeline; cheap)
    out = out * meta["X_std"] + meta["X_mean"]
    t_run = time.perf_counter() - t0

    print(f"[NF] {N:,} galaxies x {xdim} ratios (1 draw each) in {t_run:.2f} s")
    print(f"[NF] throughput: {N/t_run:,.0f} galaxies/s   ({1e6*t_run/N:.3f} us/galaxy)")
    print(f"[NF] wall time for 1e6 galaxies (excl. one-time compile): {t_run:.2f} s")
    print(f"[NF] core-seconds for 1e6 (wall x {os.cpu_count()} cores): {t_run*os.cpu_count():.1f} core-s")


if __name__ == "__main__":
    main()
