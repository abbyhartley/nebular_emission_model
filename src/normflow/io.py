"""Load the trained conditional normalizing flows and draw line-ratio samples.

The flows predict the joint distribution of eight optical emission line ratios,
log10(L_line / L_Halpha), conditioned on (log M*, log L_Halpha).

    from normflow.io import load_flow, sample_line_ratios, line_names
    flow, meta = load_flow("desi")
    x = sample_line_ratios(flow, meta, logmstar=[10.0], loglha=[41.0])
"""
from __future__ import annotations

import os
import pickle
from functools import partial
from pathlib import Path

import numpy as np

SURVEYS = {
    "sdss": ("nf_sdss_ALTB.eqx", "nf_sdss_ALTB_meta.pkl"),
    "desi": ("nf_desi_ALTB.eqx", "nf_desi_ALTB_meta.pkl"),
}

_PKG_ROOT_MODELS = Path(__file__).resolve().parents[2] / "models"


def default_models_dir() -> Path:
    """models/ next to the repo, or $NEBULAR_MODELS_DIR if the package is installed elsewhere."""
    env = os.environ.get("NEBULAR_MODELS_DIR")
    return Path(env) if env else _PKG_ROOT_MODELS


def _template(xdim: int, cond_dim: int, seed: int):
    import jax.numpy as jnp
    import jax.random as jr
    from flowjax.distributions import Normal
    from flowjax.flows import block_neural_autoregressive_flow
    from flowjax.root_finding import bisect_check_expand_search, root_finder_to_inverter

    # Wider bracket than the flowjax default; the learned densities have heavy
    # tails in the faint doublet components and the default search can fail.
    inverter = root_finder_to_inverter(
        partial(bisect_check_expand_search, midpoint=jnp.zeros(xdim), width=5,
                max_steps=1000, throw=False, max_width=200)
    )
    return block_neural_autoregressive_flow(
        key=jr.key(int(seed)), base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=cond_dim, inverter=inverter,
    )


def load_flow(survey: str, models_dir=None):
    """Return ``(flow, meta)`` for ``survey`` in {"sdss", "desi"}."""
    import equinox as eqx

    key = survey.lower()
    if key not in SURVEYS:
        raise ValueError(f"survey must be one of {sorted(SURVEYS)}, got {survey!r}")
    d = Path(models_dir) if models_dir is not None else default_models_dir()
    eqx_path, meta_path = (d / n for n in SURVEYS[key])
    for p in (eqx_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Set NEBULAR_MODELS_DIR to the directory holding the "
                f".eqx/.pkl pairs, or pass models_dir=."
            )
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)
    xdim = len(meta["resolved"]["out_cols"])
    cond_dim = int(np.atleast_1d(meta["U_mean"]).size)
    return eqx.tree_deserialise_leaves(eqx_path, _template(xdim, cond_dim, meta.get("seed", 0))), meta


def line_names(meta) -> list[str]:
    """Target line order, e.g. ['H_BETA', 'H_GAMMA', 'NII_6584', ...]."""
    out = []
    for c in meta["resolved"]["target_cols"]:
        c = c[len("LOG10_"):] if c.startswith("LOG10_") else c
        out.append(c[: -len("_FLUX")] if c.endswith("_FLUX") else c)
    return out


def sample_line_ratios(flow, meta, logmstar, loglha, seed: int = 0):
    """One sample per galaxy.

    Parameters
    ----------
    logmstar, loglha : array_like
        log10(M*/Msun) and log10(L_Halpha / erg s^-1), same length.

    Returns
    -------
    ndarray, shape (N, 8)
        log10(L_line / L_Halpha), in the order given by :func:`line_names`.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    u = np.column_stack([np.asarray(logmstar, float), np.asarray(loglha, float)])
    un = ((u - meta["U_mean"]) / meta["U_std"]).astype(np.float32)
    keys = jr.split(jr.key(int(seed)), len(un))
    x = jax.vmap(lambda k, c: flow.sample(k, sample_shape=(), condition=c))(keys, jnp.asarray(un))
    return np.asarray(x) * meta["X_std"] + meta["X_mean"]


def log_prob(flow, meta, x, logmstar, loglha):
    """Conditional log-density of observed log line ratios ``x`` (N, 8)."""
    import jax
    import jax.numpy as jnp

    u = np.column_stack([np.asarray(logmstar, float), np.asarray(loglha, float)])
    un = ((u - meta["U_mean"]) / meta["U_std"]).astype(np.float32)
    xn = ((np.asarray(x, float) - meta["X_mean"]) / meta["X_std"]).astype(np.float32)
    lp = jax.vmap(lambda xi, ui: flow.log_prob(xi, condition=ui))(jnp.asarray(xn), jnp.asarray(un))
    return np.asarray(lp) - np.sum(np.log(meta["X_std"]))
