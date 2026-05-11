import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

# column names for 8 bright optical lines we're targeting in both desi and sdss
DEFAULT_LINE_ALIASES = {"hbeta":   ["HBETA_FLUX", "H_BETA_FLUX", "HBETA"],
    "hgamma":  ["HGAMMA_FLUX", "H_GAMMA_FLUX", "HGAMMA"],
    "nii6584": ["NII_6584_FLUX", "NII6584_FLUX", "NII_6584", "NII6584"],
    "sii6716": ["SII_6716_FLUX", "SII_6717_FLUX", "SII6716_FLUX", "SII_6716", "SII6716"],
    "sii6731": ["SII_6731_FLUX", "SII6731_FLUX", "SII_6731", "SII6731"],
    "oii3726": ["OII_3726_FLUX", "OII3726_FLUX", "OII_3726", "OII3726"],
    "oii3729": ["OII_3729_FLUX", "OII3729_FLUX", "OII_3729", "OII3729"],
    "oiii5007":["OIII_5007_FLUX", "OIII5007_FLUX", "OIII_5007", "OIII5007"]}

def _resolve_col(df, aliases, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    if required:
        raise KeyError(f"Could not resolve column among aliases: {aliases}")
    return None


def train_line_ratio_flow(df, *,
    # conditioning columns (user supplies these)
    logmstar_col="LOGMSTAR",
    loglha_col="LOG_LHA",

    # optional, only needed if you're training on ratios
    logha_col=None,  # e.g. log10(Halpha_flux) OR log10(Halpha_lum) consistently

    # targets: 8 lines
    line_aliases=DEFAULT_LINE_ALIASES,
    use_ratios_to_ha=True,

    # training hyperparams
    seed=0,
    batch_size=2048,
    epochs=200,
    lr=3e-4,
    clip=1.0):
    """
    Trains a conditional NF for 8 emission lines given (logM*, logL_Hα).

    If use_ratios_to_ha=True:
        targets are log(line) - log(Hα) for each of the 8 lines,
        so you must provide logha_col in df.
        (logha_col can be flux OR luminosity, just be consistent.)

    If use_ratios_to_ha=False:
        targets are absolute log(line) values.

    Returns dict with flow + normalization metadata + resolved column names.
    """
    df = df.copy()

    # Resolve the 8 target columns in df
    line_keys = ["hbeta","hgamma","nii6584","sii6716","sii6731","oii3726","oii3729","oiii5007"]
    target_cols = [_resolve_col(df, line_aliases[k], required=True) for k in line_keys]

    required = [logmstar_col, loglha_col] + target_cols
    if use_ratios_to_ha:
        if logha_col is None:
            raise ValueError("use_ratios_to_ha=True requires logha_col to be provided.")
        required.append(logha_col)

    df = df.dropna(subset=required).reset_index(drop=True)

    # Conditioning
    U = df[[logmstar_col, loglha_col]].values.astype(np.float32)

    # Targets
    if use_ratios_to_ha:
        ratio_cols = []
        for c in target_cols:
            r = c + "_RATIO_TO_HA"
            df[r] = df[c].astype(np.float64) - df[logha_col].astype(np.float64)
            ratio_cols.append(r)
        X = df[ratio_cols].values.astype(np.float32)
        out_cols = ratio_cols
    else:
        X = df[target_cols].values.astype(np.float32)
        out_cols = target_cols

    # Normalize
    X_mean = X.mean(0); X_std = X.std(0); X_std[X_std == 0] = 1.0
    U_mean = U.mean(0); U_std = U.std(0); U_std[U_std == 0] = 1.0
    Xn = (X - X_mean) / X_std
    Un = (U - U_mean) / U_std

    Xn = jnp.asarray(Xn)
    Un = jnp.asarray(Un)

    # Build flow
    key = jr.key(seed)
    flow = block_neural_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(Xn.shape[1])),
        cond_dim=Un.shape[1])

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip),
        optax.adam(lr))
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def loss_fn(flow, x, u):
        return -jnp.mean(flow.log_prob(x, condition=u))

    @eqx.filter_jit
    def update(flow, opt_state, x, u):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow, x, u)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_inexact_array),
            opt_state,
            params=eqx.filter(flow, eqx.is_inexact_array))
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss

    rng_np = np.random.default_rng(seed + 123)

    def iter_batches(X, U, bs):
        n = X.shape[0]
        order = rng_np.permutation(n)
        for i in range(0, n, bs):
            idx = order[i:i+bs]
            yield X[idx], U[idx]

    for ep in range(1, epochs + 1):
        losses = []
        for xb, ub in iter_batches(np.array(Xn), np.array(Un), batch_size):
            flow, opt_state, loss = update(flow, opt_state, jnp.asarray(xb), jnp.asarray(ub))
            losses.append(float(loss))
        print(f"Epoch {ep:3d}  loss={np.mean(losses):.5f}")

    meta = dict(
        seed=seed,
        use_ratios_to_ha=use_ratios_to_ha,
        resolved=dict(
            logmstar_col=logmstar_col,
            loglha_col=loglha_col,
            logha_col=logha_col,
            target_cols=target_cols,
            out_cols=out_cols),
        X_mean=X_mean, X_std=X_std,
        U_mean=U_mean, U_std=U_std)

    return {"flow": flow, "meta": meta, "df_train": df}
