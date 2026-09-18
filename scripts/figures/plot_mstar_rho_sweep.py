"""
M* zero-point sweep figure: Spearman rho (ratio space) vs a constant offset
Delta added to the TEST survey's log M*, for all four train->test combos.
In-survey controls should peak at Delta=0; cross-survey transfer should improve
monotonically away from 0 (the degeneracy with the survey-relative M* zero-point).

Recomputes rho on a Delta grid (the saved CSVs only hold NLL_raw). Subsampled +
modest n_mc for speed; rho is a rank statistic and is stable to this.
"""
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model")
sys.path.insert(0, str(REPO / "scripts"))
from eval_nf_metrics_corrected import (
    prep_eval_dataframe, predict_mean_ratios, true_ratios, metrics, load_flow, FLOWS, FITS,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

DELTAS = np.round(np.arange(-0.60, 0.601, 0.05), 2)
NSUB = 25_000
NMC = 15


def shift_logM(df, meta, delta):
    d = df.copy()
    c = meta["resolved"]["logmstar_col"]
    d[c] = d[c].astype(float) + delta
    return d


def main():
    metas = {k: pickle.load(open(FLOWS[k][1], "rb")) for k in FLOWS}
    flows = {k: load_flow(FLOWS[k][0], metas[k]) for k in FLOWS}
    evalsets = {s: prep_eval_dataframe(FITS[s], survey=s) for s in FITS}

    rows = []
    for train in ("sdss", "desi"):
        for test in ("sdss", "desi"):
            df, res = evalsets[test]
            dfs = df.sample(n=min(NSUB, len(df)), random_state=0).reset_index(drop=True)
            rt = true_ratios(dfs, res).reshape(-1)
            meta, flow = metas[train], flows[train]
            print(f"\n=== {train.upper()} -> {test.upper()} ===", flush=True)
            for dlt in DELTAS:
                rp = predict_mean_ratios(flow, meta, shift_logM(dfs, meta, dlt), seed=7, n_mc=NMC).reshape(-1)
                rho = float(metrics(rp, rt)["spearman_rho"])
                rows.append(dict(train=train, test=test, delta=float(dlt), rho=rho))
                print(f"  delta={dlt:+.2f}  rho={rho:.3f}", flush=True)

    sweep = pd.DataFrame(rows)
    sweep.to_csv(REPO / "docs" / "mstar_zeropoint_rho_sweep.csv", index=False)
    print("Wrote docs/mstar_zeropoint_rho_sweep.csv")

    # ---- plot ----
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.labelsize": 16, "axes.titlesize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})
    C = {"sdss": "#0072B2", "desi": "#E69F00"}   # train-survey color (Okabe-Ito, colorblind-safe)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for train in ("sdss", "desi"):
        for test in ("sdss", "desi"):
            sub = sweep[(sweep.train == train) & (sweep.test == test)].sort_values("delta")
            cross = train != test
            label = f"{train.upper()}$\\rightarrow${test.upper()}" + (" (cross)" if cross else " (in-survey)")
            ax.plot(sub.delta, sub.rho, color=C[train], lw=2.6,
                    ls="--" if cross else "-", label=label,
                    marker="o" if cross else None, ms=4, markevery=2)
    ax.axvline(0.0, color="0.4", ls=":", lw=1.3)
    ax.set_xlabel(r"$\Delta\,\log M_\star$  applied to test survey [dex]")
    ax.set_ylabel(r"Spearman $\rho$  (line ratios)")
    ax.set_xlim(DELTAS[0], DELTAS[-1])
    ax.legend(frameon=True, framealpha=0.9, loc="lower center", ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = REPO / "figs" / f"mstar_zeropoint_rho_sweep.{ext}"
        (REPO / "figs").mkdir(exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("Wrote:", out)


if __name__ == "__main__":
    main()
