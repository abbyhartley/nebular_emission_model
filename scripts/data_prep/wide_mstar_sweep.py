"""Wider M* zero-point NLL sweep for the two cross-survey combos (find true optimum)."""
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model")
sys.path.insert(0, str(REPO / "scripts"))
from eval_nf_metrics_corrected import (prep_eval_dataframe, nll_bits, predict_mean_ratios,
                                       true_ratios, metrics, load_flow, FLOWS, FITS)

DELTAS = np.round(np.arange(-0.60, 0.601, 0.05), 2)


def shift(df, meta, d):
    x = df.copy(); c = meta["resolved"]["logmstar_col"]; x[c] = x[c].astype(float) + d; return x


def main():
    metas = {k: pickle.load(open(FLOWS[k][1], "rb")) for k in FLOWS}
    flows = {k: load_flow(FLOWS[k][0], metas[k]) for k in FLOWS}
    rows = []
    for train, test in [("sdss", "desi"), ("desi", "sdss")]:
        df, res = prep_eval_dataframe(FITS[test], survey=test)
        meta, flow = metas[train], flows[train]
        print(f"\n=== {train.upper()} -> {test.upper()} ===", flush=True)
        best = (None, 1e9)
        for d in DELTAS:
            _, nr = nll_bits(flow, meta, shift(df, meta, d), res)
            rows.append(dict(train=train, test=test, delta=float(d), nll_raw=float(nr)))
            if nr < best[1]:
                best = (float(d), float(nr))
            print(f"  delta={d:+.2f}  NLL_raw={nr:.4f}")
        # MC point metrics at delta=0 and the best delta
        rt = true_ratios(df, res).reshape(-1)
        m0 = metrics(predict_mean_ratios(flow, meta, shift(df, meta, 0.0), seed=7).reshape(-1), rt)
        mb = metrics(predict_mean_ratios(flow, meta, shift(df, meta, best[0]), seed=7).reshape(-1), rt)
        print(f"  BEST delta={best[0]:+.2f}  NLL_raw {best[1]:.3f}")
        print(f"    RMSE {m0['rmse']:.3f}->{mb['rmse']:.3f}   rho {m0['spearman_rho']:.3f}->{mb['spearman_rho']:.3f}")
    pd.DataFrame(rows).to_csv(REPO / "docs" / "mstar_zeropoint_wide_sweep.csv", index=False)
    print("\nWrote docs/mstar_zeropoint_wide_sweep.csv")


if __name__ == "__main__":
    main()
