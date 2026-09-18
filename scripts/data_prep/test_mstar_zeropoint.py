"""
Review 2.6 -- M* zero-point sensitivity of the cross-survey shift.

The cross-survey shift is claimed "at fixed M*", but M* is color-derived and carries a
possible survey-dependent zero-point offset. We add a constant delta to the TEST survey's
log M* before conditioning, and re-measure transfer. If a plausible delta (~0.1-0.15 dex)
substantially reduces the cross-survey degradation, part of the "shift" is a mass-calibration
systematic rather than intrinsic astrophysics.

Cheap NLL_raw is swept over a delta grid for all four combos (in-survey = control, should
optimize near 0). RMSE/rho (MC) are computed only at delta=0 and the best delta for the two
cross-survey combos.

Output: docs/../mstar_zeropoint_sweep.csv
"""
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model")
sys.path.insert(0, str(REPO / "scripts"))
from eval_nf_metrics_corrected import (
    prep_eval_dataframe, predict_mean_ratios, nll_bits, true_ratios, metrics, load_flow, FLOWS, FITS,
)

DELTAS = np.round(np.arange(-0.30, 0.301, 0.05), 2)


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
            meta, flow = metas[train], flows[train]
            print(f"\n=== {train.upper()} -> {test.upper()}  NLL_raw sweep ===", flush=True)
            for dlt in DELTAS:
                _, nll_raw = nll_bits(flow, meta, shift_logM(df, meta, dlt), res)
                rows.append(dict(train=train, test=test, delta=float(dlt), nll_raw=float(nll_raw)))
                print(f"  delta={dlt:+.2f}  NLL_raw={nll_raw:.4f}")

    sweep = pd.DataFrame(rows)

    # best delta per combo + MC point metrics at delta=0 and best (cross-survey only)
    print("\n=== best-fit delta (argmin NLL_raw) ===")
    summary = []
    for train in ("sdss", "desi"):
        for test in ("sdss", "desi"):
            sub = sweep[(sweep.train == train) & (sweep.test == test)].reset_index(drop=True)
            j = int(sub.nll_raw.values.argmin())
            best = float(sub.delta.values[j])
            nll0 = float(sub.nll_raw.values[np.argmin(np.abs(sub.delta.values))])
            print(f"  {train}->{test}: best delta = {best:+.2f} dex  "
                  f"(NLL_raw {nll0:.3f} at 0 -> {sub.nll_raw.values[j]:.3f} at best)")
            rec = dict(train=train, test=test, best_delta=best,
                       nll_raw_at0=nll0, nll_raw_best=float(sub.nll_raw.values[j]))
            if train != test:  # cross-survey: also MC RMSE/rho at 0 and best
                df, res = evalsets[test]; meta, flow = metas[train], flows[train]
                rt = true_ratios(df, res).reshape(-1)
                for tag, dlt in [("0", 0.0), ("best", best)]:
                    rp = predict_mean_ratios(flow, meta, shift_logM(df, meta, dlt), seed=7).reshape(-1)
                    m = metrics(rp, rt)
                    rec[f"rmse_{tag}"] = m["rmse"]; rec[f"rho_{tag}"] = m["spearman_rho"]
                print(f"      RMSE {rec['rmse_0']:.3f}->{rec['rmse_best']:.3f}  "
                      f"rho {rec['rho_0']:.3f}->{rec['rho_best']:.3f}")
            summary.append(rec)

    sweep.to_csv(REPO / "docs" / "mstar_zeropoint_sweep.csv", index=False)
    pd.DataFrame(summary).to_csv(REPO / "docs" / "mstar_zeropoint_summary.csv", index=False)
    print("\nWrote docs/mstar_zeropoint_sweep.csv and docs/mstar_zeropoint_summary.csv")


if __name__ == "__main__":
    main()
