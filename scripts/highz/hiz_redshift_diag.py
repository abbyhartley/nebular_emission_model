#!/usr/bin/env python3
"""
Does the (M*,L_Ha) box bias the DESI-COSMOS test toward low redshift, and is that why it looks good?
Uses cached Tier-1 arrays (no flow). Reports:
  - median z of training, Tier-1 all (Ha S/N>3), Tier-1 in-box
  - z-distribution of in-box vs out-of-box (is the box removing high-z galaxies?)
  - per-z-bin metrics WITHIN Tier-1 in-box  -> does accuracy degrade toward z~0.49?
"""
import numpy as np
from scipy.stats import spearmanr

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/"
A = np.load(REPO + "figs_ALTB/hiz_arrays.npz", allow_pickle=True)
z = A["t1_z"]; inbox = A["t1_inbox"]; obs = A["t1_obs"]; sn = A["t1_sn"]; pm = A["t1_predmean"]
NAMES = [str(n) for n in A["t1_names"]]

print("Training DESI-BGS median z ~ 0.15 (reference)")
print("Tier-1 ALL (Ha S/N>3):   N=%5d  z p16/50/84 = %.3f/%.3f/%.3f" %
      (len(z), *np.percentile(z, [16, 50, 84])))
print("Tier-1 IN-BOX:           N=%5d  z p16/50/84 = %.3f/%.3f/%.3f" %
      (inbox.sum(), *np.percentile(z[inbox], [16, 50, 84])))
print("Tier-1 OUT-of-box:       N=%5d  z p16/50/84 = %.3f/%.3f/%.3f" %
      ((~inbox).sum(), *np.percentile(z[~inbox], [16, 50, 84])))
print("fraction in-box vs z:")
for a, b in [(0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.49)]:
    s = (z >= a) & (z < b)
    print("   z[%.2f,%.2f]: N=%5d  in-box frac=%.2f" % (a, b, s.sum(), inbox[s].mean()))

print("\nPer-z-bin metrics WITHIN in-box (all-4-lines>3), does accuracy degrade with z?")
all4 = np.all(sn > 3, axis=1)
for a, b in [(0.05, 0.20), (0.20, 0.30), (0.30, 0.39), (0.39, 0.49)]:
    s = inbox & all4 & (z >= a) & (z < b)
    if s.sum() < 30:
        print("   z[%.2f,%.2f]: N=%d (too few)" % (a, b, s.sum())); continue
    # pooled over 4 lines
    r = (pm[s] - obs[s]).ravel(); g = np.isfinite(r); rr = r[g]
    rmse = np.sqrt(np.mean(rr**2)); bias = np.median(rr)
    rho = spearmanr(pm[s].ravel()[g], obs[s].ravel()[g]).correlation
    # [OIII] alone
    c = NAMES.index("OIII5007")
    ro = pm[s, c] - obs[s, c]; go = np.isfinite(ro)
    rmse_o = np.sqrt(np.mean(ro[go]**2)); bias_o = np.median(ro[go])
    rho_o = spearmanr(pm[s, c][go], obs[s, c][go]).correlation
    print("   z[%.2f,%.2f] N=%4d | pooled bias=%+.3f rmse=%.3f rho=%.2f | [OIII] bias=%+.3f rmse=%.3f rho=%.2f"
          % (a, b, s.sum(), bias, rmse, rho, bias_o, rmse_o, rho_o))
print("=== DONE ===")
