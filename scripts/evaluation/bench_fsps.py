# bench_fsps.py — time the SPS+Cloudy forward model (python-FSPS with Byler+2017
# Cloudy nebular tables) producing emission-line ratios per galaxy.
# This is the physical pipeline the NF replaces: SPS ionizing spectrum ->
# nebular lines via the Byler Cloudy grid (sp.get_spectrum + sp.emline_luminosity),
# exactly as in scripts/gen_byler_grid.py.
#
# Pinned to ONE core (OMP_NUM_THREADS=1) so the per-model cost is a clean
# per-core-second number; generation across galaxies is embarrassingly parallel.

import os
os.environ["SPS_HOME"] = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/FSPS"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import numpy as np
import fsps


def main():
    print("OMP_NUM_THREADS:", os.environ["OMP_NUM_THREADS"])
    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, add_neb_emission=True,
                                add_neb_continuum=False, dust_type=0)
    wl = np.array(sp.emline_wavelengths)

    def li(t):
        return int(np.argmin(np.abs(wl - t)))

    targets = dict(Hb=4862.7, Hg=4341.7, NII=6585.3, SII16=6718.3, SII31=6732.7,
                   OII26=3727.1, OII29=3730.0, OIII=5008.2, Ha=6564.6)
    I = {k: li(t) for k, t in targets.items()}

    rng = np.random.default_rng(0)

    # ---- one-time init (library load + first spectrum; not counted per galaxy) ----
    sp.params["logzsol"] = -0.3
    sp.params["gas_logz"] = -0.3
    sp.params["gas_logu"] = -2.5
    t0 = time.perf_counter()
    sp.get_spectrum(tage=3e-3)
    _ = np.asarray(sp.emline_luminosity)
    t_init = time.perf_counter() - t0
    print(f"[FSPS] one-time init / first get_spectrum: {t_init:.2f} s")

    # ---- realistic per-galaxy: each galaxy has its own Z (SSP recompute), age, U ----
    N = 40
    t0 = time.perf_counter()
    for _ in range(N):
        z = rng.uniform(-2.0, 0.5)
        u = rng.uniform(-4.0, -1.0)
        age = rng.uniform(0.5, 10.0)
        sp.params["logzsol"] = z
        sp.params["gas_logz"] = z
        sp.params["gas_logu"] = u
        sp.get_spectrum(tage=age * 1e-3)
        L = np.asarray(sp.emline_luminosity, float)
        _ = {k: L[i] for k, i in I.items()}
    t_vary = (time.perf_counter() - t0) / N
    print(f"[FSPS] per-model, vary Z/age/U (SSP recompute each): {t_vary*1000:.1f} ms/model")
    print(f"[FSPS]   -> 1e6 galaxies = {t_vary*1e6:.0f} core-s = {t_vary*1e6/3600:.2f} core-hours")

    # ---- best case (grid style): fixed Z so SSP is reused, vary only age/U ----
    sp.params["logzsol"] = -0.3
    sp.params["gas_logz"] = -0.3
    sp.get_spectrum(tage=3e-3)  # prime SSP at this Z
    M = 200
    t0 = time.perf_counter()
    for _ in range(M):
        u = rng.uniform(-4.0, -1.0)
        age = rng.uniform(0.5, 10.0)
        sp.params["gas_logu"] = u
        sp.get_spectrum(tage=age * 1e-3)
        L = np.asarray(sp.emline_luminosity, float)
    t_fix = (time.perf_counter() - t0) / M
    print(f"[FSPS] per-model, fixed Z / vary age,U (SSP reused): {t_fix*1000:.1f} ms/model")
    print(f"[FSPS]   -> 1e6 galaxies = {t_fix*1e6:.0f} core-s = {t_fix*1e6/3600:.2f} core-hours")


if __name__ == "__main__":
    main()
