"""
Generate a Byler+2017-style Cloudy nebular grid via python-FSPS: walk
(log Z/Zsun, log U, SSP age) and record the 9 target line luminosities, then
form the diagnostic log ratios. Gas metallicity is tied to stellar (Byler+2017),
n_H fixed at the FSPS/Byler default (100 cm^-3). Saves docs/byler_grid.csv.
"""
import os
os.environ.setdefault("SPS_HOME", "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/FSPS")
import numpy as np, pandas as pd, fsps

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/"
LOGZ = np.round(np.arange(-2.0, 0.51, 0.25), 3)      # gas = stellar
LOGU = np.round(np.arange(-4.0, -0.99, 0.25), 3)
AGES_MYR = np.array([0.5, 1, 2, 3, 4, 6, 8, 10])

sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, add_neb_emission=True,
                            add_neb_continuum=False, dust_type=0)
wl = np.array(sp.emline_wavelengths)
def li(t): return int(np.argmin(np.abs(wl - t)))
I = {k: li(t) for k, t in dict(Hb=4862.7, Hg=4341.7, NII=6585.3, SII16=6718.3,
     SII31=6732.7, OII26=3727.1, OII29=3730.0, OIII=5008.2, Ha=6564.6).items()}

rows = []
for z in LOGZ:
    for u in LOGU:
        sp.params["logzsol"] = z; sp.params["gas_logz"] = z; sp.params["gas_logu"] = u
        for age in AGES_MYR:
            sp.get_spectrum(tage=age * 1e-3)          # Gyr
            L = np.asarray(sp.emline_luminosity, float)
            g = {k: L[i] for k, i in I.items()}
            if min(g.values()) <= 0:
                continue
            rows.append(dict(
                logzsol=z, logu=u, age_myr=age,
                nii_ha=np.log10(g["NII"]/g["Ha"]),
                oiii_hb=np.log10(g["OIII"]/g["Hb"]),
                oii_ha=np.log10((g["OII26"]+g["OII29"])/g["Ha"]),
                sii_ha=np.log10((g["SII16"]+g["SII31"])/g["Ha"]),
            ))
df = pd.DataFrame(rows)
df.to_csv(REPO + "docs/byler_grid.csv", index=False)
print(f"grid nodes: {len(df)}   (Z {LOGZ.min()}..{LOGZ.max()}, logU {LOGU.min()}..{LOGU.max()}, ages {list(AGES_MYR)})")
for c in ["nii_ha", "oiii_hb", "oii_ha", "sii_ha"]:
    print(f"  {c:8s} range [{df[c].min():+.2f}, {df[c].max():+.2f}]")
print("Wrote docs/byler_grid.csv")
