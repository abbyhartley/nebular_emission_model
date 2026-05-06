from __future__ import annotations
import numpy as np


# --- Unified mag / K-corr convention ---
# We enforce: M = m - DM - K
def absmag(m_app, distmod, kcorr):
    return np.asarray(m_app) - np.asarray(distmod) - np.asarray(kcorr)


def restframe_color_gmr(m_g, m_r, distmod, k_g, k_r):
    """
    Rest-frame (g-r) using the *same* convention for both surveys:
      M = m - DM - K
      (g-r)_rest = M_g - M_r
    """
    M_g = absmag(m_g, distmod, k_g)
    M_r = absmag(m_r, distmod, k_r)
    return M_g - M_r


def desi_to_sdss_gmr(gmr_desi):
    """
    Empirically derived mapping from cross-match:
      (g-r)_SDSS = 0.989*(g-r)_DESI - 0.104
    """
    gmr_desi = np.asarray(gmr_desi)
    return 0.989 * gmr_desi - 0.104


def log10_ml_r_from_gmr_sdss(gmr_sdss):
    """
    Best-fit calibrated relation:
      log10(M/L_r) = 1.062*(g-r)_SDSS - 0.555
    """
    gmr_sdss = np.asarray(gmr_sdss)
    return 1.062 * gmr_sdss - 0.555


def log10_stellar_mass_color(*,
    survey: str,
    m_g, m_r,
    distmod,
    k_g, k_r,
    Msun_r: float):
    """
    Color-based stellar mass using a single convention across surveys.

    Parameters
    ----------
    survey : {"sdss", "desi"}
        - "sdss": use rest-frame (g-r) directly
        - "desi": compute rest-frame (g-r) then map to SDSS-like color using your linear transform
    m_g, m_r : array-like
        Observed apparent magnitudes in g and r (AB).
    distmod : array-like
        Distance modulus DM.
    k_g, k_r : array-like
        K-corrections into *rest-frame g* and *rest-frame r* using the same sign convention:
        M = m - DM - K
    Msun_r : float
        Solar absolute magnitude in SDSS r band (AB) used to compute L_r.

    Returns
    -------
    log10Mstar : array-like
        log10(M*/Msun)
    """
    survey = survey.lower().strip()
    if survey not in {"sdss", "desi"}:
        raise ValueError(f"survey must be 'sdss' or 'desi', got {survey!r}")

    # Rest-frame color and rest-frame r-band absolute magnitude
    gmr_rest = restframe_color_gmr(m_g, m_r, distmod, k_g, k_r)
    M_r = absmag(m_r, distmod, k_r)

    # Enforce shared SDSS-like color for M/L relation
    if survey == "desi":
        gmr_for_ml = desi_to_sdss_gmr(gmr_rest)
    else:
        gmr_for_ml = gmr_rest

    log10_ML = log10_ml_r_from_gmr_sdss(gmr_for_ml)
    log10_L = -0.4 * (np.asarray(M_r) - Msun_r)

    return log10_ML + log10_L
