# generate_data.py — Compton Camera Dataset Generator
# DFG Project — University of Siegen
#V3-CPU for 1 source per file. 
# PURPOSE:
#   Simulates realistic Compton scattering events between a scatter detector
#   (front plane) and an absorption detector (back plane). Outputs structured
#   CSV files for machine learning training and testing.
#
# PHYSICS BACKGROUND:
#   In a Compton camera, an incoming gamma ray from a radioactive source
#   undergoes Compton scattering in the front detector. The recoil electron
#   deposits energy there, and the scattered gamma continues to the back
#   detector where it is fully absorbed. By measuring the positions and
#   energies of both interactions, the original gamma direction can be
#   constrained to a cone surface (the Compton cone).
#
#   Compton scattering formula:
#       cos(theta) = 1 - m_e*c^2 * (1/E2 - 1/E1)
#   Where:
#       theta = scatter angle of the photon
#       E1    = incident photon energy (MeV)
#       E2    = scattered photon energy (MeV)
#       m_e*c^2 = electron rest mass energy = 0.511 MeV
#v2
# Physics improvements over previous version:
#   1. Klein-Nishina weighted scatter angle sampling (not uniform)- it is more likely for 1 MeV gammas to scatter at small angles, and this is now reflected in the dataset. This replaces the previous approach of deriving theta from random positions, which did not produce a physically correct distribution of scatter angles.
#   2. Exponential attenuation depth-of-interaction for scatter z
#   3. z smearing added for both scatter and absorb detectors
#   5. Derived cone geometry features exported to CSV
#cone features() computes the cone axis, opening angle, scatter-absorb distance, and scatter depth ratio based on the measured scatter and absorb positions and the measured scatter angle. These features are added to the CSV output for potential use in machine learning models.
# Usage:
#   python generate_data.py
#   python generate_data.py --train-files 2000 --test-files 200 --events 1500

import os
import argparse
import numpy as np
import pandas as pd
import config

# # ── Dataset defaults ──────────────────────────────────────────────────────────
# DEFAULT_TRAIN_FILES = 2000
# DEFAULT_TEST_FILES  = 50
# DEFAULT_EVENTS      = 1000
# DEFAULT_SEED        = 0

# # ── Event type fractions — must sum to 1.0 ───────────────────────────────────
# FRAC_TYPE1 = 0.65
# FRAC_TYPE4 = 0.15
# FRAC_TYPE2 = 0.10   # scatter singles 
# FRAC_TYPE3 = 0.10   # absorb  singles — 

# # ── Detector resolution ───────────────────────────────────────────────────────
# POS_XY_SIGMA_MM   = 1.5    # lateral (x,y) position resolution, mm
# POS_Z_SIGMA_MM    = 2.0    # depth (z) resolution — finite DOI reconstruction
# ENERGY_RES_FRAC   = 0.05   # energy resolution sigma/mean
# ANGLE_SIGMA_DEG   = 2.0    # additional electronic angular smearing

# # ── Attenuation coefficients (for exponential depth-of-interaction) ───────────
# # Linear attenuation coefficient mu for 1 MeV gammas:
# #   PMMA (scatter): ~0.070 cm^-1  = 0.0070 mm^-1
# #   LYSO (absorb):  ~0.870 cm^-1  = 0.0870 mm^-1
# MU_PMMA_PER_MM = 0.0070
# MU_LYSO_PER_MM = 0.0870

# ── CSV column order ─────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "event_id", "event_type", "source_id",
    "source_x",  "source_y",  "source_z",
    "scatter_x", "scatter_y", "scatter_z",
    "absorb_x",  "absorb_y",  "absorb_z",
    "scatter_angle", "electron_energy",
    # Derived cone geometry features
    "cone_axis_x", "cone_axis_y", "cone_axis_z",
    "cone_opening",
    "scat_absorb_dist",
    "scat_depth_ratio",
]

# =============================================================================
# PHYSICS HELPERS
# =============================================================================

def scattered_energy(E1, theta_rad):
    """
    Computes scattered photon energy E2 from the Compton formula.

    Physics:
        After scattering through angle theta, the photon energy decreases.
        A larger scatter angle means more energy transferred to the electron.
        This formula comes from relativistic energy-momentum conservation:

            E2 = E1 / (1 + (E1 / m_e*c^2) * (1 - cos(theta)))

    Parameters
    ----------
    E1        : float — incident photon energy in MeV
    theta_rad : float — scatter angle in radians

    Returns
    -------
    float — scattered photon energy E2 in MeV
    """
    return E1 / (1.0 + (E1 / config.ELECTRON_REST_MASS_MEV) * (1.0 - np.cos(theta_rad)))

def electron_energy(E1, E2):
    """
    Computes the recoil electron kinetic energy from energy conservation.

    Physics:
        The electron receives whatever energy the photon loses:
            T_e = E1 - E2

        This electron is what produces Cherenkov radiation in the PMMA
        radiator. Its energy determines whether the Cherenkov threshold
        (0.18 MeV for PMMA, n=1.49) is exceeded.

    Parameters
    ----------
    E1 : float — incident photon energy in MeV
    E2 : float — scattered photon energy in MeV

    Returns
    -------
    float — electron kinetic energy in MeV (always >= 0)
    """
    return max(0.0, E1 - E2)

def geometry_angle(src, scat, absorb):
    """
    Computes the angle between the incoming gamma (from source to scatter) and
    the outgoing gamma (from scatter to absorb) using the measured positions.
        This is the "measured" scatter angle based on geometry, which will be
        smeared to simulate electronic noise. It may differ from the true KN angle
        due to position smearing and non-collinearity.

    Physics:
        The scatter angle theta is the angle at the scatter vertex between:
            - The INCOMING gamma direction: source  --> scatter
            - The OUTGOING gamma direction: scatter --> absorb

        Using the dot-product definition of angle:
            cos(theta) = (v_in . v_out) / (|v_in| * |v_out|)

    Parameters
    ----------
    source_pos  : np.ndarray (3,) — true source position [x, y, z]
    scatter_pos : np.ndarray (3,) — Compton scatter point [x, y, z]
    absorb_pos  : np.ndarray (3,) — absorption point [x, y, z]

    Returns
    -------
    float — scatter angle in radians, range [0, pi]
    """
    v_in  = scat   - src
    v_out = absorb - scat
    n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
    if n_in < 1e-9 or n_out < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v_in / n_in, v_out / n_out), -1.0, 1.0)))

def klein_nishina_pdf(theta_rad, E1):
    """
    Computes the Klein-Nishina probability density function for a given scatter angle.
    Physics:
        The Klein-Nishina formula gives the differential cross-section for Compton
        scattering as a function of scatter angle and incident photon energy.
        It accounts for relativistic quantum effects and is not uniform in theta.
        The formula (up to normalization) is:
            dσ/dΩ ∝ (E2/E1)^2 * (E2/E1 + E1/E2 - sin^2(theta)) * sin(theta)
        where E2 is the scattered photon energy computed from the Compton formula.  
    Parameters
    ----------
    theta_rad : float — scatter angle in radians
    E1        : float — incident photon energy in MeV
    Returns
    -------
    float — unnormalized probability density for scattering at angle theta
    """ 
    # Used to sample physically correct scatter angle distributions.
    E2    = scattered_energy(E1, theta_rad)
    ratio = E2 / E1
    return ratio**2 * (ratio + 1.0 / ratio - np.sin(theta_rad)**2) * np.sin(theta_rad)

def sample_kn_theta(E1, n_samples=1):
    # Rejection sampling of theta from Klein-Nishina distribution.
    # Proposal: uniform in [0, pi]; accept with probability pdf / pdf_max.
    theta_grid = np.linspace(0.001, np.pi - 0.001, 1000)
    pdf_vals   = np.array([klein_nishina_pdf(t, E1) for t in theta_grid])
    pdf_max    = pdf_vals.max()

    results = []
    while len(results) < n_samples:
        t = np.random.uniform(0, np.pi)
        if np.random.uniform(0, pdf_max) < klein_nishina_pdf(t, E1):
            results.append(t)
    return results[0] if n_samples == 1 else np.array(results)

def sample_interaction_depth(thickness_mm, mu_per_mm):
    # Exponential attenuation: interaction probability ∝ exp(-mu * depth).
    # Sample depth via inverse CDF: d = -ln(U) / mu, clipped to [0, thickness].
    u = np.random.uniform(1e-9, 1.0)
    d = -np.log(u) / mu_per_mm
    return float(np.clip(d, 0.0, thickness_mm))

def smear_pos(pos, z_sigma=config.POS_Z_SIGMA_MM):
    '''Smears a 3D position with Gaussian noise. x and y are smeared with POS_XY_SIGMA_MM,
    while z is smeared with POS_Z_SIGMA_MM to reflect finite DOI resolution.'''
    out    = pos.copy()
    out[0] += np.random.normal(0, config.POS_XY_SIGMA_MM)
    out[1] += np.random.normal(0, config.POS_XY_SIGMA_MM)
    out[2] += np.random.normal(0, z_sigma)          # z is now also smeared
    return out

def smear_energy_val(e):
    #Smears an energy value with Gaussian noise proportional to the energy (energy resolution).
    return float(max(0.0, np.random.normal(e, config.ENERGY_RES_FRAC * e)))

def cone_features(scat, absorb, theta_rad):
    # Cone axis: unit vector from absorb to scatter (direction of incoming gamma).
    axis_raw  = scat - absorb
    axis_norm = np.linalg.norm(axis_raw)
    cone_axis = axis_raw / axis_norm if axis_norm > 1e-9 else np.array([0, 0, 1])

    dist  = float(axis_norm)
    cos_t = float(np.cos(theta_rad))

    # Depth ratio: how deep in the PMMA block the scatter occurred.
    # 0 = front face (SCATTER_PLANE_Z), 1 = back face (SCATTER_PLANE_Z - thickness)
    depth_ratio = float(np.clip(
        (config.SCATTER_PLANE_Z - scat[2]) / config.SCATTER_THICKNESS_MM, 0.0, 1.0
    ))

    return {
        "cone_axis_x":      round(float(cone_axis[0]), 6),
        "cone_axis_y":      round(float(cone_axis[1]), 6),
        "cone_axis_z":      round(float(cone_axis[2]), 6),
        "cone_opening":     round(cos_t,        6),
        "scat_absorb_dist": round(dist,          3),
        "scat_depth_ratio": round(depth_ratio,   6),
    }

def rand_scatter():
    # Physically driven: depth sampled from exponential attenuation (Beer-Lambert),
    # not uniform. Shallower interactions are more probable for 1 MeV gammas.
    depth = sample_interaction_depth(config.SCATTER_THICKNESS_MM, config.MU_PMMA_PER_MM)
    return np.array([
        np.random.uniform(*config.SCATTER_X_RANGE),
        np.random.uniform(*config.SCATTER_Y_RANGE),
        config.SCATTER_PLANE_Z - depth,
    ])

def rand_absorb():
    # Same physics: LYSO has much higher mu so absorptions are shallower.
    depth = sample_interaction_depth(config.ABSORB_THICKNESS_MM, config.MU_LYSO_PER_MM)
    return np.array([
        np.random.uniform(*config.ABSORB_X_RANGE),
        np.random.uniform(*config.ABSORB_Y_RANGE),
        config.ABSORB_PLANE_Z - depth,
    ])

# =============================================================================
# SOURCE GENERATION
# =============================================================================

def make_sources(n):
    return [
        {
            "id":  i,
            "pos": np.array([
                float(np.random.uniform(*config.SOURCE_X_RANGE)),
                float(np.random.uniform(*config.SOURCE_Y_RANGE)),
                float(np.random.uniform(*config.SOURCE_Z_RANGE)),
            ])
        }
        for i in range(n)
    ]

# =============================================================================
# EVENT TYPE 1 — True Compton coincidence
# =============================================================================

def make_type1(src, eid):
    # Step 1: Sample physically correct scatter angle from Klein-Nishina distribution.
    # This replaces the previous approach (deriving theta from random positions)
    # with proper physics — the angle is sampled FIRST, then the interaction
    # positions are constrained to be consistent with it.
    theta_true = sample_kn_theta(config.INCIDENT_ENERGY_MEV)

    E2_true = scattered_energy(config.INCIDENT_ENERGY_MEV, theta_true)
    Te_true = electron_energy(config.INCIDENT_ENERGY_MEV, E2_true)

    if Te_true < config.CHERENKOV_THRESHOLD_MEV:
        return None

    # Step 2: Sample interaction positions from physically motivated distributions.
    scat_true   = rand_scatter()   # exponential depth-of-interaction in PMMA
    absorb_true = rand_absorb()    # exponential depth-of-interaction in LYSO

    # Step 3: Smear all three coordinates (x, y, z) to simulate detector resolution.
    # z is smeared for both detectors — DOI reconstruction has finite resolution.
    scat_m   = smear_pos(scat_true)
    absorb_m = smear_pos(absorb_true)
    Te_m     = smear_energy_val(Te_true)

    # Step 4: Recompute measured angle from smeared positions + extra electronic smearing.
    theta_m = geometry_angle(src["pos"], scat_m, absorb_m)
    theta_m = float(np.clip(
        np.random.normal(theta_m, np.radians(config.ANGLE_SIGMA_DEG)), 0.0, np.pi
    ))

    cf = cone_features(scat_m, absorb_m, theta_m)

    return {
        "event_id": eid, "event_type": 1, "source_id": src["id"],
        "source_x": round(src["pos"][0], 3),
        "source_y": round(src["pos"][1], 3),
        "source_z": round(src["pos"][2], 3),
        "scatter_x": round(scat_m[0], 3),   "scatter_y": round(scat_m[1], 3),   "scatter_z": round(scat_m[2], 3),
        "absorb_x":  round(absorb_m[0], 3), "absorb_y":  round(absorb_m[1], 3), "absorb_z":  round(absorb_m[2], 3),
        "scatter_angle":   round(np.degrees(theta_m), 4),
        "electron_energy": round(Te_m, 6),
        **cf,
    }

# =============================================================================
# EVENT TYPE 4 — Fake coincidence (background)
# =============================================================================

def make_type4(eid):
    """
    Generates one Type 4 (fake coincidence) event.

    Physics:
        Fake coincidences occur when two completely UNRELATED gamma
        interactions fall within the coincidence time window of the
        electronics. Because the two hits have no physical relationship,
        the reconstructed Compton cone does NOT point to any real source.
        This is one of the most significant background sources in Compton
        cameras and is especially problematic at high count rates.

        Common causes:
            - Two decay gammas from different atoms arriving simultaneously
            - SiPM dark counts coinciding with a real gamma interaction
            - Gammas from outside the field of view

        We simulate this by randomly generating one scatter-like hit and one absorb-like hit, with
        no correlation between their positions or energies. The "scatter angle"
        is also random and has no physical meaning. The source position is set to NaN since there is no real source.
    Parameters
    ----------
    eid : int — unique event ID for this fake coincidence
    Returns
    -------
    dict — event data with random scatter and absorb positions, random angle, and NaN source
    """
    scat_pos   = rand_scatter()
    absorb_pos = rand_absorb()
    # Fake angle: random uniform — no physical meaning
    theta_fake = float(np.random.uniform(0, np.pi))
    cf = cone_features(scat_pos, absorb_pos, theta_fake)
    return {
        "event_id": eid, "event_type": 4, "source_id": -1,
        "source_x": np.nan, "source_y": np.nan, "source_z": np.nan,
        "scatter_x": round(scat_pos[0], 3),   "scatter_y": round(scat_pos[1], 3),   "scatter_z": round(scat_pos[2], 3),
        "absorb_x":  round(absorb_pos[0], 3), "absorb_y":  round(absorb_pos[1], 3), "absorb_z":  round(absorb_pos[2], 3),
        "scatter_angle":   round(np.degrees(theta_fake), 4),
        "electron_energy": round(float(np.random.uniform(0, config.INCIDENT_ENERGY_MEV)), 6),
        **cf,
    }

# =============================================================================
# Half EVENTS —  Type 2 and 3
# =============================================================================

def make_type2(src, eid):
    '''Generates one Type 2 (scatter single) event.
      Physics:
        A scatter single occurs when a gamma scatters in the front detector
        but the scattered gamma fails to reach the back detector. Causes:
            - Scattered gamma exits the field of view at a large angle
            - Scattered gamma is absorbed in material between detectors
            - Scattered gamma energy falls below the back detector threshold

        Only the front (scatter) detector fires.
        Absorption columns are NaN — no back-detector information exists.

        These events cannot form a complete Compton cone but carry
        partial information about the interaction vertex and electron energy.
    '''
    # Scatter single: only front detector fires.
    # Electron energy sampled from physically possible range using KN kinematics.
    theta_kn = sample_kn_theta(config.INCIDENT_ENERGY_MEV)
    E2       = scattered_energy(config.INCIDENT_ENERGY_MEV, theta_kn)
    Te       = smear_energy_val(electron_energy(config.INCIDENT_ENERGY_MEV, E2))
    scat_m   = smear_pos(rand_scatter())
    return {
        "event_id": eid, "event_type": 2, "source_id": src["id"],
        "source_x": round(src["pos"][0], 3),
        "source_y": round(src["pos"][1], 3),
        "source_z": round(src["pos"][2], 3),
        "scatter_x": round(scat_m[0], 3), "scatter_y": round(scat_m[1], 3), "scatter_z": round(scat_m[2], 3),
        "absorb_x": np.nan, "absorb_y": np.nan, "absorb_z": np.nan,
        "scatter_angle": np.nan, "electron_energy": round(Te, 6),
        "cone_axis_x": np.nan, "cone_axis_y": np.nan, "cone_axis_z": np.nan,
        "cone_opening": np.nan, "scat_absorb_dist": np.nan, "scat_depth_ratio":
            round(float(np.clip((config.SCATTER_PLANE_Z - scat_m[2]) / config.SCATTER_THICKNESS_MM, 0, 1)), 6),
    }

def make_type3(eid):
    '''Generates one Type 3 (absorb single) event.
      Physics:
        An absorb single occurs when a gamma is fully absorbed in the back detector
        without a corresponding scatter hit in the front detector. Causes:
            - Gamma enters back detector at a shallow angle, missing the front plane
            - Scatter hit falls below front detector threshold
            - Scattered gamma is absorbed in material between detectors
        Only the back (absorb) detector fires.
        Scatter columns are NaN — no front-detector information exists.
        These events provide no information about the scatter vertex or electron energy,
        but they do indicate that a gamma was absorbed and can be used for certain types of analysis.
    '''
    absorb_m = smear_pos(rand_absorb())
    E_dep    = smear_energy_val(float(np.random.exponential(0.3) + 0.1))
    return {
        "event_id": eid, "event_type": 3, "source_id": -1,
        "source_x": np.nan, "source_y": np.nan, "source_z": np.nan,
        "scatter_x": np.nan, "scatter_y": np.nan, "scatter_z": np.nan,
        "absorb_x": round(absorb_m[0], 3), "absorb_y": round(absorb_m[1], 3), "absorb_z": round(absorb_m[2], 3),
        "scatter_angle": np.nan, "electron_energy": np.nan,
        "cone_axis_x": np.nan, "cone_axis_y": np.nan, "cone_axis_z": np.nan,
        "cone_opening": np.nan, "scat_absorb_dist": np.nan, "scat_depth_ratio": np.nan,
    }

# =============================================================================
# END Half EVENTS
# =============================================================================

# =============================================================================
# SCENE ASSEMBLY
# =============================================================================

def build_scene(sources, n_events, start_id):
    n1 = int(n_events * config.FRAC_TYPE1)
    n4 = int(n_events * config.FRAC_TYPE4)
    n2 = int(n_events * config.FRAC_TYPE2)
    n3 = int(n_events * config.FRAC_TYPE3)
    n1 += n_events - (n1 + n4 + n2 + n3)

    events = []
    eid    = start_id

    accepted, attempts = 0, 0
    while accepted < n1 and attempts < n1 * 15:
        attempts += 1
        src = sources[np.random.randint(len(sources))]
        ev  = make_type1(src, eid)
        if ev is not None:
            events.append(ev); eid += 1; accepted += 1

    for _ in range(n4):
        events.append(make_type4(eid)); eid += 1
    for _ in range(n2):
        src = sources[np.random.randint(len(sources))]
        events.append(make_type2(src, eid)); eid += 1

    for _ in range(n3):
        events.append(make_type3(eid)); eid += 1

    np.random.shuffle(events) # final shuffle to mix event types
    return events, eid

# =============================================================================
# DATASET LOOP
# =============================================================================

def get_num_sources_for_file(file_idx):
    """
    Determines the number of sources for a given file based on config.NUM_SOURCES.
    
    Config options:
        int            → always return that fixed number
        (min, max)     → return random int in [min, max]
        [val1, val2...]→ cycle through list based on file index
    """
    cfg = config.NUM_SOURCES
    
    if isinstance(cfg, int):
        # Fixed number
        return cfg
    elif isinstance(cfg, tuple) and len(cfg) == 2:
        # Random range (min, max)
        return int(np.random.randint(cfg[0], cfg[1] + 1))
    elif isinstance(cfg, list):
        # Cycle through list
        return cfg[file_idx % len(cfg)]
    else:
        # Fallback
        print(f"  [WARN] Unrecognised NUM_SOURCES format: {cfg}. Defaulting to 1.")
        return 1

def generate_dataset(n_files, n_events, out_dir, seed_offset, label):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  {label.upper()}  |  {n_files} files  |  {n_events} events each")
    print(f"  Output: {out_dir}/")
    print(f"  NUM_SOURCES config: {config.NUM_SOURCES}")
    print(f"{'='*55}")
    global_eid = 0

    for idx in range(n_files):
        np.random.seed(seed_offset + idx)
        n_src = get_num_sources_for_file(idx)
        sources = make_sources(n_src)
        events, global_eid = build_scene(sources, n_events, global_eid)
        df   = pd.DataFrame(events, columns=CSV_COLUMNS)
        path = os.path.join(out_dir, f"SIM_events_source_{idx:04d}.csv")
        df.to_csv(path, index=False)

        if (idx + 1) % max(1, n_files // 10) == 0 or idx == n_files - 1:
            counts = df["event_type"].value_counts().sort_index().to_dict()
            print(f"  [{idx+1:04d}/{n_files}]  {len(df)} events (sources={n_src})  types={counts}")

    print(f"  Done. ~{n_files * n_events:,} total events.\n")

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compton Camera Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-files", type=int, default=config.DEFAULT_TRAIN_FILES)
    p.add_argument("--test-files",  type=int, default=config.DEFAULT_TEST_FILES)
    p.add_argument("--events",      type=int, default=config.DEFAULT_EVENTS)
    p.add_argument("--train-dir",   type=str, default=config.TRAIN_DIR)
    p.add_argument("--test-dir",    type=str, default=config.TEST_DIR)
    p.add_argument("--seed",        type=int, default=config.DEFAULT_SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n  Physics   : KN angle sampling  |  exp DOI  |  z-smearing ON")
    print(f"  Geometry  : src_z {config.SOURCE_Z_RANGE}  scat_z [{config.SCATTER_PLANE_Z - config.SCATTER_THICKNESS_MM:.0f},{config.SCATTER_PLANE_Z:.0f}]  absorb_z [{config.ABSORB_PLANE_Z - config.ABSORB_THICKNESS_MM:.0f},{config.ABSORB_PLANE_Z:.0f}]")
    print(f"  Features  : {len(CSV_COLUMNS) - 9} derived cone features added to raw 8")
    print(f"  Fractions : T1={config.FRAC_TYPE1}  T4={config.FRAC_TYPE4}  T2={config.FRAC_TYPE2}  T3={config.FRAC_TYPE3}")

    generate_dataset(args.train_files, args.events, args.train_dir, args.seed,         "train")
    generate_dataset(args.test_files,  args.events, args.test_dir,  args.seed + 10000, "test")

    print("  All datasets complete.")
    print(f"  Train → {args.train_dir}/")
    print(f"  Test  → {args.test_dir}/\n")
