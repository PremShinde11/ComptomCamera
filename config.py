# =============================================================================
# config.py
# =============================================================================
# Single source of truth for ALL constants used across the project.
# Every other script imports from here. Nothing is hardcoded elsewhere.
#
# DFG Compton Camera Project — University of Siegen
# =============================================================================

import os
import torch


# =============================================================================
# PATHS
# =============================================================================

TRAIN_DIR   = os.path.join("data", "trainn")   # training CSV files
TEST_DIR    = os.path.join("data", "test")    # test CSV files
MODEL_DIR   = "models"                        # saved model checkpoints
RESULTS_DIR = "results"                       # evaluation plots and metrics

# ── Dataset defaults ──────────────────────────────────────────────────────────
DEFAULT_TRAIN_FILES = 0
DEFAULT_TEST_FILES  = 500
DEFAULT_EVENTS      = 1500
DEFAULT_SEED        = 0

# ── Event type fractions — must sum to 1.0 ───────────────────────────────────
FRAC_TYPE1 = 0.65
FRAC_TYPE4 = 0.15
FRAC_TYPE2 = 0.10   # scatter singles (Compton scatter in PMMA but no absorption in LYSO)
FRAC_TYPE3 = 0.10   # absorb  singles (energy deposition in LYSO without a recorded scatter in PMMA)

# ── Detector resolution ───────────────────────────────────────────────────────
POS_XY_SIGMA_MM   = 1.5    # lateral (x,y) position resolution, mm
POS_Z_SIGMA_MM    = 2.0    # depth (z) resolution — finite DOI reconstruction
ENERGY_RES_FRAC   = 0.05   # energy resolution sigma/mean
ANGLE_SIGMA_DEG   = 2.0    # additional electronic angular smearing

# ── Attenuation coefficients (for exponential depth-of-interaction) ───────────
# Linear attenuation coefficient mu for 1 MeV gammas:
#   PMMA (scatter): ~0.070 cm^-1  = 0.0070 mm^-1
#   LYSO (absorb):  ~0.870 cm^-1  = 0.0870 mm^-1
MU_PMMA_PER_MM = 0.0070
MU_LYSO_PER_MM = 0.0870

# =============================================================================
# GEOMETRY  (all distances in mm)
# =============================================================================

# Source volume — sources are distributed in a 3D volume, not a flat plane
SOURCE_X_RANGE       = (-80.0,   80.0)   # mm — lateral extent in x
SOURCE_Y_RANGE       = (-80.0,   80.0)   # mm — lateral extent in y
SOURCE_Z_RANGE       = (-20.0,   20.0)   # mm — depth range (key prediction target)

# Scatter detector (PMMA radiator block — front detector)
SCATTER_PLANE_Z      = -50.0             # mm — nominal front face of PMMA block
SCATTER_THICKNESS_MM =  25.0             # mm — PMMA block thickness (from proposal)
SCATTER_X_RANGE      = (-100.0, 100.0)   # mm — active detector area
SCATTER_Y_RANGE      = (-100.0, 100.0)   # mm — active detector area

# Absorption detector (LYSO scintillator — back detector)
ABSORB_PLANE_Z       = -100.0            # mm — nominal front face of LYSO crystals
ABSORB_THICKNESS_MM  =  20.0             # mm — crystal length (from proposal)
ABSORB_X_RANGE       = (-100.0, 100.0)   # mm — active detector area
ABSORB_Y_RANGE       = (-100.0, 100.0)   # mm — active detector area

# =============================================================================
# PHYSICS CONSTANTS  (all energies in MeV)
# =============================================================================

INCIDENT_ENERGY_MEV     = 1.0            # E1: initial gamma energy
ELECTRON_REST_MASS_MEV  = 0.5109989461   # m_e * c^2
CHERENKOV_THRESHOLD_MEV = 0.18           # minimum detectable electron energy in PMMA

# =============================================================================
# INPUT FEATURES
# =============================================================================
# 14 features per event — 8 raw measurements + 6 derived cone geometry features.
#
# Raw measurements (what the detector records):
#   scatter_x/y/z     — Compton scatter point in the PMMA block
#   absorb_x/y/z      — absorption point in the LYSO scintillator
#   scatter_angle     — Compton scatter angle theta (degrees)
#   electron_energy   — recoil electron kinetic energy (MeV)
#
# Derived cone geometry features (computed from the raw measurements):
#   cone_axis_x/y/z   — unit vector from absorb → scatter (gamma direction)
#                        This is the cone axis — explicitly encodes the 3D
#                        direction of the incoming gamma ray.
#   cone_opening      — cos(theta) — the cone half-angle in a form the network
#                        can use directly without computing trigonometry.
#   scat_absorb_dist  — Euclidean distance scatter→absorb (mm).
#                        Encodes how far the scattered gamma travelled.
#   scat_depth_ratio  — scatter_z normalised within the PMMA block thickness.
#                        Tells the network how deep in the radiator the
#                        interaction occurred — important for Cherenkov reconstruction.
#
# WHY ADD DERIVED FEATURES?
#   The Transformer must learn these geometric relationships from scratch if
#   only raw positions are given. Providing them explicitly:
#     1. Reduces the effective learning problem.
#     2. Makes the cone axis directly accessible — the single most important
#        geometric object in Compton camera reconstruction.
#     3. Improves convergence speed and final accuracy.

INPUT_FEATURE_NAMES = [
    # ── Raw measurements ──────────────────────────────────────────────────
    "scatter_x",
    "scatter_y",
    "scatter_z",
    "absorb_x",
    "absorb_y",
    "absorb_z",
    "scatter_angle",
    "electron_energy",
    # ── Derived cone geometry ─────────────────────────────────────────────
    "cone_axis_x",      # unit vector components (absorb → scatter direction)
    "cone_axis_y",
    "cone_axis_z",
    "cone_opening",     # cos(theta) — the cone half-angle
    "scat_absorb_dist", # Euclidean distance scatter → absorb  (mm)
    "scat_depth_ratio", # scatter_z position within PMMA block  [0, 1]
]
N_INPUT_FEATURES = len(INPUT_FEATURE_NAMES)   # = 14

# Only use true coincidence events for source localisation training
SIGNAL_EVENT_TYPE = 1

# =============================================================================
# FEATURE NORMALISATION BOUNDS
# =============================================================================
# Each feature is scaled to [-1, +1] using min/max bounds.
# Bounds for raw features come from detector geometry.
# Bounds for derived features come from their mathematical definitions.
# IMPORTANT: these must stay fixed between training and inference.

# Max possible scatter→absorb distance: diagonal across detector volumes
# sqrt((200)^2 + (200)^2 + (70)^2) ≈ 296 mm  → round up to 300
_MAX_SCAT_ABSORB_DIST = 300.0

FEATURE_BOUNDS = {
    # Raw measurements
    "scatter_x":        (-105.0,  105.0),   # detector active area + margin
    "scatter_y":        (-105.0,  105.0),
    "scatter_z":        ( -75.0,  -50.0),   # PMMA block z range
    "absorb_x":         (-105.0,  105.0),
    "absorb_y":         (-105.0,  105.0),
    "absorb_z":         (-120.0, -100.0),   # LYSO crystal z range
    "scatter_angle":    (   0.0,  180.0),   # degrees — full Compton range
    "electron_energy":  (   0.0,    1.0),   # MeV — max = INCIDENT_ENERGY_MEV
    # Derived cone geometry
    "cone_axis_x":      (  -1.0,    1.0),   # unit vector component
    "cone_axis_y":      (  -1.0,    1.0),
    "cone_axis_z":      (  -1.0,    1.0),
    "cone_opening":     (  -1.0,    1.0),   # cos(theta) ∈ [-1, +1]
    "scat_absorb_dist": (   0.0,  _MAX_SCAT_ABSORB_DIST),
    "scat_depth_ratio": (   0.0,    1.0),   # already in [0, 1]
}

# =============================================================================
# OUTPUT TARGETS
# =============================================================================
# The model predicts the 3D source position (x, y, z).
# Targets are also normalised to [-1, +1] during training.

TARGET_FEATURE_NAMES = ["source_x", "source_y", "source_z"]
N_TARGET_FEATURES    = 3

TARGET_BOUNDS = {
    "source_x": (-80.0,  80.0),   # mm — matches SOURCE_X_RANGE
    "source_y": (-80.0,  80.0),   # mm — matches SOURCE_Y_RANGE
    "source_z": (-20.0,  20.0),   # mm — matches SOURCE_Z_RANGE
}

# Spatial heatmap output (for the CNN decoder head)
HEATMAP_SIZE    = 64              # pixels per side (64 × 64 grid)
HEATMAP_X_RANGE = (-100.0, 100.0) # mm — world extent the heatmap covers
HEATMAP_Y_RANGE = (-100.0, 100.0) # mm

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# Event padding — all scenes are padded/truncated to this number of events.
# Set to cover ~80% of Type 1 events in a 1500-event file (65% Type 1 = 975).
MAX_EVENTS_PER_SCENE = 1000

# Transformer encoder
D_MODEL         = 128    # embedding dimension (must be divisible by N_HEADS)
N_HEADS         =   4    # number of self-attention heads
N_ENCODER_LAYERS=   4    # number of stacked Transformer encoder layers
DIM_FEEDFORWARD = 256    # inner dimension of Transformer FFN sublayer
DROPOUT         = 0.1    # dropout rate applied throughout the Transformer

# CNN spatial decoder
LATENT_SPATIAL_DIM = 4           # reshape latent to (D_MODEL, 4, 4) before decoding
DECODER_CHANNELS   = [128, 64, 32, 16]  # output channels at each upsampling stage
# Four upsampling stages: 4 -> 8 -> 16 -> 32 -> 64 (matches HEATMAP_SIZE=64)

# Multi-source prediction settings
MAX_SOURCES          = 5
CONFIDENCE_THRESHOLD = 0.5   # discard predictions below this (tune post-training)
LAMBDA_COUNT         = 0.05   # weight for count classification loss (reduced to prioritize coordinate learning)
LAMBDA_CONFIDENCE    = 2.0   # weight for per-prediction confidence loss
LAMBDA_REGULARISE    = 0.05   # push ghost predictions toward origin (tune if slots drift or cluster)

# =============================================================================
# TRAINING
# =============================================================================

BATCH_SIZE         =  16
LEARNING_RATE      = 1e-3  # initial learning rate for AdamW optimizer
WEIGHT_DECAY       = 1e-4  # L2 regularisation strength (AdamW's "weight decay" parameter)
EPOCHS             = 100
EARLY_STOP_PATIENCE=  25   # stop if val loss does not improve for N epochs
LR_PATIENCE        =   8   # reduce LR after N epochs without improvement
LR_FACTOR          = 0.7   # multiply LR by this factor on plateau
GRAD_CLIP_MAX_NORM = 1.0   # gradient clipping maximum L2 norm

# Coordinate loss is weighted much higher — gives direct, strong gradient
# signal on mm accuracy. Heatmap BCE otherwise dominates and slows XY.
LAMBDA_COORD   = 6.0   # weight of direct coordinate MSE loss (MLP head) - increased to prioritize position learning
LAMBDA_HEATMAP = 0.1   # weight of spatial heatmap BCE loss  (CNN head)

# =============================================================================
# EVALUATION
# =============================================================================

HISTOGRAM_BINS = 40     # number of bins in error distribution histograms
SAVE_PLOTS     = True   # save PNG figures to RESULTS_DIR

# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
