# train.py — Training pipeline for ComptonSourceLocaliser
# DFG Project — University of Siegen
#
# Key fixes vs previous version:
#   1. Dataset handles ALL sources per scene (not just the first one).
#   2. Target heatmap accumulates a Gaussian blob for EVERY source.
#   3. Target coordinates store ALL source positions (padded to MAX_SOURCES).
#   4. Model output head now predicts MAX_SOURCES × 3 coordinates.
#   5. Loss uses per-source minimum matching (avoids ordering ambiguity).
#   6. source_mm stores all ground-truth sources for correct metric computation.
#   7. Derived cone features (14 total) handled correctly in normalisation.
#
# Usage:
#   python train.py
#   python train.py --epochs 100 --batch-size 16

import os
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import glob

import config
from model import ComptonSourceLocaliser, ComptonLocalisationLoss

# Maximum number of sources the model can predict per scene.
# Matches generate_data.py: 1–5 sources per scene.
MAX_SOURCES = 5


# =============================================================================
# NORMALISATION UTILITIES
# =============================================================================

def normalise_feature(value, feat_name, bounds_dict):
    lo, hi = bounds_dict[feat_name]
    return 2.0 * (value - lo) / (hi - lo) - 1.0

def denormalise_coordinate(normalised_value, coord_name):
    lo, hi = config.TARGET_BOUNDS[coord_name]
    return (normalised_value + 1.0) / 2.0 * (hi - lo) + lo


# =============================================================================
# TARGET HEATMAP — multi-source
# =============================================================================

def build_target_heatmap(sources_mm, heatmap_size=config.HEATMAP_SIZE,
                          x_range=config.HEATMAP_X_RANGE,
                          y_range=config.HEATMAP_Y_RANGE,
                          blob_sigma_px=2.0):
    """
    Builds a 2D heatmap with one Gaussian blob per source.

    FIX: previously only the first source was placed on the heatmap.
    Now all sources in the scene contribute.

    Parameters
    ----------
    sources_mm : np.ndarray (N_sources, 3) — true source positions in mm

    Returns
    -------
    np.ndarray (1, H, W) — multi-source probability heatmap
    """
    heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    y_grid, x_grid = np.ogrid[:heatmap_size, :heatmap_size]

    for sx, sy, _ in sources_mm:
        # Convert mm to pixel
        ix = int(np.clip((sx - x_range[0]) / (x_range[1] - x_range[0]) * heatmap_size,
                          0, heatmap_size - 1))
        iy = int(np.clip((sy - y_range[0]) / (y_range[1] - y_range[0]) * heatmap_size,
                          0, heatmap_size - 1))
        dist_sq = (x_grid - ix)**2 + (y_grid - iy)**2
        heatmap += np.exp(-dist_sq / (2.0 * blob_sigma_px**2))

    # Normalise so the peak is always 1.0
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap[np.newaxis, :, :].astype(np.float32)


def build_target_coords(sources_mm, max_sources=MAX_SOURCES):
    """
    Packs all source coordinates into a fixed-size array (max_sources × 3).

    FIX: previously only the first source was stored as the target.
    Now all sources are included, padded with NaN for absent slots.

    The model predicts max_sources × 3 values. Absent sources have NaN
    targets — the loss function ignores those positions.

    Parameters
    ----------
    sources_mm : np.ndarray (N_sources, 3) — true positions in mm

    Returns
    -------
    normalised_coords : np.ndarray (max_sources, 3) — targets in [-1, +1],
                        NaN for absent source slots
    valid_mask        : np.ndarray (max_sources,)   — True where source exists
    """
    n = len(sources_mm)
    normalised = np.full((max_sources, 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros(max_sources, dtype=bool)

    coord_names = ["source_x", "source_y", "source_z"]
    for i in range(min(n, max_sources)):
        for j, cname in enumerate(coord_names):
            normalised[i, j] = float(normalise_feature(
                sources_mm[i, j], cname, config.TARGET_BOUNDS
            ))
        valid_mask[i] = True

    return normalised, valid_mask


def build_confidence_labels(n_true_sources, max_sources=MAX_SOURCES):
    """
    Creates confidence labels for training.
    
    A prediction slot gets label=1 if it corresponds to a real source,
    label=0 otherwise (for unmatched/absent slots).
    
    Parameters
    ----------
    n_true_sources : int — number of true sources in the scene
    
    Returns
    -------
    np.ndarray (max_sources,) — confidence labels (1 for real, 0 for padding)
    """
    confidence = np.zeros(max_sources, dtype=np.float32)
    confidence[:min(n_true_sources, max_sources)] = 1.0
    return confidence


# =============================================================================
# DATASET
# =============================================================================

class ComptonEventDataset(Dataset):
    """
    Loads CSV event files. Returns one scene per index.

    FIX: collects ALL unique sources (up to MAX_SOURCES), not just the first.
    FIX: Added geometric data augmentation (X-flip and Y-flip) to quadruple effective dataset size.
    """

    def __init__(self, csv_file_paths, augment=False):
        self.csv_file_paths = csv_file_paths
        self.augment = augment
        print(f"  Dataset: {len(csv_file_paths)} scenes  |  up to {MAX_SOURCES} sources each")
        if augment:
            print(f"  ✓ Augmentation enabled (X/Y flips)")

    def __len__(self):
        return len(self.csv_file_paths)

    def __getitem__(self, idx):
        csv_path = self.csv_file_paths[idx]
        try:
            raw_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [WARN] Cannot read {csv_path}: {e}")
            return self._empty_sample()

        # Keep only Type 1 (true coincidence) events
        signal_df = raw_df[raw_df["event_type"] == config.SIGNAL_EVENT_TYPE].copy()
        if len(signal_df) == 0:
            return self._empty_sample()

        # ── Input: normalise 14 features ──────────────────────────────────
        raw_feat = signal_df[config.INPUT_FEATURE_NAMES].values.astype(np.float32)
        norm_feat = np.zeros_like(raw_feat)
        for col, name in enumerate(config.INPUT_FEATURE_NAMES):
            norm_feat[:, col] = normalise_feature(
                raw_feat[:, col], name, config.FEATURE_BOUNDS
            )

        # Replace any NaN in derived features with 0 (can arise from edge cases)
        norm_feat = np.nan_to_num(norm_feat, nan=0.0)

        # ── Pad / truncate events to MAX_EVENTS_PER_SCENE ─────────────────
        n_real = len(norm_feat)
        max_ev = config.MAX_EVENTS_PER_SCENE
        padded = np.zeros((max_ev, config.N_INPUT_FEATURES), dtype=np.float32)
        mask   = np.ones(max_ev, dtype=bool)   # True = padding

        if n_real >= max_ev:
            padded[:max_ev] = norm_feat[:max_ev]
            mask[:max_ev]   = False
        else:
            padded[:n_real] = norm_feat
            mask[:n_real]   = False

        # ── Collect ALL unique sources (FIX: was only taking the first) ───
        unique_sources = (signal_df[["source_x", "source_y", "source_z"]]
                          .drop_duplicates().values.astype(np.float32))
        # Clip to max we can handle
        unique_sources = unique_sources[:MAX_SOURCES]

        # ── Data augmentation: X-flip and Y-flip (geometric symmetry) ─────
        if self.augment:
            # X-flip: negate x-coordinates in features and targets
            if np.random.rand() < 0.5:
                x_feature_cols = [0, 3, 8]  # scatter_x, absorb_x, cone_axis_x
                norm_feat[:, x_feature_cols] *= -1
                unique_sources[:, 0] *= -1  # source_x
            
            # Y-flip: negate y-coordinates in features and targets
            if np.random.rand() < 0.5:
                y_feature_cols = [1, 4, 9]  # scatter_y, absorb_y, cone_axis_y
                norm_feat[:, y_feature_cols] *= -1
                unique_sources[:, 1] *= -1  # source_y

        # ── Build multi-source targets ─────────────────────────────────────
        target_heatmap = build_target_heatmap(unique_sources)
        target_coords, source_valid_mask = build_target_coords(unique_sources)
        confidence_labels = build_confidence_labels(len(unique_sources))
        count_target = len(unique_sources)  # true number of sources (0-5)

        return {
            "events":            torch.tensor(padded,             dtype=torch.float32),
            "padding_mask":      torch.tensor(mask,               dtype=torch.bool),
            "target_heatmap":    torch.tensor(target_heatmap,     dtype=torch.float32),
            "target_coords":     torch.tensor(target_coords,      dtype=torch.float32),
            "source_valid_mask": torch.tensor(source_valid_mask,  dtype=torch.bool),
            "confidence_labels": torch.tensor(confidence_labels,  dtype=torch.float32),
            "count_target":      torch.tensor(count_target,       dtype=torch.long),
            "sources_mm":        torch.tensor(unique_sources,     dtype=torch.float32),
            "n_sources":         torch.tensor(len(unique_sources), dtype=torch.long),
        }

    def _empty_sample(self):
        return {
            "events":            torch.zeros(config.MAX_EVENTS_PER_SCENE, config.N_INPUT_FEATURES),
            "padding_mask":      torch.ones(config.MAX_EVENTS_PER_SCENE, dtype=torch.bool),
            "target_heatmap":    torch.zeros(1, config.HEATMAP_SIZE, config.HEATMAP_SIZE),
            "target_coords":     torch.full((MAX_SOURCES, 3), float("nan")),
            "source_valid_mask": torch.zeros(MAX_SOURCES, dtype=torch.bool),
            "confidence_labels": torch.zeros(MAX_SOURCES, dtype=torch.float32),
            "count_target":      torch.tensor(0, dtype=torch.long),
            "sources_mm":        torch.zeros(1, 3),
            "n_sources":         torch.tensor(0, dtype=torch.long),
        }


# =============================================================================
# CUSTOM COLLATE — handles variable n_sources across scenes in a batch
# =============================================================================

def collate_scenes(batch):
    """
    Custom collate function.

    sources_mm varies in shape (1–5, 3) across scenes so we cannot use
    the default collate. Everything else is fixed-size and collates normally.
    """
    keys_fixed = ["events", "padding_mask", "target_heatmap",
                  "target_coords", "source_valid_mask", "confidence_labels",
                  "count_target", "n_sources"]

    collated = {k: torch.stack([b[k] for b in batch]) for k in keys_fixed}

    # sources_mm: store as a list of tensors (variable size)
    collated["sources_mm"] = [b["sources_mm"] for b in batch]

    return collated


# =============================================================================
# LOSS — multi-source aware with count and confidence
# =============================================================================

# Use the updated loss from model.py which includes count and confidence terms
from model import ComptonLocalisationLoss as MultiSourceLoss


def compute_minimum_cost_matching(pred_coords, target_coords, source_valid_mask):
    """
    Performs minimum-cost matching between predicted and true sources.
    
    FIX: Confidence labels must be built AFTER matching, not before.
    
    Uses Hungarian algorithm for one-to-one assignment (no two true sources
    can claim the same prediction slot).
    
    Parameters
    ----------
    pred_coords : torch.Tensor (batch, MAX_SOURCES, 3) - predicted coordinates
    target_coords : torch.Tensor (batch, MAX_SOURCES, 3) - true coordinates (NaN for absent)
    source_valid_mask : torch.Tensor (batch, MAX_SOURCES) - True where source exists
    
    Returns
    -------
    matched_pred_indices : list of sets - for each batch, which prediction slots were matched
    count_targets : torch.Tensor (batch,) - derived from matching (not dataset)
    """
    try:
        from scipy.optimize import linear_sum_assignment
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("[WARNING] scipy not available, falling back to greedy matching")
    
    batch_size = pred_coords.shape[0]
    matched_pred_indices = []
    count_targets = []
    
    for b in range(batch_size):
        valid = source_valid_mask[b]  # (MAX_SOURCES,)
        
        if valid.sum() == 0:
            # No true sources - no matches
            matched_pred_indices.append(set())
            count_targets.append(0)
            continue
        
        true_pts = target_coords[b][valid]  # (n_true, 3)
        pred_pts = pred_coords[b]           # (MAX_SOURCES, 3)
        
        # Build cost matrix: (n_true, MAX_SOURCES)
        n_true = len(true_pts)
        n_pred = len(pred_pts)
        cost_matrix = torch.zeros(n_true, n_pred, device=pred_coords.device)
        
        for i in range(n_true):
            for j in range(n_pred):
                cost_matrix[i, j] = ((true_pts[i] - pred_pts[j]) ** 2).sum()
        
        # Convert to numpy for scipy
        if has_scipy:
            cost_np = cost_matrix.detach().cpu().numpy()  # detach first!
            true_idx, pred_idx = linear_sum_assignment(cost_np)
            matched_slots = set(pred_idx.tolist())
        else:
            # Fallback: greedy with uniqueness constraint
            used_preds = set()
            matched_slots = set()
            
            # Sort by minimum distance (most constrained first)
            min_dists, _ = cost_matrix.min(dim=1)
            order = torch.argsort(min_dists)
            
            for i in order:
                d = cost_matrix[i].clone()
                d[list(used_preds)] = float('inf')
                best_j = int(d.argmin().item())
                
                if d[best_j] < float('inf'):
                    used_preds.add(best_j)
                    matched_slots.add(best_j)
        
        matched_pred_indices.append(matched_slots)
        count_targets.append(len(matched_slots))
    
    return matched_pred_indices, torch.tensor(count_targets, dtype=torch.long, device=pred_coords.device)


def build_confidence_labels_from_matching(matched_pred_indices, batch_size):
    """
    Creates confidence labels AFTER matching.
    
    FIX: Previously labels were assigned to slots 0..N-1, but matching might
    assign true source 0 to prediction slot 3. This fixes that bug.
    
    Parameters
    ----------
    matched_pred_indices : list of sets - matched prediction slot indices per batch
    batch_size : int
    
    Returns
    -------
    confidence_labels : torch.Tensor (batch, MAX_SOURCES) - 1 if slot was matched, else 0
    """
    from train import MAX_SOURCES
    
    confidence_labels = torch.zeros(batch_size, MAX_SOURCES, dtype=torch.float32)
    
    for b, matched_slots in enumerate(matched_pred_indices):
        for slot_idx in range(MAX_SOURCES):
            if slot_idx in matched_slots:
                confidence_labels[b, slot_idx] = 0.9  # soft positive label
            else:
                confidence_labels[b, slot_idx] = 0.1  # soft negative label
    
    return confidence_labels


# =============================================================================
# METRIC HELPERS
# =============================================================================

def compute_errors_mm(pred_coords_norm, sources_mm_list):
    """
    Computes mm errors for multi-source scenes using nearest-neighbour matching.

    FIX: was comparing a single predicted point to a single ground truth.
    Now matches each true source to its nearest prediction.

    Parameters
    ----------
    pred_coords_norm : torch.Tensor (B, MAX_SOURCES, 3) — normalised predictions
    sources_mm_list  : list of tensors, each (n_sources, 3) in mm

    Returns
    -------
    dict of mean absolute errors in mm
    """
    coord_names = ["source_x", "source_y", "source_z"]
    all_err_x, all_err_y, all_err_z, all_err_xy, all_err_xyz = [], [], [], [], []

    pred_np = pred_coords_norm.cpu().numpy()   # (B, MAX_SOURCES, 3)

    for b, true_mm in enumerate(sources_mm_list):
        true_np = true_mm.numpy()              # (n_true, 3)

        # Denormalise all predicted positions to mm
        pred_mm = np.stack([
            denormalise_coordinate(pred_np[b, :, j], coord_names[j])
            for j in range(3)
        ], axis=1)                             # (MAX_SOURCES, 3)

        # Match each true source to nearest prediction
        for tx, ty, tz in true_np:
            dists = np.sqrt(
                (pred_mm[:, 0] - tx)**2 +
                (pred_mm[:, 1] - ty)**2 +
                (pred_mm[:, 2] - tz)**2
            )
            best = np.argmin(dists)
            px, py, pz = pred_mm[best]
            ex, ey, ez = abs(tx-px), abs(ty-py), abs(tz-pz)
            all_err_x.append(ex); all_err_y.append(ey); all_err_z.append(ez)
            all_err_xy.append(np.sqrt(ex**2 + ey**2))
            all_err_xyz.append(np.sqrt(ex**2 + ey**2 + ez**2))

    if not all_err_xyz:
        return {k: 0.0 for k in ["err_x_mm","err_y_mm","err_z_mm","err_xy_mm","err_xyz_mm"]}

    return {
        "err_x_mm":   float(np.mean(all_err_x)),
        "err_y_mm":   float(np.mean(all_err_y)),
        "err_z_mm":   float(np.mean(all_err_z)),
        "err_xy_mm":  float(np.mean(all_err_xy)),
        "err_xyz_mm": float(np.mean(all_err_xyz)),
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def run_one_epoch(model, loader, criterion, optimizer, device, is_training):
    model.train() if is_training else model.eval()

    acc_total = acc_heat = acc_coord = acc_conf = acc_count = 0.0
    all_err_xyz = []
    count_correct = 0.0
    n_batches   = 0

    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for batch in loader:
            events       = batch["events"].to(device)
            pad_mask     = batch["padding_mask"].to(device)
            t_heatmap    = batch["target_heatmap"].to(device)
            t_coords     = batch["target_coords"].to(device)
            valid_mask   = batch["source_valid_mask"].to(device)
            sources_mm   = batch["sources_mm"]           # list, stays on CPU

            pred_heatmap, pred_coords_and_conf, pred_count_logits = model(events, pad_mask)

            # Extract coordinates for matching (first 3 channels)
            pred_coords_for_matching = pred_coords_and_conf[:, :, :3]
            
            # FIX Issue 1 & 2: Compute matching FIRST, then derive confidence and count labels
            matched_indices, derived_count_targets = compute_minimum_cost_matching(
                pred_coords_for_matching, t_coords, valid_mask
            )
            
            # Build confidence labels from matching result (not from dataset)
            batch_size = pred_coords_and_conf.shape[0]
            conf_labels = build_confidence_labels_from_matching(matched_indices, batch_size)
            conf_labels = conf_labels.to(device)
            
            # Also build matched slot mask for regularization (Issue 4)
            matched_slot_mask = torch.zeros(batch_size, MAX_SOURCES, dtype=torch.bool, device=device)
            for b, matched_slots in enumerate(matched_indices):
                for slot_idx in matched_slots:
                    matched_slot_mask[b, slot_idx] = True
            
            # Use derived count targets (from matching), not dataset labels
            # This prevents contradiction between count and confidence losses
            count_tgt = derived_count_targets

            # Compute loss with post-matching labels and matched_slot_mask for regularization
            total, lh, lc, lconf, lcount = criterion(
                pred_heatmap, t_heatmap,
                pred_coords_and_conf, t_coords,
                conf_labels, pred_count_logits, count_tgt,
                matched_slot_mask  # FIX Issue 4: pass for unmatched slot regularization
            )

            if is_training:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
                optimizer.step()

            acc_total += total.item()
            acc_heat  += lh.item()
            acc_coord += lc.item()
            acc_conf  += lconf.item()
            acc_count += lcount.item()

            # Track count accuracy (FIX Issue 8 - missing metric)
            # Definition: EXACT match - predicted N must equal true N exactly
            # This is strict but necessary for reliable multi-source detection
            pred_counts = torch.argmax(pred_count_logits, dim=-1)
            count_correct += (pred_counts == count_tgt).float().sum().item()

            # Extract coordinates (first 3 channels) for error computation
            errs = compute_errors_mm(pred_coords_for_matching.detach(), sources_mm)
            all_err_xyz.append(errs["err_xyz_mm"])
            n_batches += 1

    return {
        "total_loss":   acc_total   / n_batches,
        "heatmap_loss": acc_heat    / n_batches,
        "coord_loss":   acc_coord   / n_batches,
        "conf_loss":    acc_conf    / n_batches,
        "count_loss":   acc_count   / n_batches,
        "err_xyz_mm":   float(np.mean(all_err_xyz)),
        "count_acc":    count_correct / (n_batches * loader.batch_size if n_batches > 0 else 1),
    }


# =============================================================================
# LOGGER
# =============================================================================

class Logger:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def log(self, epoch, tr, va, lr, t):
        self.rows.append({
            "epoch": epoch,
            "train_loss": round(tr["total_loss"],   6),
            "train_heat": round(tr["heatmap_loss"], 6),
            "train_coord": round(tr["coord_loss"],  6),
            "train_conf": round(tr["conf_loss"],   6),
            "train_count": round(tr["count_loss"],  6),
            "train_count_acc": round(tr.get("count_acc", 0.0), 4),
            "val_loss":   round(va["total_loss"],   6),
            "val_heat":   round(va["heatmap_loss"], 6),
            "val_coord":  round(va["coord_loss"],  6),
            "val_conf":   round(va["conf_loss"],   6),
            "val_count":  round(va["count_loss"],  6),
            "val_count_acc": round(va.get("count_acc", 0.0), 4),
            "val_xyz_mm": round(va["err_xyz_mm"],   4),
            "lr":         lr,
            "time_s":     round(t, 1),
        })
        pd.DataFrame(self.rows).to_csv(self.path, index=False)

    def print(self, epoch, n_ep, tr, va, lr, t, best):
        marker = " ← BEST" if best else ""
        # FIX Issue 6: Show all individual loss components for debugging
        print(f"  Epoch {epoch:4d}/{n_ep}"
              f"  | Train: {tr['total_loss']:.3f} (H:{tr['heatmap_loss']:.3f} C:{tr['coord_loss']:.3f})"
              f"  | Val: {va['total_loss']:.3f} (H:{va['heatmap_loss']:.3f} Co:{va['coord_loss']:.3f} Cf:{va['conf_loss']:.3f} N:{va['count_loss']:.3f})"
              f"  | Acc:{va.get('count_acc', 0.0):.1%} XYZ:{va['err_xyz_mm']:6.2f}mm"
              f"  | LR: {lr:.2e}"
              f"  | {t:5.1f}s{marker}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(args):
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    device = config.DEVICE

    print(f"\n{'='*65}")
    print(f"  COMPTON SOURCE LOCALISER — TRAINING")
    print(f"  Device: {device}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}")
    print(f"  Sources per scene: 1–{MAX_SOURCES}  |  Features: {config.N_INPUT_FEATURES}")
    print(f"{'='*65}\n")

    csv_files = sorted(glob.glob(os.path.join(config.TRAIN_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in '{config.TRAIN_DIR}'. Run generate_data.py first.")

    print(f"  Found {len(csv_files)} CSV files.")
    dataset = ComptonEventDataset(csv_files, augment=True)

    n_train = int(0.80 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    print(f"  Train: {n_train}  |  Val: {n_val}\n")

    # Create separate datasets for train (with augmentation) and val (without)
    train_csvs = [csv_files[i] for i in range(len(csv_files)) if i < n_train]
    val_csvs = [csv_files[i] for i in range(len(csv_files)) if i >= n_train]
    train_ds_aug = ComptonEventDataset(train_csvs, augment=True)
    val_ds_noaug = ComptonEventDataset(val_csvs, augment=False)

    train_loader = DataLoader(train_ds_aug, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type=="cuda"),
                              collate_fn=collate_scenes)
    val_loader   = DataLoader(val_ds_noaug,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type=="cuda"),
                              collate_fn=collate_scenes)

    model     = ComptonSourceLocaliser().to(device)
    criterion = MultiSourceLoss(
        lambda_heatmap=config.LAMBDA_HEATMAP,
        lambda_coord=config.LAMBDA_COORD,
        lambda_confidence=config.LAMBDA_CONFIDENCE,
        lambda_count=config.LAMBDA_COUNT,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=config.LR_FACTOR,
                                                      patience=config.LR_PATIENCE)
    logger    = Logger(os.path.join(config.MODEL_DIR, "training_log.csv"))
    best_path = os.path.join(config.MODEL_DIR, "best_model.pth")

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Loss weights: H={config.LAMBDA_HEATMAP:.1f} C={config.LAMBDA_COORD:.1f} Conf={config.LAMBDA_CONFIDENCE:.1f} Count={config.LAMBDA_COUNT:.1f}")
    print(f"  {'─'*63}")

    best_val  = float("inf")
    no_improve = 0

    def get_warmup_lr(epoch, warmup_epochs=5, base_lr=1e-3, min_lr=1e-4):
        if epoch <= warmup_epochs:
            return min_lr + (base_lr - min_lr) * (epoch / warmup_epochs)
        return base_lr

    warmup_epochs = 5

    for epoch in range(1, args.epochs + 1):
        if epoch <= warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, warmup_epochs=warmup_epochs, base_lr=args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        t0 = time.time()
        tr = run_one_epoch(model, train_loader, criterion, optimizer, device, True)
        va = run_one_epoch(model, val_loader,   criterion, None,      device, False)
        t  = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        is_best = va["total_loss"] < best_val
        if is_best:
            best_val   = va["total_loss"]
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    best_val,
                "val_xyz_mm":  va["err_xyz_mm"],
                "max_sources": config.MAX_SOURCES,
                "n_features":  config.N_INPUT_FEATURES,
            }, best_path)
        else:
            no_improve += 1

        if epoch > warmup_epochs:
            scheduler.step(va["total_loss"])
        logger.log(epoch, tr, va, lr, t)
        logger.print(epoch, args.epochs, tr, va, lr, t, is_best)

        if no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} ({config.EARLY_STOP_PATIENCE} epochs without improvement).")
            break

    print(f"\n  Best val loss: {best_val:.6f}  →  {best_path}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--epochs",         type=int,   default=config.EPOCHS)
    p.add_argument("--batch-size",     type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=config.LEARNING_RATE)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_arguments())
