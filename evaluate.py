# evaluate.py — Full test-set evaluation for ComptonSourceLocaliser
# DFG Project — University of Siegen
#
# Fixes vs previous version:
#   1. Reads ALL sources from each test CSV (not just the first).
#   2. Nearest-neighbour matching between predictions and all true sources.
#   3. Tracks miss rate (true sources with no nearby prediction).
#   4. Histograms now reflect per-source errors across multi-source scenes.
#
# Usage:
#   python evaluate.py
#   python evaluate.py --model-path models/best_model.pth

import os
import argparse

import numpy as np
import pandas as pd
import torch
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
from model import ComptonSourceLocaliser
from train import (
    ComptonEventDataset, denormalise_coordinate,
    MAX_SOURCES, collate_scenes
)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: '{model_path}'. Run train.py first.")

    ck = torch.load(model_path, map_location=device)
    print(f"  Checkpoint: epoch={ck.get('epoch','?')}  "
          f"val_loss={ck.get('val_loss', float('nan')):.4f}  "
          f"val_xyz={ck.get('val_xyz_mm', float('nan')):.2f} mm")

    model = ComptonSourceLocaliser().to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


# =============================================================================
# PREDICTION EXTRACTION — finds up to MAX_SOURCES peaks from heatmap
# =============================================================================

def extract_heatmap_peaks(heatmap_np, depth_unused=None,
                           sigma=2.0, rel_thr=0.5, abs_thr=0.3,
                           min_dist_px=10, border_px=8):
    """
    Finds local maxima in the heatmap as predicted XY positions.
    Returns list of (x_mm, y_mm) — Z comes from the coordinate head.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    smooth = gaussian_filter(heatmap_np, sigma=sigma)

    # Zero out border to prevent edge artifacts
    smooth[:border_px, :]  = 0
    smooth[-border_px:, :] = 0
    smooth[:, :border_px]  = 0
    smooth[:, -border_px:] = 0

    thr   = max(smooth.max() * rel_thr, abs_thr)
    local = maximum_filter(smooth, size=min_dist_px) == smooth
    peaks = local & (smooth > thr)

    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return []

    # Sort by intensity descending, keep top MAX_SOURCES
    strengths = smooth[ys, xs]
    order     = np.argsort(strengths)[::-1]
    results   = []
    for i in order[:MAX_SOURCES]:
        x_mm = (xs[i] / config.HEATMAP_SIZE) * (config.HEATMAP_X_RANGE[1] - config.HEATMAP_X_RANGE[0]) + config.HEATMAP_X_RANGE[0]
        y_mm = (ys[i] / config.HEATMAP_SIZE) * (config.HEATMAP_Y_RANGE[1] - config.HEATMAP_Y_RANGE[0]) + config.HEATMAP_Y_RANGE[0]
        results.append((float(x_mm), float(y_mm)))
    return results


# =============================================================================
# SINGLE SCENE INFERENCE — uses count prediction + confidence filtering
# =============================================================================

def predict_scene(model, sample, device):
    """
    Runs inference on one scene.

    Uses the new architecture:
        1. Predict source count from count head
        2. Get all coordinates + confidence from coordinate head
        3. Keep top-N predictions by confidence where N = predicted count
        4. Apply confidence threshold as upper bound

    Returns
    -------
    pred_coords_mm : np.ndarray (N_pred, 3) — filtered predicted positions in mm
    pred_confidences : np.ndarray (N_pred,) — confidence scores for kept predictions
    true_sources_mm : np.ndarray (N_true, 3)     — all ground truth positions in mm
    pred_heatmap    : np.ndarray (H, W)
    pred_n          : int — predicted source count
    """
    events   = sample["events"].unsqueeze(0).to(device)
    pad_mask = sample["padding_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_heatmap_t, pred_coords_and_conf_t, pred_count_logits_t = model(events, pad_mask)

    # Extract predicted count: argmax over logits
    pred_count = int(torch.argmax(pred_count_logits_t[0], dim=-1).item())  # scalar 0-5

    # Extract coordinates and confidence
    # pred_coords_and_conf_t: (1, MAX_SOURCES, 4) - last dim is (x,y,z,confidence_logit)
    pred_norm = pred_coords_and_conf_t.squeeze(0).cpu().numpy()   # (MAX_SOURCES, 4)
    
    # Get confidence scores via sigmoid
    confidences = 1.0 / (1.0 + np.exp(-pred_norm[:, 3]))  # sigmoid on logit
    
    # Sort by confidence descending
    sorted_indices = np.argsort(confidences)[::-1]
    
    # FIX Issue 3: Resolve conflict between count prediction and confidence threshold
    n_above_threshold = int((confidences >= config.CONFIDENCE_THRESHOLD).sum())
    
    # Use count prediction but never exceed what confidence supports
    if n_above_threshold > 0:
        final_n = min(pred_count, n_above_threshold)
    else:
        final_n = pred_count  # trust count prediction even if no conf above threshold
    
    final_n = max(final_n, 1)  # always return at least 1 prediction
    
    # Keep top-N predictions where N = final_n
    top_n_indices = sorted_indices[:final_n]
    
    coord_names = ["source_x", "source_y", "source_z"]
    pred_mm_list = []
    pred_conf_list = []
    for idx in top_n_indices:
        xyz = [denormalise_coordinate(pred_norm[idx, j], coord_names[j]) for j in range(3)]
        pred_mm_list.append(xyz)
        pred_conf_list.append(confidences[idx])
    
    pred_mm = np.array(pred_mm_list) if pred_mm_list else np.zeros((0, 3))
    pred_conf = np.array(pred_conf_list) if pred_conf_list else np.zeros((0,))

    true_mm = sample["sources_mm"].numpy()               # (N_true, 3)
    heatmap = pred_heatmap_t.squeeze().cpu().numpy()     # (H, W)

    return pred_mm, pred_conf, true_mm, heatmap, pred_count


# =============================================================================
# NEAREST-NEIGHBOUR MATCHING
# =============================================================================

def match_sources(true_mm, pred_mm):
    """
    Greedy nearest-neighbour matching: each true source gets its closest
    available prediction. Returns list of (true, pred, errors) dicts.
    """
    used     = set()
    matches  = []

    for tx, ty, tz in true_mm:
        best_dist, best_j = float("inf"), None
        for j in range(len(pred_mm)):
            if j in used:
                continue
            d = np.sqrt((tx - pred_mm[j,0])**2 +
                        (ty - pred_mm[j,1])**2 +
                        (tz - pred_mm[j,2])**2)
            if d < best_dist:
                best_dist, best_j = d, j

        if best_j is not None:
            used.add(best_j)
            px, py, pz = pred_mm[best_j]
            ex = abs(tx - px); ey = abs(ty - py); ez = abs(tz - pz)
            matches.append({
                "true": (tx, ty, tz), "pred": (px, py, pz),
                "err_x": ex, "err_y": ey, "err_z": ez,
                "err_xy":  np.sqrt(ex**2 + ey**2),
                "err_xyz": np.sqrt(ex**2 + ey**2 + ez**2),
            })
        else:
            matches.append({"true": (tx, ty, tz), "pred": None})

    return matches


# =============================================================================
# FULL TEST SET EVALUATION
# =============================================================================

def evaluate_test_set(model, test_csv_paths, device):
    test_ds = ComptonEventDataset(test_csv_paths)
    rows    = []

    print(f"\n  Running inference on {len(test_csv_paths)} test scenes ...")
    print(f"  {'─'*63}")

    for i in range(len(test_ds)):
        sample   = test_ds[i]
        filename = os.path.basename(test_csv_paths[i])

        if sample["padding_mask"].all():
            print(f"  [SKIP] {filename}")
            continue

        pred_mm, pred_conf, true_mm, heatmap, pred_n = predict_scene(model, sample, device)
        matches = match_sources(true_mm, pred_mm)

        for j, m in enumerate(matches):
            row = {"filename": filename,
                   "n_sources_true": int(sample["n_sources"].item()),
                   "n_sources_pred": pred_n}
            tx, ty, tz = m["true"]
            row.update({"true_x": round(tx,3), "true_y": round(ty,3), "true_z": round(tz,3)})

            if m["pred"] is not None:
                px, py, pz = m["pred"]
                # FIX Issue 6: Log confidence score for each matched prediction
                row.update({
                    "pred_x": round(px,3), "pred_y": round(py,3), "pred_z": round(pz,3),
                    "pred_confidence": round(float(pred_conf[j]), 4),
                    "err_x": round(m["err_x"],4), "err_y": round(m["err_y"],4),
                    "err_z": round(m["err_z"],4), "err_xy": round(m["err_xy"],4),
                    "err_xyz": round(m["err_xyz"],4), "matched": True
                })
            else:
                row.update({
                    "pred_x": np.nan, "pred_y": np.nan, "pred_z": np.nan,
                    "pred_confidence": np.nan,
                    "err_x": np.nan, "err_y": np.nan, "err_z": np.nan,
                    "err_xy": np.nan, "err_xyz": np.nan, "matched": False
                })
            rows.append(row)

        if (i+1) % 10 == 0 or (i+1) == len(test_ds):
            matched_now = [m for m in matches if m["pred"] is not None]
            last_xyz = np.mean([m["err_xyz"] for m in matched_now]) if matched_now else float("nan")
            print(f"  {i+1:4d}/{len(test_ds)}  |  sources: {len(true_mm)}  |  last mean xyz={last_xyz:.2f} mm")

    return pd.DataFrame(rows)


# =============================================================================
# THRESHOLD SWEEP — find optimal confidence threshold post-training
# =============================================================================

def threshold_sweep(results_df, thresholds=None):
    """
    Sweeps confidence threshold from low to high and reports metrics.
    
    FIX: CONFIDENCE_THRESHOLD is a hyperparameter that needs post-training tuning.
    This function helps find the optimal operating point.
    
    Parameters
    ----------
    results_df : pd.DataFrame — evaluation results with pred_confidence column
    thresholds : np.ndarray — threshold values to try (default: 0.1 to 0.9 step 0.05)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    print(f"\n{'='*65}")
    print(f"  CONFIDENCE THRESHOLD SWEEP")
    print(f"{'='*65}")
    print(f"  {'Thr':>6}  {'Kept':>6}  {'Miss Rate':>10}  {'Mean XYZ':>10}  {'Median XYZ':>11}")
    print(f"  {'─'*65}")
    
    rows = []
    for thr in thresholds:
        filtered = results_df[results_df["pred_confidence"] >= thr]
        kept = len(filtered)
        miss_rate = 1 - kept / len(results_df) if len(results_df) > 0 else 0
        
        if kept > 0:
            mean_xyz = filtered["err_xyz"].mean()
            median_xyz = filtered["err_xyz"].median()
        else:
            mean_xyz = float('nan')
            median_xyz = float('nan')
        
        print(f"  {thr:>6.2f}  {kept:>6d}  {miss_rate*100:>9.1f}%  "
              f"{mean_xyz:>9.2f}mm  {median_xyz:>10.2f}mm")
        
        rows.append({
            "threshold": thr,
            "kept": kept,
            "miss_rate": miss_rate,
            "mean_xyz": mean_xyz,
            "median_xyz": median_xyz,
        })
    
    print(f"{'='*65}\n")
    
    # Find optimal threshold (minimum mean XYZ while keeping reasonable recall)
    df_rows = pd.DataFrame(rows)
    valid = df_rows[(df_rows["miss_rate"] < 0.5) & (df_rows["kept"] > 10)]
    
    if len(valid) > 0:
        best_idx = valid["mean_xyz"].argmin()
        best_thr = valid.iloc[best_idx]["threshold"]
        best_xyz = valid.iloc[best_idx]["mean_xyz"]
        print(f"  Recommended threshold: {best_thr:.2f} (mean XYZ={best_xyz:.2f}mm)")
        print(f"  This keeps {valid.iloc[best_idx]['kept']} sources with "
              f"{valid.iloc[best_idx]['miss_rate']*100:.1f}% miss rate\n")
    
    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary(df):
    matched = df[df["matched"] == True]
    missed  = df[df["matched"] == False]
    n_total = len(df)
    n_miss  = len(missed)

    print(f"\n{'='*65}")
    print(f"  TEST RESULTS  |  {n_total} sources  |  {n_miss} missed  "
          f"({100*n_miss/max(n_total,1):.1f}% miss rate)")
    print(f"{'='*65}")
    print(f"  {'Metric':<14}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'90th':>9}")
    print(f"  {'─'*56}")

    for col, label in [("err_x","X error  "), ("err_y","Y error  "),
                        ("err_z","Z error  "), ("err_xy","XY dist  "),
                        ("err_xyz","XYZ dist ")]:
        v = matched[col].dropna().values
        if len(v) == 0:
            continue
        print(f"  {label:<14}  {np.mean(v):>7.3f}mm"
              f"  {np.median(v):>7.3f}mm"
              f"  {np.std(v):>7.3f}mm"
              f"  {np.percentile(v,90):>8.3f}mm")

    # Breakdown by n_sources
    print(f"\n  By scene complexity:")
    for ns in sorted(df["n_sources_true"].unique()):
        sub = matched[matched["n_sources_true"] == ns]
        if len(sub) == 0:
            continue
        print(f"    {ns} source(s):  n={len(sub):4d}  "
              f"mean_xyz={sub['err_xyz'].mean():.2f} mm  "
              f"median_xyz={sub['err_xyz'].median():.2f} mm")

    print(f"{'='*65}\n")


# =============================================================================
# HISTOGRAMS
# =============================================================================

def plot_histograms(df, out_path):
    matched = df[df["matched"] == True]

    plt.rcParams.update({
        "font.family": "monospace",
        "axes.facecolor": "#0d1b2a", "figure.facecolor": "#070d1a",
        "axes.edgecolor": "#1a3a5c", "axes.labelcolor": "#cfe4ff",
        "xtick.color": "#cfe4ff", "ytick.color": "#cfe4ff",
        "text.color": "#cfe4ff", "grid.color": "#1a3a5c",
        "grid.linestyle": "--", "grid.alpha": 0.5,
    })

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Compton Camera — Error Distributions\n"
        f"{len(df)} sources total  |  {len(matched)} matched  |  {len(df)-len(matched)} missed",
        fontsize=13, color="#00e5ff", y=0.98,
    )
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.35)

    specs = [
        (gs[0, 0:2], "err_x",   "X error (mm)",         "#00bfa5"),
        (gs[0, 2:4], "err_y",   "Y error (mm)",         "#448aff"),
        (gs[0, 4:6], "err_z",   "Z error (mm)",         "#9c27b0"),
        (gs[1, 1:3], "err_xy",  "XY lateral dist (mm)", "#ff6d00"),
        (gs[1, 3:5], "err_xyz", "XYZ 3D dist (mm)",     "#ff5252"),
    ]

    for gs_pos, col, xlabel, color in specs:
        ax  = fig.add_subplot(gs_pos)
        v   = matched[col].dropna().values
        if len(v) == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue

        mean_v, med_v, p90_v = np.mean(v), np.median(v), np.percentile(v, 90)

        ax.hist(v, bins=config.HISTOGRAM_BINS, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(mean_v, color="#ffd740", lw=1.8, ls="--", label=f"Mean   {mean_v:.2f} mm")
        ax.axvline(med_v,  color="#ffffff", lw=1.8, ls="-",  label=f"Median {med_v:.2f} mm")
        ax.axvline(p90_v,  color="#ff5252", lw=1.2, ls=":",  label=f"90th   {p90_v:.2f} mm")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(xlabel.split(" (")[0], fontsize=11, color=color, pad=6)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.3, labelcolor="white")
        ax.grid(True)
        ax.text(0.98, 0.05, f"σ={np.std(v):.2f} mm", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#aaaaaa")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Histograms saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def evaluate(args):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    device = config.DEVICE

    print(f"\n{'='*65}")
    print(f"  EVALUATION  |  Device: {device}  |  Model: {args.model_path}")
    print(f"{'='*65}\n")

    model          = load_model(args.model_path, device)
    test_csv_paths = sorted(glob.glob(os.path.join(config.TEST_DIR, "*.csv")))

    if not test_csv_paths:
        raise FileNotFoundError(f"No CSVs in '{config.TEST_DIR}'.")

    print(f"  Found {len(test_csv_paths)} test files.")
    results_df = evaluate_test_set(model, test_csv_paths, device)

    print_summary(results_df)

    csv_out = os.path.join(config.RESULTS_DIR, "evaluation_results.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"  Results saved  → {csv_out}")

    plot_histograms(results_df, os.path.join(config.RESULTS_DIR, "error_histograms.png"))
    
    # FIX: Run threshold sweep to find optimal confidence threshold
    threshold_sweep(results_df)
    
    print(f"  Evaluation complete.\n")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-path", type=str,
                   default=os.path.join(config.MODEL_DIR, "best_model.pth"))
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
