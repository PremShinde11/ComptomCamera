# inspect_scene.py — Single scene visual inspection
# DFG Project — University of Siegen
#
# Fixes vs previous version:
#   1. Reads and plots ALL ground-truth sources in the scene (not just first).
#   2. Shows ALL model predictions (up to MAX_SOURCES).
#   3. Draws matching lines between each true source and its nearest prediction.
#   4. Error bar chart shows per-source errors with source index labels.
#   5. Metrics panel lists every source separately.
#
# Usage:
#   python inspect_scene.py --csv data/test/SIM_events_source_0000.csv --save

import os
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend by default
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

import config
from model import ComptonSourceLocaliser
from train import ComptonEventDataset, denormalise_coordinate, MAX_SOURCES
from evaluate import load_model, predict_scene, match_sources, extract_heatmap_peaks

# ── Visual style ──────────────────────────────────────────────────────────────
HEATMAP_CMAP = LinearSegmentedColormap.from_list("ch", [
    "#070d1a","#0d1b35","#003366","#0066cc","#00bfa5","#ffd740","#ff6d00","#ff0000"
])
COLOR_TRUE   = "#ff5252"   # red   — ground truth
COLOR_PRED   = "#00e5ff"   # cyan  — prediction
COLOR_GRID   = "#1a3a5c"
COLOR_TEXT   = "#cfe4ff"

# Distinct colours for multiple sources (up to 5)
SOURCE_COLORS = ["#ff5252", "#ff9800", "#ffd740", "#69f0ae", "#ea80fc"]
PRED_COLORS   = ["#00e5ff", "#80d8ff", "#b9f6ca", "#ffe57f", "#ff80ab"]


def dark_ax(ax):
    ax.set_facecolor("#0d1b2a")
    ax.tick_params(colors=COLOR_TEXT, labelsize=9)
    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.yaxis.label.set_color(COLOR_TEXT)
    ax.title.set_color(COLOR_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(COLOR_GRID)
    ax.grid(True, color=COLOR_GRID, ls="--", alpha=0.5, lw=0.8)


# =============================================================================
# PANEL 1 — XY Heatmap with all sources
# =============================================================================

def plot_heatmap(ax, heatmap, true_sources, pred_sources):
    x0, x1 = config.HEATMAP_X_RANGE
    y0, y1 = config.HEATMAP_Y_RANGE

    # Scale heatmap to its own max for visibility even when undertrained
    vmax = max(float(heatmap.max()), 1e-4)
    ax.imshow(heatmap, origin="lower", extent=[x0,x1,y0,y1],
              cmap=HEATMAP_CMAP, vmin=0, vmax=vmax, aspect="equal")

    if vmax < 0.1:
        ax.text(0.5, 0.02, f"(peak={vmax:.4f} — needs more training)",
                transform=ax.transAxes, ha="center", fontsize=7,
                color="#ff6d00", style="italic")

    # Plot all true sources
    for i, (tx, ty, tz) in enumerate(true_sources):
        ax.scatter(tx, ty, c=SOURCE_COLORS[i % len(SOURCE_COLORS)],
                   s=160, marker="x", lw=2.5, zorder=5, label=f"True src {i}")

    # Plot all predictions
    for i, (px, py, pz) in enumerate(pred_sources):
        ax.scatter(px, py, c=PRED_COLORS[i % len(PRED_COLORS)],
                   s=160, marker="+", lw=2.5, zorder=5, label=f"Pred {i}")

    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Predicted XY Heatmap", fontsize=11, pad=6)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.3, labelcolor="white")
    dark_ax(ax)


# =============================================================================
# PANEL 2 — XY projection with matching lines
# =============================================================================

def plot_xy_projection(ax, matches):
    for i, m in enumerate(matches):
        tx, ty, tz = m["true"]
        c_true = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        ax.scatter(tx, ty, c=c_true, s=200, marker="X", zorder=5,
                   label=f"True {i}  ({tx:.0f},{ty:.0f})")

        if m["pred"] is not None:
            px, py, pz = m["pred"]
            c_pred = PRED_COLORS[i % len(PRED_COLORS)]
            ax.scatter(px, py, c=c_pred, s=200, marker="P", zorder=5,
                       label=f"Pred {i}  ({px:.0f},{py:.0f})")
            ax.plot([tx,px],[ty,py], color="#ffffff", lw=1.0, ls="--", alpha=0.6)
        else:
            ax.text(tx+3, ty+3, "MISSED", color="#ff5252", fontsize=7)

    ax.set_xlim(config.SOURCE_X_RANGE); ax.set_ylim(config.SOURCE_Y_RANGE)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("XY Projection — all sources", fontsize=11, pad=6)
    ax.legend(fontsize=7, loc="best", framealpha=0.3, labelcolor="white")
    dark_ax(ax)


# =============================================================================
# PANEL 3 — Depth (XZ) projection
# =============================================================================

def plot_depth_projection(ax, matches, horiz="x", vert="z"):
    idx = {"x": 0, "y": 1, "z": 2}
    hi = idx[horiz]; vi = idx[vert]

    ax.axhline(config.SCATTER_PLANE_Z, color="#334466", lw=1, alpha=0.6, label="Scatter plane")
    ax.axhline(config.ABSORB_PLANE_Z,  color="#223355", lw=1, alpha=0.6, label="Absorb plane")
    ax.axhspan(*config.SOURCE_Z_RANGE, alpha=0.08, color="#00bfa5", label="Source z range")

    for i, m in enumerate(matches):
        th = m["true"][hi]; tv = m["true"][vi]
        ax.scatter(th, tv, c=SOURCE_COLORS[i % len(SOURCE_COLORS)],
                   s=200, marker="X", zorder=5, label=f"True {i}")
        if m["pred"] is not None:
            ph = m["pred"][hi]; pv = m["pred"][vi]
            ax.scatter(ph, pv, c=PRED_COLORS[i % len(PRED_COLORS)],
                       s=200, marker="P", zorder=5, label=f"Pred {i}")
            ax.plot([th,ph],[tv,pv], color="#ffffff", lw=1.0, ls="--", alpha=0.6)

    x_rng = config.SOURCE_X_RANGE if horiz == "x" else config.SOURCE_Y_RANGE
    z_rng = (config.SOURCE_Z_RANGE[0]-15, config.SOURCE_Z_RANGE[1]+15)
    ax.set_xlim(x_rng); ax.set_ylim(z_rng)
    ax.set_xlabel(f"{horiz} (mm)"); ax.set_ylabel(f"{vert} (mm)")
    ax.set_title(f"{horiz.upper()}{vert.upper()} Projection", fontsize=11, pad=6)
    ax.legend(fontsize=7, loc="best", framealpha=0.3, labelcolor="white")
    dark_ax(ax)


# =============================================================================
# PANEL 4 — Per-source error bars
# =============================================================================

def plot_error_bars(ax, matches):
    labels, values, colors = [], [], []
    for i, m in enumerate(matches):
        if m["pred"] is None:
            labels.append(f"src {i} MISSED"); values.append(0); colors.append("#555555")
            continue
        labels.append(f"src {i} XYZ"); values.append(m["err_xyz"]); colors.append(SOURCE_COLORS[i % len(SOURCE_COLORS)])
        labels.append(f"src {i} XY");  values.append(m["err_xy"]);  colors.append(PRED_COLORS[i % len(PRED_COLORS)])
        labels.append(f"src {i} Z");   values.append(m["err_z"]);   colors.append("#9c27b0")

    if not values:
        ax.text(0.5, 0.5, "no matched sources", transform=ax.transAxes, ha="center")
        dark_ax(ax); return

    bars = ax.barh(labels, values, color=colors, alpha=0.8, height=0.55, edgecolor="none")
    for b, v in zip(bars, values):
        ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2,
                f"{v:.2f} mm", va="center", ha="left", fontsize=8, color=COLOR_TEXT)

    ax.set_xlabel("Error (mm)")
    ax.set_xlim(0, max(values)*1.35 if values else 1)
    ax.set_title("Per-source Errors", fontsize=11, pad=6)
    dark_ax(ax)


# =============================================================================
# PANEL 5 — Metrics text
# =============================================================================

def plot_metrics_text(ax, matches, filename, n_type1, true_n, pred_n, confidences):
    ax.axis("off")
    ax.set_facecolor("#0d1b2a")
    for sp in ax.spines.values():
        sp.set_edgecolor(COLOR_GRID)

    lines = [
        (f"FILE", os.path.basename(filename), "#00e5ff"),
        (f"Type-1 events", str(n_type1),      "#69f0ae"),
        ("", "", ""),
        # FIX Issue 7: Show count prediction panel
        (f"── COUNT PREDICTION ──", "", "#ffd740"),
        (f"  True N",        str(true_n),  "#69f0ae"),
        (f"  Predicted N",   str(pred_n),  "#ff5252" if pred_n != true_n else "#69f0ae"),
    ]
    
    # Add confidence scores with used/discarded markers (FIX inspection display)
    if len(confidences) > 0:
        lines.append((f"  Confidences (used | discarded):", "", "#00bfa5"))
        for i, conf_val in enumerate(confidences):
            if i < pred_n:
                marker = "✓"
                color = "#69f0ae"  # green for used
            else:
                marker = "✗"
                color = "#ff5252"  # red for discarded
            lines.append((f"    Slot {i}: {conf_val:.2f} {marker}", "", color))
    
    lines.append(("", "", ""))

    for i, m in enumerate(matches):
        tx, ty, tz = m["true"]
        tc = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        lines.append((f"── Source {i} ──", "", "#aaaaaa"))
        lines.append((f"  true",  f"({tx:.1f}, {ty:.1f}, {tz:.1f})", tc))
        if m["pred"] is not None:
            px, py, pz = m["pred"]
            pc = PRED_COLORS[i % len(PRED_COLORS)]
            conf_val = confidences[i] if i < len(confidences) else None
            conf_suffix = f" (conf={conf_val:.2f})" if conf_val is not None else ""
            lines.append((f"  pred{conf_suffix}",  f"({px:.1f}, {py:.1f}, {pz:.1f})", pc))
            lines.append((f"  XYZ err", f"{m['err_xyz']:.2f} mm",  "#ff5252"))
            lines.append((f"  XY  err", f"{m['err_xy']:.2f} mm",   "#ff6d00"))
            lines.append((f"  Z   err", f"{m['err_z']:.2f} mm",    "#9c27b0"))
        else:
            lines.append(("  pred", "MISSED", "#ff5252"))
        lines.append(("","",""))

    y = 0.97; dy = 0.044
    for label, value, color in lines:
        if label == "" and value == "":
            y -= dy * 0.4; continue
        if value == "":
            ax.text(0.03, y, label, transform=ax.transAxes,
                    fontsize=8.5, color=color, fontfamily="monospace",
                    fontweight="bold", va="top")
        else:
            ax.text(0.03, y, f"{label:<16}", transform=ax.transAxes,
                    fontsize=8, color="#aaaaaa", fontfamily="monospace", va="top")
            ax.text(0.55, y, value, transform=ax.transAxes,
                    fontsize=8, color=color, fontfamily="monospace",
                    fontweight="bold", va="top")
        y -= dy


# =============================================================================
# MAIN INSPECTION FUNCTION
# =============================================================================

def inspect_scene(csv_path, model_path, save, out_dir):
    device  = config.DEVICE
    model   = load_model(model_path, device)
    dataset = ComptonEventDataset([csv_path])
    sample  = dataset[0]

    if sample["padding_mask"].all():
        raise ValueError(f"No valid Type-1 events in '{csv_path}'.")

    # Ground truth: ALL sources
    true_mm = sample["sources_mm"].numpy()    # (N_true, 3)

    # Predictions: uses count head + confidence filtering
    pred_mm, pred_conf, _, heatmap, pred_n = predict_scene(model, sample, device)

    # Match all true to predictions
    matches = match_sources(true_mm, pred_mm)

    # For heatmap panel: extract peaks as predicted XY from heatmap
    heatmap_peaks = extract_heatmap_peaks(heatmap)

    # Scene info
    raw_df      = pd.read_csv(csv_path)
    n_type1     = int((raw_df["event_type"] == config.SIGNAL_EVENT_TYPE).sum())
    true_n      = len(true_mm)

    # Console output with count prediction (FIX Issue 7)
    print(f"\n{'='*60}"); print(f"  {os.path.basename(csv_path)}")
    print(f"  True sources: {true_n}  |  Predicted: {pred_n}  |  Type-1 events: {n_type1}")
    if len(pred_conf) > 0:
        print(f"  Confidences: [{', '.join([f'{c:.2f}' for c in pred_conf])}]")
    for i, m in enumerate(matches):
        tx, ty, tz = m["true"]
        if m["pred"] is not None:
            px, py, pz = m["pred"]
            conf_str = f" (conf={pred_conf[i]:.2f})" if i < len(pred_conf) else ""
            print(f"  Src {i}: True({tx:.1f},{ty:.1f},{tz:.1f}) → "
                  f"Pred({px:.1f},{py:.1f},{pz:.1f}){conf_str}  xyz={m['err_xyz']:.2f}mm")
        else:
            print(f"  Src {i}: True({tx:.1f},{ty:.1f},{tz:.1f}) → MISSED")
    print(f"{'='*60}\n")

    # Build figure
    plt.rcParams.update({"font.family":"monospace","figure.facecolor":"#070d1a"})
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor("#070d1a")
    n_pred = len(pred_mm)
    fig.suptitle(
        f"Compton Camera — Scene Inspection\n"
        f"{os.path.basename(csv_path)}  |  True N={true_n}  |  Pred N={pred_n}",
        fontsize=13, color="#00e5ff", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    # Coordinate head predictions as (x,y) list for heatmap overlay
    coord_pred_xy = [(pred_mm[j,0], pred_mm[j,1], pred_mm[j,2])
                     for j in range(min(len(pred_mm), MAX_SOURCES))]

    ax_heat = fig.add_subplot(gs[0, 0])
    plot_heatmap(ax_heat, heatmap,
                 [(tx,ty,tz) for tx,ty,tz in true_mm],
                 coord_pred_xy)

    ax_xy = fig.add_subplot(gs[0, 1])
    plot_xy_projection(ax_xy, matches)

    ax_txt = fig.add_subplot(gs[0, 2])
    plot_metrics_text(ax_txt, matches, csv_path, n_type1, true_n, pred_n, pred_conf.tolist() if len(pred_conf) > 0 else [])

    ax_xz = fig.add_subplot(gs[1, 0])
    plot_depth_projection(ax_xz, matches, "x", "z")

    ax_yz = fig.add_subplot(gs[1, 1])
    plot_depth_projection(ax_yz, matches, "y", "z")

    ax_bars = fig.add_subplot(gs[1, 2])
    plot_error_bars(ax_bars, matches)

    # Legend
    legend_els = (
        [mpatches.Patch(color=SOURCE_COLORS[i], label=f"True source {i}") for i in range(true_n)] +
        [mpatches.Patch(color=PRED_COLORS[i],   label=f"Prediction {i}")  for i in range(true_n)]
    )
    fig.legend(handles=legend_els, loc="lower center", ncol=min(true_n*2,6),
               fontsize=9, framealpha=0.3, labelcolor="white",
               facecolor="#0d1b2a", edgecolor=COLOR_GRID, bbox_to_anchor=(0.5, 0.0))

    # fig = plt.figure(figsize=(20, 12))
    # fig.patch.set_facecolor("#070d1a")

# ADD THIS LINE
    fig.set_constrained_layout(True)

    # Always save the figure - interactive display disabled for compatibility
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out  = os.path.join(out_dir, f"inspect_{base}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Figure saved → {out}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",        type=str, default=os.path.join(config.TEST_DIR, "SIM_events_source_0017.csv"))
    p.add_argument("--model-path", type=str, default=os.path.join(config.MODEL_DIR, "best_model.pth"))
    p.add_argument("--save",       action="store_true", default=False)
    p.add_argument("--output-dir", type=str, default=config.RESULTS_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_scene(args.csv, args.model_path, args.save, args.output_dir)
