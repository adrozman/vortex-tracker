#!/usr/bin/env python
"""
debug_vortex_tracks.py - Standalone diagnostic tool for inspecting and debugging tip vortex tracks.

Usage examples:
  # 1. Print summary table of all tracks (length, coordinates, Q_max, timestamps, and outlier flags):
  python debug_vortex_tracks.py --h5 velocity_and_vortex_data.h5

  # 2. Inspect a specific track by name, number, or plot label (e.g. track_03, 3, or "Vortex 3")
  #    and generate a multi-panel zoomed contour figure of all timesteps in that track:
  python debug_vortex_tracks.py --h5 velocity_and_vortex_data.h5 --track 3

  # 3. Automatically inspect all flagged secondary / outlier tracks at once:
  python debug_vortex_tracks.py --h5 velocity_and_vortex_data.h5 --all_outliers

  # 4. Inspect full-field Q-criterion for a specific timestep (e.g. step 22117 or 64165):
  python debug_vortex_tracks.py --h5 velocity_and_vortex_data.h5 --step 22117
"""

import os
import re
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def resolve_track_key(grp, user_input):
    """Resolve track name from user input (e.g. 'track_03', '3', 'Vortex 4', etc.)."""
    user_str = str(user_input).strip()
    sorted_keys = sorted(grp.keys())

    # Exact match
    if user_str in grp:
        return user_str

    # Numeric match (e.g. '3' -> 3rd track or track_03)
    m = re.search(r'\d+', user_str)
    if m:
        num = int(m.group())
        # Try zero-padded track_XX
        cand1 = f"track_{num:02d}"
        if cand1 in grp:
            return cand1
        cand2 = f"track_{num}"
        if cand2 in grp:
            return cand2
        # Try 1-based index from sorted list (Vortex 1, Vortex 2, etc.)
        if 1 <= num <= len(sorted_keys):
            return sorted_keys[num - 1]

    raise KeyError(f"Could not resolve track '{user_input}'. Available tracks: {sorted_keys}")

def summarize_tracks(hf):
    print("=" * 95)
    print(f"{'Plot Label':<12} {'H5 Track':<12} {'Pts':<5} {'Timestep Range':<22} {'X/D Range':<20} {'Mean Q_max':<12} {'Status'}")
    print("=" * 95)
    
    grp = hf['vortices']
    track_names = sorted(grp.keys())
    
    # Calculate baseline Q_max across primary vortices (length >= 5)
    long_tracks = [tr for tr in track_names if len(grp[tr]['time']) >= 5]
    baseline_q = []
    for tr in long_tracks:
        baseline_q.extend(grp[tr]['q_max'][:])
    median_primary_q = np.median(baseline_q) if len(baseline_q) > 0 else 1e7

    outliers = []
    for idx, tr in enumerate(track_names):
        gtr = grp[tr]
        t = gtr['time'][:]
        x = gtr['X_over_D'][:]
        z = gtr['Z_over_D'][:]
        q = gtr['q_max'][:] if 'q_max' in gtr else np.zeros_like(x)
        
        label = f"Vortex {idx+1}"
        t_str = f"{int(t[0])} -> {int(t[-1])}"
        x_str = f"[{x.min():.3f}, {x.max():.3f}]"
        q_mean_str = f"{np.mean(q):.2e}"
        
        flags = []
        if np.mean(q) < 0.3 * median_primary_q:
            flags.append("Weak Q (Secondary/Wake)")
        if len(t) < 5:
            flags.append(f"Short ({len(t)} pts)")
        
        status = ", ".join(flags) if flags else "Primary Vortex"
        if flags:
            outliers.append(tr)

        print(f"{label:<12} {tr:<12} {len(t):<5} {t_str:<22} {x_str:<20} {q_mean_str:<12} {status}")
    
    print("=" * 95)
    if outliers:
        print(f"Flagged candidate offset/secondary tracks: {', '.join(outliers)}")
    print("=" * 95)
    return outliers

def inspect_track(hf, track_input, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    grp = hf['vortices']
    track_name = resolve_track_key(grp, track_input)
    sorted_keys = sorted(grp.keys())
    vortex_label = f"Vortex {sorted_keys.index(track_name) + 1}"
            
    gtr = grp[track_name]
    t_arr = gtr['time'][:]
    x_arr = gtr['X_over_D'][:]
    z_arr = gtr['Z_over_D'][:]
    q_arr = gtr['q_max'][:] if 'q_max' in gtr else np.zeros_like(x_arr)
    a_arr = gtr['semi_major'][:] if 'semi_major' in gtr else np.zeros_like(x_arr)
    b_arr = gtr['semi_minor'][:] if 'semi_minor' in gtr else np.zeros_like(x_arr)
    ang_arr = gtr['angle_deg'][:] if 'angle_deg' in gtr else np.zeros_like(x_arr)

    print(f"\n==========================================================================================")
    print(f"Detailed Breakdown: {vortex_label} ({track_name}) - {len(t_arr)} Timesteps")
    print(f"==========================================================================================")
    print(f"{'Index':<6} {'Timestep':<10} {'X/D':<10} {'Z/D':<10} {'Q_max [s^-2]':<14} {'semi-major':<12} {'semi-minor':<12} {'Angle(deg)':<10}")
    print("-" * 88)
    for i in range(len(t_arr)):
        print(f"{i:<6} {int(t_arr[i]):<10} {x_arr[i]:<10.4f} {z_arr[i]:<10.4f} {q_arr[i]:<14.2e} {a_arr[i]:<12.4f} {b_arr[i]:<12.4f} {ang_arr[i]:<10.1f}")
    print("-" * 88)

    if 'q_criterion/Q_snapshots' not in hf:
        print("Note: Q_snapshots dataset not found in HDF5 file. Skipping image generation.")
        return

    X_rot = hf['grid/X_over_D'][:]
    Z_rot = hf['grid/Z_over_D'][:]
    timesteps = hf['time/timesteps'][:]
    Q_snaps = hf['q_criterion/Q_snapshots']

    n_pts = len(t_arr)
    fig, axes = plt.subplots(n_pts, 1, figsize=(10, 3.3 * n_pts), squeeze=False)
    
    tip_x = np.linspace(-0.35, 0.10, 200)
    tip_z = -0.33 * tip_x - 0.005

    for i in range(n_pts):
        ax = axes[i, 0]
        step_val = t_arr[i]
        t_idx = np.where(timesteps == step_val)[0][0]
        q_grid = Q_snaps[t_idx]

        xc = x_arr[i]
        zc = z_arr[i]

        pad_x = 0.08
        pad_z = 0.06
        ax.set_xlim(xc - pad_x, xc + pad_x)
        ax.set_ylim(zc - pad_z, zc + pad_z)

        im = ax.pcolormesh(X_rot, Z_rot, q_grid, cmap='RdBu_r', vmin=-1e7, vmax=2e7, shading='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(r"$Q\ [\mathrm{s}^{-2}]$", fontsize=10)

        ax.plot(tip_x, tip_z, 'k--', linewidth=1.8, label='Rotor tip path')

        # Highlight THIS detection
        ax.plot(xc, zc, 'go', markersize=8, markeredgecolor='black', label=f"{vortex_label} Core")
        if a_arr[i] > 0 and b_arr[i] > 0:
            e = Ellipse((xc, zc), width=2*a_arr[i], height=2*b_arr[i], angle=ang_arr[i],
                        facecolor='none', edgecolor='lime', linestyle='-', linewidth=2.0)
            ax.add_patch(e)

        # Plot other tracks at this timestep for spatial context
        for other_name in grp.keys():
            if other_name == track_name:
                continue
            other_gtr = grp[other_name]
            other_t = other_gtr['time'][:]
            if step_val in other_t:
                other_idx = np.where(other_t == step_val)[0][0]
                ox = other_gtr['X_over_D'][other_idx]
                oz = other_gtr['Z_over_D'][other_idx]
                ax.plot(ox, oz, 'mo', markersize=6)
                other_lbl = f"Vortex {sorted_keys.index(other_name) + 1}"
                ax.text(ox, oz + 0.008, other_lbl, color='magenta', fontsize=9, weight='bold', ha='center')

        ax.set_title(f"{vortex_label} ({track_name}) @ Step {int(step_val)} (t_index={t_idx}): (X/D={xc:.4f}, Z/D={zc:.4f})  |  Q_max = {q_arr[i]:.2e} s^-2",
                     fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(r"$X/D$", fontsize=11)
        ax.set_ylabel(r"$Z/D$", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.subplots_adjust(hspace=0.35, left=0.10, right=0.92, top=0.96, bottom=0.04)
    out_file = os.path.join(output_dir, f"debug_{track_name}.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Generated diagnostic figure: {out_file}")

def inspect_timestep(hf, step_val, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    timesteps = hf['time/timesteps'][:]
    if step_val not in timesteps:
        raise ValueError(f"Timestep {step_val} not in dataset. Range: [{timesteps[0]}, {timesteps[-1]}]")
    
    t_idx = np.where(timesteps == step_val)[0][0]
    X_rot = hf['grid/X_over_D'][:]
    Z_rot = hf['grid/Z_over_D'][:]
    q_grid = hf['q_criterion/Q_snapshots'][t_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(X_rot, Z_rot, q_grid, cmap='RdBu_r', vmin=-1e7, vmax=2e7, shading='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$Q\ [\mathrm{s}^{-2}]$", fontsize=12)

    tip_x = np.linspace(-0.35, 0.10, 200)
    tip_z = -0.33 * tip_x - 0.005
    ax.plot(tip_x, tip_z, 'k--', linewidth=2.0, label='Rotor tip path')

    grp = hf['vortices']
    sorted_keys = sorted(grp.keys())
    print(f"\nDetections at Step {int(step_val)} (t_index={t_idx}):")
    for tr_name in sorted_keys:
        gtr = grp[tr_name]
        t = gtr['time'][:]
        if step_val in t:
            idx = np.where(t == step_val)[0][0]
            xp = gtr['X_over_D'][idx]
            zp = gtr['Z_over_D'][idx]
            qp = gtr['q_max'][idx] if 'q_max' in gtr else 0.0
            lbl = f"Vortex {sorted_keys.index(tr_name) + 1}"
            ax.plot(xp, zp, 'o', markersize=8)
            ax.text(xp, zp + 0.008, f"{lbl}\n({tr_name})\nQ={qp:.1e}", fontsize=9, weight='bold', ha='center')
            print(f"  {lbl} ({tr_name}): X/D={xp:.4f}, Z/D={zp:.4f}, Q_max={qp:.2e}")

    ax.set_xlim(-0.35, 0.10)
    ax.set_ylim(-0.02, 0.32)
    ax.set_xlabel(r"$X/D$", fontsize=14)
    ax.set_ylabel(r"$Z/D$", fontsize=14)
    ax.set_title(f"Full Field Q-criterion Snapshot @ Step {int(step_val)}", fontsize=15, pad=10)
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"debug_step_{int(step_val)}.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Generated timestep figure: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Diagnostic utility for tip vortex tracks in HDF5.")
    parser.add_argument("--h5", type=str, default="velocity_and_vortex_data.h5", help="Path to HDF5 file")
    parser.add_argument("--track", type=str, default=None, help="Track name or number (e.g., 'track_03', '3', or 'Vortex 3')")
    parser.add_argument("--all_outliers", action="store_true", help="Generate debug plots for all candidate outlier tracks")
    parser.add_argument("--step", type=float, default=None, help="Specific timestep to inspect (e.g., 22117)")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for debug plots")

    args = parser.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(f"HDF5 file not found: {args.h5}")

    with h5py.File(args.h5, 'r') as hf:
        if args.step is not None:
            inspect_timestep(hf, args.step, output_dir=args.outdir)
        elif args.all_outliers:
            outliers = summarize_tracks(hf)
            for out_tr in outliers:
                print(f"\nProcessing outlier: {out_tr}")
                inspect_track(hf, out_tr, output_dir=args.outdir)
        elif args.track is not None:
            inspect_track(hf, args.track, output_dir=args.outdir)
        else:
            outliers = summarize_tracks(hf)
            if outliers:
                print("\nTip: To inspect any candidate offset track in detail, run:")
                print(f"  python debug_vortex_tracks.py --h5 {args.h5} --track {outliers[0]}")

if __name__ == "__main__":
    main()

