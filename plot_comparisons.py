import os
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import re
import time

def safe_savefig(target, filepath, dpi=200, **kwargs):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    for attempt in range(5):
        try:
            target.savefig(filepath, dpi=dpi, **kwargs)
            return
        except OSError as e:
            if attempt < 4:
                time.sleep(0.15 * (2 ** attempt))
            else:
                raise e

def plot_all_standalone_figures(h5_path, output_dir="plots", exclude_tracks=None, min_points=2,
                                xlim=(-0.75, 0.75), ylim=(-0.75, 0.75), phase_step=22207, period=288):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading HDF5 data from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as hf:
        X_rot = hf['grid/X_over_D'][:]
        Z_rot = hf['grid/Z_over_D'][:]
        timesteps = hf['time/timesteps'][:]
        
        U_mean_norm = hf['mean/U_mean_norm'][:]
        W_mean_norm = hf['mean/W_mean_norm'][:]
        
        u_var_norm = hf['variance/u_variance_norm'][:]
        w_var_norm = hf['variance/w_variance_norm'][:]
        
        has_q = 'q_criterion' in hf
        if has_q:
            Q_mean = hf['q_criterion/Q_mean'][:]
            Q_snap0 = hf['q_criterion/Q_snapshots'][0]
        else:
            Q_mean = None
            Q_snap0 = None
            
        u_prime_norm0 = hf['unsteady/u_prime_norm'][0]
        w_prime_norm0 = hf['unsteady/w_prime_norm'][0]
        
        # Extract vortex tracks
        tracks = []
        if 'vortices' in hf:
            grp_vort = hf['vortices']
            for tr_name in sorted(grp_vort.keys()):
                gtr = grp_vort[tr_name]
                tracks.append({
                    'name': tr_name,
                    'time': gtr['time'][:],
                    'X_over_D': gtr['X_over_D'][:],
                    'Z_over_D': gtr['Z_over_D'][:],
                    'mesh_x': gtr['mesh_x'][:] if 'mesh_x' in gtr else None,
                    'mesh_z': gtr['mesh_z'][:] if 'mesh_z' in gtr else None,
                    'semi_major': gtr['semi_major'][:],
                    'semi_minor': gtr['semi_minor'][:],
                    'angle_deg': gtr['angle_deg'][:],
                    'bnd_pts_step0': gtr['bnd_pts_step0'][:] if 'bnd_pts_step0' in gtr else None
                })

        x_mesh = hf['grid/x_mesh'][:] if 'grid/x_mesh' in hf else None
        z_mesh = hf['grid/z_mesh'][:] if 'grid/z_mesh' in hf else None
        U_snap0 = (hf['mean/U_mean'][:] + hf['unsteady/u_prime'][0]) if 'unsteady/u_prime' in hf else None
        W_snap0 = (hf['mean/W_mean'][:] + hf['unsteady/w_prime'][0]) if 'unsteady/w_prime' in hf else None
        u_inf = hf.attrs.get('U_inf_nondim', 10.0 / 340.0)

    # Spatial domain limits (defaults to full extracted data bounds [-0.75, 0.75])
    if xlim is None:
        xlim = (-0.75, 0.75)
    else:
        xlim = tuple(xlim)
    if ylim is None:
        ylim = (-0.75, 0.75)
    else:
        ylim = tuple(ylim)

    # ----------------------------------------------------
    # 1. Mean U Velocity: <\bar{U}> / U_\infty
    # ----------------------------------------------------
    print("Generating mean_U_over_Uinf.png...")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(X_rot, Z_rot, U_mean_norm, cmap='Spectral_r', vmin=-1.0, vmax=3.0, shading='auto')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$X/D$", fontsize=16)
    ax.set_ylabel(r"$Z/D$", fontsize=16)
    ax.set_title(r"$\langle\bar{U}\rangle / U_\infty$", fontsize=18, pad=12)
    ax.tick_params(labelsize=14)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$\langle\bar{U}\rangle / U_\infty$", fontsize=16)
    plt.tight_layout()
    out1 = os.path.join(output_dir, "mean_U_over_Uinf.png")
    safe_savefig(plt, out1, dpi=200)
    plt.close()
    print(f"  Saved {out1}")

    # ----------------------------------------------------
    # 2. Mean W Velocity: <\bar{W}> / U_\infty
    # ----------------------------------------------------
    print("Generating mean_W_over_Uinf.png...")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(X_rot, Z_rot, W_mean_norm, cmap='Spectral_r', vmin=-3.0, vmax=3.0, shading='auto')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$X/D$", fontsize=16)
    ax.set_ylabel(r"$Z/D$", fontsize=16)
    ax.set_title(r"$\langle\bar{W}\rangle / U_\infty$", fontsize=18, pad=12)
    ax.tick_params(labelsize=14)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$\langle\bar{W}\rangle / U_\infty$", fontsize=16)
    plt.tight_layout()
    out2 = os.path.join(output_dir, "mean_W_over_Uinf.png")
    safe_savefig(plt, out2, dpi=200)
    plt.close()
    print(f"  Saved {out2}")

    # ----------------------------------------------------
    # 3. Unsteady U Variance: <u'^2> / U_\infty^2
    # ----------------------------------------------------
    print("Generating variance_u_prime.png...")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(X_rot, Z_rot, u_var_norm, cmap='Spectral_r', vmin=0.0, vmax=0.6, shading='auto')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$X/D$", fontsize=16)
    ax.set_ylabel(r"$Z/D$", fontsize=16)
    ax.set_title(r"$\langle \bar{u}'^2 \rangle / U_\infty^2$", fontsize=18, pad=12)
    ax.tick_params(labelsize=14)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$\langle \bar{u}'^2 \rangle / U_\infty^2$", fontsize=16)
    plt.tight_layout()
    out3 = os.path.join(output_dir, "variance_u_prime.png")
    safe_savefig(plt, out3, dpi=200)
    plt.close()
    print(f"  Saved {out3}")

    # ----------------------------------------------------
    # 4. Unsteady W Variance: <w'^2> / U_\infty^2
    # ----------------------------------------------------
    print("Generating variance_w_prime.png...")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(X_rot, Z_rot, w_var_norm, cmap='Spectral_r', vmin=0.0, vmax=0.6, shading='auto')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$X/D$", fontsize=16)
    ax.set_ylabel(r"$Z/D$", fontsize=16)
    ax.set_title(r"$\langle \bar{w}'^2 \rangle / U_\infty^2$", fontsize=18, pad=12)
    ax.tick_params(labelsize=14)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$\langle \bar{w}'^2 \rangle / U_\infty^2$", fontsize=16)
    plt.tight_layout()
    out4 = os.path.join(output_dir, "variance_w_prime.png")
    safe_savefig(plt, out4, dpi=200)
    plt.close()
    print(f"  Saved {out4}")

    # ----------------------------------------------------
    # 5. Vortex Trajectories (Matching experiment_vortex_trajectory.png)
    # ----------------------------------------------------
    print("Generating vortex_trajectories.png...")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    
    # Optional subtle background Q contour
    if Q_mean is not None:
        ax.pcolormesh(X_rot, Z_rot, Q_mean, cmap='RdBu_r', vmin=-5e6, vmax=5e6, shading='auto', alpha=0.35)

    # Filter tracks if user requested exclusions or min_points
    exclude_set = set()
    if exclude_tracks:
        for ex in exclude_tracks:
            ex_str = str(ex).strip()
            exclude_set.add(ex_str.lower())
            m = re.search(r'\d+', ex_str)
            if m:
                num = int(m.group())
                exclude_set.add(f"track_{num:02d}".lower())
                exclude_set.add(f"track_{num}".lower())
                exclude_set.add(f"vortex {num}".lower())
                exclude_set.add(str(num))

    # Plot individual vortex trajectories
    colors = ['#0000CC', '#0066FF', '#0099FF', '#33CC33', '#FF6600', '#CC0000', '#9900CC']
    for idx, tr in enumerate(tracks):
        v_lbl = f"Vortex {idx+1}"
        if tr['name'].lower() in exclude_set or v_lbl.lower() in exclude_set or str(idx+1) in exclude_set:
            print(f"  Excluding {v_lbl} ({tr['name']}) from trajectory plot.")
            continue
        if len(tr['time']) < min_points:
            print(f"  Filtering {v_lbl} ({tr['name']}): length {len(tr['time'])} < {min_points} points.")
            continue
        c = colors[idx % len(colors)]
        label_text = v_lbl
        ax.plot(tr['X_over_D'], tr['Z_over_D'], 'o-', color=c, linewidth=2.5, markersize=5.5, label=label_text)

    # Rotor tip path reference line spanning across the domain
    tip_x = np.linspace(xlim[0], xlim[1], 300)
    tip_z = -0.33 * tip_x - 0.005
    ax.plot(tip_x, tip_z, 'k-', linewidth=2.2, label='Rotor tip path')
    
    # Annotations along tip path (matching experimental figure)
    ax.plot([-0.20, -0.10, 0.0], [-0.33*(-0.20)-0.005, -0.33*(-0.10)-0.005, -0.005], 'r*', markersize=8)
    ax.annotate(r"$120^\circ$" + "\n0.81R", xy=(-0.20, -0.33*(-0.20)-0.005), xytext=(-0.22, -0.01),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), fontsize=11, ha='center')
    ax.annotate(r"$105^\circ$" + "\n0.72R", xy=(-0.10, -0.33*(-0.10)-0.005), xytext=(-0.11, -0.05),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), fontsize=11, ha='center')
    ax.annotate(r"$90^\circ$" + "\n0.70R", xy=(0.0, -0.005), xytext=(0.0, -0.08),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), fontsize=11, ha='center')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$X/D$", fontsize=16)
    ax.set_ylabel(r"$Z/D$", fontsize=16)
    ax.set_title("Extracted Tip Vortex Trajectories", fontsize=18, pad=12)
    ax.tick_params(labelsize=14)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    out5 = os.path.join(output_dir, "vortex_trajectories.png")
    safe_savefig(plt, out5, dpi=200)
    plt.close()
    print(f"  Saved {out5}")

    # ----------------------------------------------------
    # 6. Visual Verification: Q Contour with Fitted Ellipses & Centers
    # ----------------------------------------------------
    if Q_snap0 is not None:
        print("Generating vortex_core_verification.png...")
        fig, ax = plt.subplots(figsize=(9, 7.5))
        im = ax.pcolormesh(X_rot, Z_rot, Q_snap0, cmap='RdBu_r', vmin=-5e6, vmax=5e6, shading='auto')
        
        # Plot detected ellipses and centers for timestep 0
        vort_idx = 1
        for tr in tracks:
            if len(tr['time']) > 0 and tr['time'][0] == timesteps[0]:
                xc = tr['X_over_D'][0]
                zc = tr['Z_over_D'][0]
                major = tr['semi_major'][0]
                minor = tr['semi_minor'][0]
                angle = tr['angle_deg'][0]
                
                # Plot center point
                ax.plot(xc, zc, 'yo', markersize=8, markeredgecolor='black',
                        label='Calculated Core Center' if vort_idx==1 else "")
                
                # Plot FWHM boundary points if available
                if tr['bnd_pts_step0'] is not None:
                    ax.plot(tr['bnd_pts_step0'][:, 0], tr['bnd_pts_step0'][:, 1], 'm.', markersize=5,
                            label='FWHM Boundary Points' if vort_idx==1 else "")

                # Plot 2-sigma fitted ellipse
                ell = Ellipse((xc, zc), width=2*major, height=2*minor, angle=angle,
                              edgecolor='yellow', facecolor='none', linewidth=2.0, linestyle='--',
                              label='Fitted FWHM Ellipse' if vort_idx==1 else "")
                ax.add_patch(ell)
                
                ax.text(xc, zc + 0.02, f"Vortex {vort_idx}\n({xc:.3f}, {zc:.3f})",
                        color='yellow', fontsize=11, fontweight='bold', ha='center')
                vort_idx += 1

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$X/D$", fontsize=16)
        ax.set_ylabel(r"$Z/D$", fontsize=16)
        ax.set_title(f"Visual Verification: Vortex Peak Centers & FWHM Ellipses (Step {timesteps[0]:.0f})", fontsize=16, pad=12)
        ax.tick_params(labelsize=14)
        ax.legend(loc='upper left', fontsize=12)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(r"$Q\ [\mathrm{s}^{-2}]$", fontsize=16)
        plt.tight_layout()
        out6 = os.path.join(output_dir, "vortex_core_verification.png")
        safe_savefig(plt, out6, dpi=200)
        plt.close()
        print(f"  Saved {out6}")

    # ----------------------------------------------------
    # 7. Sample Instantaneous Unsteady Snapshots: u' and w'
    # ----------------------------------------------------
    step0_name = f"{timesteps[0]:.0f}"
    print(f"Generating instantaneous unsteady snapshots for step {step0_name}...")
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(X_rot, Z_rot, u_prime_norm0, cmap='RdBu_r', vmin=-1.0, vmax=1.0, shading='auto')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$X/D$", fontsize=16)
    ax.set_ylabel(r"$Z/D$", fontsize=16)
    ax.set_title(rf"Instantaneous $u'/U_\infty$ (Step {step0_name})", fontsize=18, pad=12)
    ax.tick_params(labelsize=14)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$u'/U_\infty$", fontsize=16)
    plt.tight_layout()
    out7 = os.path.join(output_dir, f"u_prime_step_{step0_name}.png")
    safe_savefig(plt, out7, dpi=200)
    plt.close()
    print(f"  Saved {out7}")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(X_rot, Z_rot, w_prime_norm0, cmap='RdBu_r', vmin=-1.0, vmax=1.0, shading='auto')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$X/D$", fontsize=16)
    ax.set_ylabel(r"$Z/D$", fontsize=16)
    ax.set_title(rf"Instantaneous $w'/U_\infty$ (Step {step0_name})", fontsize=18, pad=12)
    ax.tick_params(labelsize=14)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"$w'/U_\infty$", fontsize=16)
    plt.tight_layout()
    out8 = os.path.join(output_dir, f"w_prime_step_{step0_name}.png")
    safe_savefig(plt, out8, dpi=200)
    plt.close()
    print(f"  Saved {out8}")

    # ----------------------------------------------------
    # 8. 1D Radial Slices Verification Profiles (Q & Swirl Velocity)
    # ----------------------------------------------------
    if Q_snap0 is not None and x_mesh is not None and z_mesh is not None:
        from scipy.interpolate import RegularGridInterpolator
        print("Generating 1D radial slice verification profiles...")
        x_lin = x_mesh[:, 0]
        z_lin = z_mesh[0, :]
        interp_q = RegularGridInterpolator((x_lin, z_lin), Q_snap0, bounds_error=False, fill_value=0.0)
        interp_u = RegularGridInterpolator((x_lin, z_lin), U_snap0, bounds_error=False, fill_value=0.0) if U_snap0 is not None else None
        interp_w = RegularGridInterpolator((x_lin, z_lin), W_snap0, bounds_error=False, fill_value=0.0) if W_snap0 is not None else None

        valid_tracks_step0 = [tr for tr in tracks if len(tr['time']) > 0 and tr['time'][0] == timesteps[0] and tr['mesh_x'] is not None]
        n_vort = len(valid_tracks_step0)

        if n_vort > 0:
            fig, axes = plt.subplots(n_vort, 2, figsize=(12, 3.2 * n_vort))
            if n_vort == 1:
                axes = np.expand_dims(axes, axis=0)

            r_sample = np.linspace(-1.5, 1.5, 100) # inches
            r_D = r_sample / 24.0 # rotor diameter 24 in

            for idx, tr in enumerate(valid_tracks_step0):
                xc_m = tr['mesh_x'][0]
                zc_m = tr['mesh_z'][0]
                xc_r = tr['X_over_D'][0]
                zc_r = tr['Z_over_D'][0]

                pts_h = np.column_stack((xc_m + r_sample, np.full_like(r_sample, zc_m)))
                pts_v = np.column_stack((np.full_like(r_sample, xc_m), zc_m + r_sample))

                q_h = interp_q(pts_h)
                q_v = interp_q(pts_v)

                # Q Profile
                ax1 = axes[idx, 0]
                ax1.plot(r_D, q_h / 1e6, 'b-', label='Horizontal Slice (X)', linewidth=2)
                ax1.plot(r_D, q_v / 1e6, 'r--', label='Vertical Slice (Z)', linewidth=2)
                ax1.axvline(0, color='gray', linestyle=':', alpha=0.7)
                ax1.set_title(f"Vortex {idx+1} at X/D={xc_r:.3f}, Z/D={zc_r:.3f} — Q Profile", fontsize=12)
                ax1.set_xlabel("Radial Distance r/D", fontsize=11)
                ax1.set_ylabel(r"$Q\ (\times 10^6\ \mathrm{s}^{-2})$", fontsize=11)
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='upper right', fontsize=10)

                # Swirl Velocity Profile
                ax2 = axes[idx, 1]
                if interp_w is not None and interp_u is not None:
                    w_h = interp_w(pts_h)
                    w_c = interp_w(np.array([[xc_m, zc_m]]))[0]
                    v_swirl_h = w_h - w_c

                    u_v = interp_u(pts_v)
                    u_c = interp_u(np.array([[xc_m, zc_m]]))[0]
                    v_swirl_v = -(u_v - u_c)

                    ax2.plot(r_D, v_swirl_h / u_inf, 'b-', label=r'$V_\theta$ (Horiz)', linewidth=2)
                    ax2.plot(r_D, v_swirl_v / u_inf, 'r--', label=r'$V_\theta$ (Vert)', linewidth=2)
                    ax2.axvline(0, color='gray', linestyle=':', alpha=0.7)
                    ax2.axhline(0, color='gray', linestyle=':', alpha=0.7)

                    ax2.set_title(f"Vortex {idx+1} — Swirl Velocity Profile", fontsize=12)
                    ax2.set_xlabel("Radial Distance r/D", fontsize=11)
                    ax2.set_ylabel(r"$V_\theta / U_\infty$", fontsize=11)
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='upper right', fontsize=10)

            plt.tight_layout()
            out9 = os.path.join(output_dir, "vortex_1d_slices_verification.png")
            safe_savefig(plt, out9, dpi=200)
            plt.close()
            print(f"  Saved {out9}")

    # ----------------------------------------------------
    # 9. Phase-Averaged Q-Criterion at Agreed Rotor Phase
    # ----------------------------------------------------
    if has_q:
        with h5py.File(h5_path, 'r') as hf_local:
            if 'q_criterion/Q_snapshots' in hf_local:
                Q_snapshots = hf_local['q_criterion/Q_snapshots'][:]
            else:
                Q_snapshots = None
        
        if Q_snapshots is not None and len(Q_snapshots) > 0:
            target_phase = phase_step % period
            matching_indices = [i for i, t in enumerate(timesteps) if int(round(t)) % period == target_phase]

            if not matching_indices:
                phase_diffs = [min(abs((int(round(t)) % period) - target_phase), period - abs((int(round(t)) % period) - target_phase)) for t in timesteps]
                min_diff = min(phase_diffs)
                matching_indices = [i for i, diff in enumerate(phase_diffs) if diff == min_diff]
                print(f"Warning: Exact phase {target_phase} not found. Using closest available phase offset {min_diff} steps.")

            matching_steps = [int(round(timesteps[i])) for i in matching_indices]
            print(f"Generating phase_average_q.png for phase step {phase_step} (phase mod {period} = {target_phase})...")
            print(f"  Averaging {len(matching_indices)} snapshot(s) at timestep(s): {matching_steps}")

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if len(matching_indices) == 1:
                    Q_phase_avg = Q_snapshots[matching_indices[0]]
                    title_str = rf"Phase-Averaged $Q$-Criterion (Step {matching_steps[0]})"
                else:
                    Q_phase_avg = np.nanmean(Q_snapshots[matching_indices], axis=0)
                    title_str = rf"Phase-Averaged $Q$-Criterion ($N={len(matching_indices)}$ Revs, Phase {target_phase})"

            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.pcolormesh(X_rot, Z_rot, Q_phase_avg, cmap='RdBu_r', vmin=-5e6, vmax=5e6, shading='auto')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(r"$X/D$", fontsize=16)
            ax.set_ylabel(r"$Z/D$", fontsize=16)
            ax.set_title(title_str, fontsize=18, pad=12)
            ax.tick_params(labelsize=14)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label(r"$Q\ [\mathrm{s}^{-2}]$", fontsize=16)
            plt.tight_layout()
            out_phase_q = os.path.join(output_dir, "phase_average_q.png")
            safe_savefig(plt, out_phase_q, dpi=200)
            plt.close()
            print(f"  Saved {out_phase_q}")

            # If multiple matching revolutions, also save the single instantaneous snapshot at the latest revolution
            if len(matching_indices) > 1:
                latest_idx = matching_indices[-1]
                latest_step = matching_steps[-1]
                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.pcolormesh(X_rot, Z_rot, Q_snapshots[latest_idx], cmap='RdBu_r', vmin=-5e6, vmax=5e6, shading='auto')
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel(r"$X/D$", fontsize=16)
                ax.set_ylabel(r"$Z/D$", fontsize=16)
                ax.set_title(rf"Instantaneous $Q$-Criterion (Step {latest_step})", fontsize=18, pad=12)
                ax.tick_params(labelsize=14)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=14)
                cbar.set_label(r"$Q\ [\mathrm{s}^{-2}]$", fontsize=16)
                plt.tight_layout()
                out_single_q = os.path.join(output_dir, f"q_criterion_step_{latest_step}.png")
                safe_savefig(plt, out_single_q, dpi=200)
                plt.close()
                print(f"  Saved {out_single_q}")

    print("\nAll standalone figures generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate standalone publication-quality figures from HDF5 output.")
    parser.add_argument("--h5", type=str, default="velocity_and_vortex_data.h5", help="Path to velocity_and_vortex_data.h5")
    parser.add_argument("--outdir", type=str, default="plots", help="Directory to save generated PNG images")
    parser.add_argument("--exclude", nargs="*", default=None, help="List of tracks to exclude from trajectory plot (e.g. --exclude 3 or --exclude track_03 'Vortex 4')")
    parser.add_argument("--min_points", type=int, default=2, help="Minimum number of points required to plot a trajectory (default 2)")
    parser.add_argument("--xlim", nargs=2, type=float, default=[-0.75, 0.75], help="X limits for spatial plots (default: -0.75 0.75)")
    parser.add_argument("--ylim", nargs=2, type=float, default=[-0.75, 0.75], help="Z limits for spatial plots (default: -0.75 0.75)")
    parser.add_argument("--piv_limits", action="store_true", help="Use experimental PIV camera limits (-0.30 to 0.08, -0.10 to 0.40) instead of full domain")
    parser.add_argument("--phase_step", type=int, default=22207, help="Reference phase timestep to identify rotor phase angle (default 22207)")
    parser.add_argument("--period", type=int, default=288, help="Rotor blade passing period in timesteps (default 288)")
    args = parser.parse_args()
    
    if args.piv_limits:
        plot_xlim = (-0.3, 0.08)
        plot_ylim = (-0.1, 0.4)
    else:
        plot_xlim = args.xlim
        plot_ylim = args.ylim

    plot_all_standalone_figures(
        args.h5, 
        args.outdir, 
        exclude_tracks=args.exclude, 
        min_points=args.min_points,
        xlim=plot_xlim,
        ylim=plot_ylim,
        phase_step=args.phase_step,
        period=args.period
    )

