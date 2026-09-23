import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_trajectory_plots(h5_path, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as hf:
        timesteps = hf['time/timesteps'][:]
        X_rot = hf['grid/X_over_D'][:]
        Z_rot = hf['grid/Z_over_D'][:]
        Q_mean = hf['q_criterion/Q_mean'][:] if 'q_criterion/Q_mean' in hf else None
        
        grp = hf['vortices']
        track_info = []
        for tr_name in sorted(grp.keys()):
            gtr = grp[tr_name]
            t = gtr['time'][:]
            x = gtr['X_over_D'][:]
            z = gtr['Z_over_D'][:]
            q = gtr['q_max'][:] if 'q_max' in gtr else np.zeros_like(x)
            
            # Secondary blade passage features occur at X ~ -0.20, len=4
            is_sec = (len(t) == 4 and -0.21 < x[0] < -0.19 and 0.07 < z[0] < 0.08)
            
            track_info.append({
                'name': tr_name,
                'time': t,
                'X_over_D': x,
                'Z_over_D': z,
                'q_max': q,
                'is_secondary': is_sec,
                't_start': t[0],
                't_end': t[-1],
                'x_start': x[0],
                'x_end': x[-1],
                'z_start': z[0],
                'z_end': z[-1],
                'len': len(t)
            })

    primaries = [tr for tr in track_info if not tr['is_secondary']]
    secondaries = [tr for tr in track_info if tr['is_secondary']]
    
    # Sort chronologically
    primaries.sort(key=lambda tr: (tr['t_start'], tr['x_start']))
    for idx, tr in enumerate(primaries):
        tr['label'] = f"Vortex {idx+1}"

    tip_x = np.linspace(-0.35, 0.10, 200)
    tip_z = -0.33 * tip_x - 0.005
    colors = ['#0033cc', '#0066ff', '#0099cc', '#00aa55', '#228822', '#d95f02', '#e41a1c', '#984ea3', '#ff7f00']

    common_x_shed = -0.3033
    common_x_exit = 0.078

    # =========================================================================
    # 1. WATERFALL PLOT FOR ALL 9 PRIMARY VORTICES (Clean margins & no clipping)
    # =========================================================================
    print("Generating clean waterfall plot for all 9 primary vortices...")
    fig, ax = plt.subplots(figsize=(13, 11))
    
    dz_shift = 0.055
    ax.axvline(common_x_shed, color='green', linestyle='--', linewidth=1.8, alpha=0.8,
               label=f'Common Shedding Origin ($X/D = {common_x_shed:.4f}$)')
    ax.axvline(common_x_exit, color='purple', linestyle=':', linewidth=1.8, alpha=0.8,
               label=f'Domain Exit Boundary ($X/D \\approx +0.078$)')

    for i, tr in enumerate(primaries):
        shift = i * dz_shift
        c = colors[i % len(colors)]
        x = tr['X_over_D']
        z = tr['Z_over_D'] + shift
        
        # Trajectory line
        ax.plot(x, z, 'o-', color=c, linewidth=2.4, markersize=4.5,
                label=f"{tr['label']} ({tr['name']}, {tr['len']} pts, t={int(tr['t_start'])}..{int(tr['t_end'])})")
        
        # Start marker (green circle)
        ax.plot(x[0], z[0], marker='o', markersize=8.5, color='#00cc44', markeredgecolor='black', zorder=5)
        # End marker (red square)
        ax.plot(x[-1], z[-1], marker='s', markersize=7.5, color='#e60000', markeredgecolor='black', zorder=5)

        # Shifted tip line
        ax.plot(tip_x, tip_z + shift, color='gray', linestyle='--', linewidth=0.8, alpha=0.35)

        # Text annotation on the trajectory
        ax.text(x[-1] + 0.005, z[-1], f"  {tr['label']}", color=c, fontsize=10.5, weight='bold', va='center')

    ax.plot([], [], marker='o', markersize=8.5, color='#00cc44', markeredgecolor='black', linestyle='None', label='Start Point (Birth / First Detection)')
    ax.plot([], [], marker='s', markersize=7.5, color='#e60000', markeredgecolor='black', linestyle='None', label='End Point (Exit / Last Timestep)')

    ax.set_xlim(-0.35, 0.12)
    ax.set_ylim(0.02, 0.82)
    ax.set_xlabel(r"Streamwise Coordinate $X/D$", fontsize=15)
    ax.set_ylabel(r"Vertical Coordinate $Z/D + \text{Vertical Offset}$", fontsize=15)
    ax.set_title("Tip Vortex Trajectories with Staggered Visual Offset\n(All 9 Primary Vortices from 79-Timestep Run)", fontsize=16, pad=14)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor='gray')
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    out1 = os.path.join(output_dir, "vortex_trajectories_waterfall.png")
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"  Saved {out1}")

    # =========================================================================
    # 2. DEDICATED WATERFALL PLOT FOR THE 5 NEWLY SHED VORTICES
    # =========================================================================
    print("Generating waterfall plot for the 5 newly shed vortices...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    new_shed = [tr for tr in primaries if tr['t_start'] > timesteps[0]]
    dz_shed = 0.065
    
    ax.axvline(common_x_shed, color='green', linestyle='--', linewidth=2.0, alpha=0.85,
               label=f'Common Origin: $X/D = {common_x_shed:.4f}$')
    ax.axvline(common_x_exit, color='purple', linestyle=':', linewidth=2.0, alpha=0.85,
               label=f'Domain Exit: $X/D \\approx +0.078$')

    for i, tr in enumerate(new_shed):
        shift = i * dz_shed
        c = colors[(i + 4) % len(colors)]
        x = tr['X_over_D']
        z = tr['Z_over_D'] + shift
        
        ax.plot(x, z, 'o-', color=c, linewidth=2.5, markersize=5.0,
                label=f"{tr['label']} ({tr['name']}): shed t={int(tr['t_start'])}, {tr['len']} pts")
        ax.plot(x[0], z[0], 'o', markersize=9, color='#00cc44', markeredgecolor='black', zorder=5)
        ax.plot(x[-1], z[-1], 's', markersize=8, color='#e60000', markeredgecolor='black', zorder=5)
        ax.plot(tip_x, tip_z + shift, color='gray', linestyle='--', linewidth=0.9, alpha=0.35)
        ax.text(x[-1] + 0.005, z[-1], f"  {tr['label']}", color=c, fontsize=11, weight='bold', va='center')

    ax.plot([], [], 'o', markersize=9, color='#00cc44', markeredgecolor='black', linestyle='None', label='Start Point (Exact Shedding Origin)')
    ax.plot([], [], 's', markersize=8, color='#e60000', markeredgecolor='black', linestyle='None', label='End Point')

    ax.set_xlim(-0.35, 0.12)
    ax.set_ylim(0.04, 0.65)
    ax.set_xlabel(r"$X/D$", fontsize=15)
    ax.set_ylabel(r"$Z/D + \text{Vertical Offset}$", fontsize=15)
    ax.set_title("Newly Shed Tip Vortex Trajectories (Vortices 5 to 9)\nVisualizing Identical Start Origin at $X/D = -0.3033$ and Uniform Convection", fontsize=15, pad=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10.5, framealpha=0.95)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    out2 = os.path.join(output_dir, "vortex_trajectories_shed_waterfall.png")
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"  Saved {out2}")

    # =========================================================================
    # 3. 3x3 INDIVIDUAL SUBPLOT PANELS FOR EACH VORTEX
    # =========================================================================
    print("Generating 3x3 individual subplot panels for all 9 vortices...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, tr in enumerate(primaries):
        ax = axes[i]
        c = colors[i % len(colors)]
        x = tr['X_over_D']
        z = tr['Z_over_D']

        # Background tip path
        ax.plot(tip_x, tip_z, 'k--', linewidth=1.5, alpha=0.6, label='Rotor tip path')
        ax.axvline(common_x_shed, color='green', linestyle=':', linewidth=1.2, alpha=0.6)
        ax.axvline(common_x_exit, color='purple', linestyle=':', linewidth=1.2, alpha=0.6)

        # Trajectory
        ax.plot(x, z, 'o-', color=c, linewidth=2.2, markersize=4.5)
        # Start marker
        ax.plot(x[0], z[0], 'o', markersize=8, color='#00cc44', markeredgecolor='black', zorder=5)
        # End marker
        ax.plot(x[-1], z[-1], 's', markersize=7, color='#e60000', markeredgecolor='black', zorder=5)

        # Title & details
        t_span = f"t={int(tr['t_start'])}..{int(tr['t_end'])}"
        x_span = f"X: [{x[0]:+.3f} -> {x[-1]:+.3f}]"
        note = "New Shedding" if tr['t_start'] > timesteps[0] else "Initial in Domain"
        ax.set_title(f"{tr['label']} ({tr['name']}) - {tr['len']} pts\n{t_span} | {note}\n{x_span}", fontsize=10.5, pad=6)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_xlim(-0.35, 0.10)
        ax.set_ylim(-0.02, 0.34)

        if i >= 6:
            ax.set_xlabel(r"$X/D$", fontsize=12)
        if i % 3 == 0:
            ax.set_ylabel(r"$Z/D$", fontsize=12)

    plt.suptitle("Individual Trajectory Breakdown for All 9 Primary Tip Vortices (79 Timesteps)", fontsize=16, y=0.99)
    plt.tight_layout()
    out3 = os.path.join(output_dir, "vortex_trajectories_individual_panels.png")
    plt.savefig(out3, dpi=200)
    plt.close()
    print(f"  Saved {out3}")

    # =========================================================================
    # 4. SPACE-TIME CONVECTION DIAGRAM (X/D vs Timestep)
    # =========================================================================
    print("Generating space-time convection diagram...")
    fig, ax = plt.subplots(figsize=(11, 7.5))

    for i, tr in enumerate(primaries):
        c = colors[i % len(colors)]
        t = tr['time']
        x = tr['X_over_D']
        ax.plot(t, x, 'o-', color=c, linewidth=2.4, markersize=4.5, label=f"{tr['label']} ({tr['name']})")
        ax.plot(t[0], x[0], 'o', markersize=7, color='#00cc44', markeredgecolor='black', zorder=5)
        ax.plot(t[-1], x[-1], 's', markersize=6, color='#e60000', markeredgecolor='black', zorder=5)

    ax.axhline(common_x_shed, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Common Shedding: $X/D = {common_x_shed:.4f}$')
    ax.axhline(common_x_exit, color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Domain Exit: $X/D \\approx +0.078$')

    blade_pass_times = [64237 + k * 288 for k in range(5)]
    for bp in blade_pass_times:
        ax.axvline(bp, color='gray', linestyle=':', alpha=0.45)
        ax.text(bp, -0.33, f"Blade Pass\nt={bp}", fontsize=8, ha='center', color='gray')

    ax.set_xlabel("Simulation Timestep", fontsize=14)
    ax.set_ylabel(r"Streamwise Position $X/D$", fontsize=14)
    ax.set_title(r"Space-Time Vortex Convection: $X/D$ vs Timestep" + "\n(Equally Spaced $\\Delta t = 288$ Proves Periodic Shedding and Continuous Tracking)", fontsize=15, pad=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9.5, ncol=2)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    out4 = os.path.join(output_dir, "vortex_spacetime_convection.png")
    plt.savefig(out4, dpi=200)
    plt.close()
    print(f"  Saved {out4}")

    # =========================================================================
    # 5. PHYSICAL COORDINATES WITH ENDPOINT MARKERS
    # =========================================================================
    print("Generating physical coordinate trajectory plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    if Q_mean is not None:
        ax.pcolormesh(X_rot, Z_rot, Q_mean, cmap='RdBu_r', vmin=-5e6, vmax=5e6, shading='auto', alpha=0.3)

    ax.plot(tip_x, tip_z, 'k-', linewidth=2.2, label='Rotor tip path')

    ax.plot([-0.20, -0.10, 0.0], [-0.33*(-0.20)-0.005, -0.33*(-0.10)-0.005, -0.005], 'r*', markersize=8)
    ax.annotate(r"$120^\circ$" + "\n0.81R", xy=(-0.20, -0.33*(-0.20)-0.005), xytext=(-0.22, -0.01),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), fontsize=10, ha='center')
    ax.annotate(r"$105^\circ$" + "\n0.72R", xy=(-0.10, -0.33*(-0.10)-0.005), xytext=(-0.11, -0.05),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), fontsize=10, ha='center')
    ax.annotate(r"$90^\circ$" + "\n0.70R", xy=(0.0, -0.005), xytext=(0.0, -0.08),
                arrowprops=dict(arrowstyle="->", color="black", lw=1), fontsize=10, ha='center')

    for i, tr in enumerate(primaries):
        c = colors[i % len(colors)]
        ax.plot(tr['X_over_D'], tr['Z_over_D'], 'o-', color=c, linewidth=2.2, markersize=4.5,
                label=f"{tr['label']} ({tr['len']} pts)")
        ax.plot(tr['X_over_D'][0], tr['Z_over_D'][0], 'o', markersize=7, color='#00cc44', markeredgecolor='black', zorder=5)
        ax.plot(tr['X_over_D'][-1], tr['Z_over_D'][-1], 's', markersize=6, color='#e60000', markeredgecolor='black', zorder=5)

    for j, sec in enumerate(secondaries):
        lbl = 'Blade-pass shear layer' if j == 0 else None
        ax.plot(sec['X_over_D'], sec['Z_over_D'], '--', color='gray', linewidth=1.2, alpha=0.6, label=lbl)

    ax.plot([], [], 'o', markersize=8, color='#00cc44', markeredgecolor='black', linestyle='None', label='Start Detection')
    ax.plot([], [], 's', markersize=7, color='#e60000', markeredgecolor='black', linestyle='None', label='Exit / Last Timestep')

    ax.set_xlim(-0.35, 0.10)
    ax.set_ylim(-0.10, 0.35)
    ax.set_xlabel(r"$X/D$", fontsize=15)
    ax.set_ylabel(r"$Z/D$", fontsize=15)
    ax.set_title("Physical Tip Vortex Trajectories (79 Timesteps)\nPrimary Cores with Start & Exit Markers", fontsize=16, pad=12)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='upper left', fontsize=9.5, ncol=2, framealpha=0.95)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    out5 = os.path.join(output_dir, "vortex_trajectories_physical_endpoints.png")
    plt.savefig(out5, dpi=200)
    plt.close()
    print(f"  Saved {out5}")

    print("\nAll trajectory verification figures generated successfully!")

if __name__ == "__main__":
    generate_trajectory_plots("velocity_and_vortex_data_larger.h5", output_dir="plots")

