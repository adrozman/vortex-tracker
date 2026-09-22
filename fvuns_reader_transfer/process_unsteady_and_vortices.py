import os
import sys
import glob
import re
import argparse
import time
import numpy as np
import h5py
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.ndimage import label

# Local import
from fvuns_reader import read_fvuns
from ellipse_fit import fit_ellipse_moments

def process_unsteady_and_vortices(
    input_pattern="coviz/pivplane_*.fvuns",
    output_h5="velocity_and_vortex_data.h5",
    dx=0.078125,
    dz=None,
    start=None,
    end=None,
    step=None,
    theta_deg=-20.0,           # [deg] Rotates entire grid & velocities to rotor/freestream frame
    rotor_diam=24.0,           # [in] Rotor diameter D for X/D, Z/D normalization
    box_limit_factor=0.75,     # Square box domain half-width (+/- box_limit_factor * D)
    u_inf_nondim=10.0 / 340.0, # Freestream Mach (U_inf / a_inf) for velocity normalization
    q_threshold=1.0e6,         # [1/s^2] Dimensional Q-criterion vortex detection threshold
    sound_speed=340.0,         # [m/s] Reference speed of sound
    length_scale=0.0254        # [m/in] Grid unit in meters (1 in = 0.0254 m)
):
    """
    Processes planar PIV slice files (.fvuns):
      1. Interpolates density, momentum, and Q-criterion onto a square Cartesian grid (+/- 0.75 D).
      2. Rotates coordinates and velocities to the rotor reference frame (theta_deg).
      3. Computes Reynolds decomposition: time-mean U, W, unsteady components u', w',
         and Reynolds normal stresses / variances <u'^2>, <w'^2>.
      4. Detects tip vortex cores for each timestep, fits ellipses (center, semi-axes, tilt),
         and tracks vortex trajectories over time.
      5. Exports all data to an HDF5 file accessible by MATLAB and Python.
    """
    if dz is None:
        dz = dx

    t_start = time.time()
    files = glob.glob(input_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {input_pattern}")

    # Extract numeric step identifiers from filenames
    file_nums = []
    for f in files:
        m = re.search(r'\d+', os.path.basename(f))
        file_nums.append(int(m.group()) if m else -1)

    sorted_indices = sorted(range(len(files)), key=lambda i: file_nums[i])
    files = [files[i] for i in sorted_indices]
    file_nums = [file_nums[i] for i in sorted_indices]

    # Filter files if start, end, or step are provided
    if start is not None or end is not None or step is not None:
        base_start = start if start is not None else file_nums[0]
        filtered_files = []
        filtered_nums = []
        for f, num in zip(files, file_nums):
            if start is not None and num < start:
                continue
            if end is not None and num > end:
                continue
            if step is not None and (num - base_start) % step != 0:
                continue
            filtered_files.append(f)
            filtered_nums.append(num)
        files = filtered_files
        file_nums = filtered_nums

    print(f"Found {len(files)} files to process:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    # 1. Establish Cartesian grid bounds
    # Square domain bounded by +/- (box_limit_factor * rotor_diam), default +/- 0.75 D (+/- 18 inches).
    # This avoids reading all grids in the template file to find max values and eliminates
    # circular blank regions for a clean, uniform Cartesian grid.
    limit_x = box_limit_factor * rotor_diam
    limit_z = box_limit_factor * rotor_diam

    # Build symmetric grid covering [-limit, +limit] with step dx, dz
    n_half_x = int(np.round(limit_x / dx))
    n_half_z = int(np.round(limit_z / dz))
    x_lin = np.arange(-n_half_x * dx, n_half_x * dx + dx * 0.5, dx)
    z_lin = np.arange(-n_half_z * dz, n_half_z * dz + dz * 0.5, dz)
    nx, nz = len(x_lin), len(z_lin)
    XX, ZZ = np.meshgrid(x_lin, z_lin, indexing='ij')
    query_points = np.column_stack((XX.flatten(), ZZ.flatten()))

    print(f"\nCartesian square grid bounds: +/- {box_limit_factor:.2f} * D (+/- {limit_x:.2f} inches)")
    print(f"Grid dimensions: {nx} x {nz} ({nx * nz} query points)")
    print(f"X range: [{x_lin[0]:.2f}, {x_lin[-1]:.2f}], Z range: [{z_lin[0]:.2f}, {z_lin[-1]:.2f}]")

    # Coordinate rotation: X_rot, Z_rot in rotor units (divided by D)
    theta = np.radians(theta_deg)
    X_rot = (-XX / rotor_diam) * np.cos(theta) - (ZZ / rotor_diam) * np.sin(theta)
    Z_rot = (-XX / rotor_diam) * np.sin(theta) + (ZZ / rotor_diam) * np.cos(theta)

    # Q scaling constant: convert non-dim Q to dimensional 1/s^2
    q_scale = (sound_speed / length_scale) ** 2

    # Storage for time series
    timesteps = []
    U_snapshots = []
    W_snapshots = []
    Q_snapshots = []
    vortex_detections_per_step = []

    # 2. Process each slice
    print("\nProcessing slices...")
    for f_idx, filepath in enumerate(files):
        t_slice = time.time()
        bname = os.path.basename(filepath)
        data = read_fvuns(filepath)
        current_time = float(data['constants']['time'])
        timesteps.append(current_time)

        all_valid_pts = []
        all_rho = []
        all_rhou = []
        all_rhow = []
        all_qq = []

        for g in data['grids']:
            bndry_nodes = set()
            for b in g.get('boundary_faces', []):
                if len(b['connectivity']) > 0:
                    bndry_nodes.update(b['connectivity'].flatten())
            if 0 in bndry_nodes:
                bndry_nodes.remove(0)
            if len(bndry_nodes) == 0:
                continue
            valid_indices = np.array(sorted(list(bndry_nodes))) - 1
            all_valid_pts.append(np.column_stack((g['x'][valid_indices], g['z'][valid_indices])))
            all_rho.append(g['variables']['density [nondim]'][valid_indices])
            all_rhou.append(g['variables']['rhou [nondim]; momentum [nondim]'][valid_indices])
            all_rhow.append(g['variables']['rhow [nondim]'][valid_indices])
            if 'QQ [nondim]' in g['variables']:
                all_qq.append(g['variables']['QQ [nondim]'][valid_indices])

        pts = np.concatenate(all_valid_pts)
        rho = np.concatenate(all_rho)
        rhou = np.concatenate(all_rhou)
        rhow = np.concatenate(all_rhow)

        # Velocities: U inverted consistent with X inversion
        u_orig = -rhou / rho
        w_orig = rhow / rho

        # Rotated velocities
        u_rot = u_orig * np.cos(theta) - w_orig * np.sin(theta)
        w_rot = u_orig * np.sin(theta) + w_orig * np.cos(theta)

        # Delaunay interpolation (triangulation computed once per file geometry)
        tri = Delaunay(pts)
        interp_u = LinearNDInterpolator(tri, u_rot)
        interp_w = LinearNDInterpolator(tri, w_rot)

        u_grid = interp_u(query_points).reshape((nx, nz))
        w_grid = interp_w(query_points).reshape((nx, nz))

        U_snapshots.append(u_grid)
        W_snapshots.append(w_grid)

        # Q-criterion interpolation & vortex detection
        if len(all_qq) > 0:
            qq_raw = np.concatenate(all_qq) * q_scale
            interp_q = LinearNDInterpolator(tri, qq_raw)
            q_grid = interp_q(query_points).reshape((nx, nz))
            Q_snapshots.append(q_grid)

            # Vortex core detection using sub-pixel quadratic peak search and FWHM spoke sampling:
            tip_line_z = -0.33 * X_rot + 0.02
            roi_mask = (q_grid > q_threshold) & (Z_rot > tip_line_z) & (X_rot > -0.35) & (X_rot < 0.12)
            labeled_array, num_features = label(roi_mask)

            step_ellipses = []
            num_spokes = 16
            spoke_angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
            r_search = np.linspace(0, 1.2, 100)  # search up to 1.2 inches radius

            for feat in range(1, num_features + 1):
                mask = (labeled_array == feat)
                if np.sum(mask) < 15:  # filter noise artifacts
                    continue

                # Sub-pixel peak search
                sub_q = np.where(mask, q_grid, -1e9)
                ix, iz = np.unravel_index(np.argmax(sub_q), q_grid.shape)

                if 1 <= ix < q_grid.shape[0] - 1 and 1 <= iz < q_grid.shape[1] - 1:
                    patch_q = q_grid[ix-1:ix+2, iz-1:iz+2]
                    dq_dx = (patch_q[2, 1] - patch_q[0, 1]) / 2.0
                    d2q_dx2 = patch_q[2, 1] - 2 * patch_q[1, 1] + patch_q[0, 1]
                    off_x = -dq_dx / d2q_dx2 if abs(d2q_dx2) > 1e-5 else 0.0

                    dq_dz = (patch_q[1, 2] - patch_q[1, 0]) / 2.0
                    d2q_dz2 = patch_q[1, 2] - 2 * patch_q[1, 1] + patch_q[1, 0]
                    off_z = -dq_dz / d2q_dz2 if abs(d2q_dz2) > 1e-5 else 0.0

                    off_x = np.clip(off_x, -0.5, 0.5)
                    off_z = np.clip(off_z, -0.5, 0.5)
                    mesh_xc = float(XX[ix, iz] + off_x * dx)
                    mesh_zc = float(ZZ[ix, iz] + off_z * dz)
                else:
                    mesh_xc = float(XX[ix, iz])
                    mesh_zc = float(ZZ[ix, iz])

                xc_rot = float((-mesh_xc / rotor_diam) * np.cos(theta) - (mesh_zc / rotor_diam) * np.sin(theta))
                zc_rot = float((-mesh_xc / rotor_diam) * np.sin(theta) + (mesh_zc / rotor_diam) * np.cos(theta))
                q_max = float(q_grid[ix, iz])

                # Extract FWHM (Q = 0.5 * Q_max) contour boundary points along 16 radial spokes
                bnd_pts_rot = []
                for phi in spoke_angles:
                    x_spoke = mesh_xc + r_search * np.cos(phi)
                    z_spoke = mesh_zc + r_search * np.sin(phi)
                    pts_spoke = np.column_stack((x_spoke, z_spoke))
                    q_spoke = interp_q(pts_spoke)

                    target_q = 0.5 * q_max
                    below = np.where(q_spoke < target_q)[0]
                    if len(below) > 0:
                        idx_b = below[0]
                        if idx_b > 0:
                            q1, q2 = q_spoke[idx_b-1], q_spoke[idx_b]
                            r1, r2 = r_search[idx_b-1], r_search[idx_b]
                            r_half = r1 + (target_q - q1) * (r2 - r1) / (q2 - q1 + 1e-12)
                        else:
                            r_half = r_search[0]
                    else:
                        r_half = r_search[-1]

                    x_bnd_m = mesh_xc + r_half * np.cos(phi)
                    z_bnd_m = mesh_zc + r_half * np.sin(phi)
                    x_bnd_r = float((-x_bnd_m / rotor_diam) * np.cos(theta) - (z_bnd_m / rotor_diam) * np.sin(theta))
                    z_bnd_r = float((-x_bnd_m / rotor_diam) * np.sin(theta) + (z_bnd_m / rotor_diam) * np.cos(theta))
                    bnd_pts_rot.append((x_bnd_r, z_bnd_r))

                bnd_pts_rot = np.array(bnd_pts_rot)

                # Fit ellipse to boundary points relative to peak center
                dx_b = bnd_pts_rot[:, 0] - xc_rot
                dz_b = bnd_pts_rot[:, 1] - zc_rot
                cov = np.cov(dx_b, dz_b)
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = eigvals.argsort()[::-1]
                a_rot = float(np.sqrt(2 * max(eigvals[order[0]], 1e-6)))
                b_rot = float(np.sqrt(2 * max(eigvals[order[1]], 1e-6)))
                ang_rot = float(np.degrees(np.arctan2(eigvecs[1, order[0]], eigvecs[0, order[0]])))

                step_ellipses.append({
                    'time': current_time,
                    'center': (xc_rot, zc_rot),
                    'mesh_center': (mesh_xc, mesh_zc),
                    'semi_major': a_rot,
                    'semi_minor': b_rot,
                    'angle_deg': ang_rot,
                    'q_max': q_max,
                    'bnd_pts': bnd_pts_rot
                })

            # Sort detected vortices by downstream position (X_rot)
            step_ellipses.sort(key=lambda e: e['center'][0])
            vortex_detections_per_step.append(step_ellipses)
        else:
            vortex_detections_per_step.append([])

        print(f"[{f_idx+1}/{len(files)}] Step {current_time:.0f}: {len(vortex_detections_per_step[-1])} vortex cores detected ({time.time() - t_slice:.2f}s)")

    # Convert snapshots to 3D arrays: shape (N_time, Nx, Nz)
    U_arr = np.array(U_snapshots, dtype=np.float32)
    W_arr = np.array(W_snapshots, dtype=np.float32)
    Q_arr = np.array(Q_snapshots, dtype=np.float32) if Q_snapshots else None
    timesteps_arr = np.array(timesteps, dtype=np.float64)

    # 3. Reynolds Decomposition
    print("\nComputing time-mean and unsteady velocity components...")
    with np.errstate(invalid='ignore'):
        U_mean = np.nanmean(U_arr, axis=0)
        W_mean = np.nanmean(W_arr, axis=0)

        # Unsteady components: u' = u - \bar{u}, w' = w - \bar{w}
        u_prime = U_arr - U_mean[np.newaxis, :, :]
        w_prime = W_arr - W_mean[np.newaxis, :, :]

        # Variances: <u'^2>, <w'^2>
        u_variance = np.nanmean(u_prime ** 2, axis=0)
        w_variance = np.nanmean(w_prime ** 2, axis=0)

    # Normalization by U_inf
    U_mean_norm = U_mean / u_inf_nondim
    W_mean_norm = W_mean / u_inf_nondim
    u_prime_norm = u_prime / u_inf_nondim
    w_prime_norm = w_prime / u_inf_nondim
    u_variance_norm = u_variance / (u_inf_nondim ** 2)
    w_variance_norm = w_variance / (u_inf_nondim ** 2)

    # 4. Vortex Tracking Across Timesteps
    print("\nTracking vortex trajectories across timesteps...")
    raw_tracks = []
    for t_idx, detections in enumerate(vortex_detections_per_step):
        t = timesteps[t_idx]
        if t_idx == 0:
            for det in detections:
                raw_tracks.append([det])
        else:
            unmatched = list(detections)
            for tr in raw_tracks:
                last_det = tr[-1]
                dt_steps = (t - last_det['time']) / 18.0 if len(timesteps) > 1 else 1.0
                dt_steps = max(dt_steps, 1.0)
                if dt_steps > 2.5:
                    continue
                last_x, last_z = last_det['center']
                best_idx = None
                best_dist = 1e9
                for cand_idx, cand in enumerate(unmatched):
                    cand_x, cand_z = cand['center']
                    dx_cand = cand_x - last_x
                    dz_cand = cand_z - last_z
                    dist = np.sqrt(dx_cand**2 + dz_cand**2)
                    # Relaxed convection bounds: downstream movement
                    if -0.01 <= dx_cand <= 0.08 * dt_steps and abs(dz_cand) <= 0.06 * dt_steps:
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = cand_idx
                if best_idx is not None:
                    matched = unmatched.pop(best_idx)
                    tr.append(matched)
            for rem in unmatched:
                raw_tracks.append([rem])

    # Final pass: polynomial outlier rejection and minimum length check
    tracks = []
    for tr in raw_tracks:
        if len(tr) < 2:
            continue
        if len(tr) >= 4:
            x_arr = np.array([d['center'][0] for d in tr])
            z_arr = np.array([d['center'][1] for d in tr])
            poly = np.polyfit(x_arr, z_arr, 2)
            z_fit = np.polyval(poly, x_arr)
            residuals = np.abs(z_arr - z_fit)
            std_res = np.std(residuals)
            valid = residuals <= max(3 * std_res, 0.02)
            clean_tr = [tr[i] for i in range(len(tr)) if valid[i]]
        else:
            clean_tr = tr
        if len(clean_tr) >= 2:
            tracks.append(clean_tr)

    tracks.sort(key=lambda tr: tr[0]['center'][0])
    print(f"Formed {len(tracks)} continuous vortex tracks:")
    for idx, tr in enumerate(tracks):
        print(f"  Track {idx+1}: {len(tr)} steps (t={tr[0]['time']:.0f} to {tr[-1]['time']:.0f}), X/D: [{tr[0]['center'][0]:.3f} -> {tr[-1]['center'][0]:.3f}], Z/D: [{tr[0]['center'][1]:.3f} -> {tr[-1]['center'][1]:.3f}]")

    # 5. Export to HDF5
    print(f"\nWriting results to HDF5 file: {output_h5}...")
    with h5py.File(output_h5, 'w') as hf:
        # File-level metadata
        hf.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        hf.attrs['description'] = 'Planar PIV slices: mean, unsteady velocities, and tip vortex trajectories'
        hf.attrs['rotor_diameter_inches'] = float(rotor_diam)
        hf.attrs['rotation_angle_deg'] = float(theta_deg)
        hf.attrs['U_inf_nondim'] = float(u_inf_nondim)
        hf.attrs['speed_of_sound_mps'] = float(sound_speed)
        hf.attrs['num_timesteps'] = len(timesteps)
        hf.attrs['grid_nx'] = nx
        hf.attrs['grid_nz'] = nz

        # Group 1: Grid Coordinates
        grp_grid = hf.create_group('grid')
        grp_grid.create_dataset('X_over_D', data=X_rot.astype(np.float32), compression='gzip')
        grp_grid.create_dataset('Z_over_D', data=Z_rot.astype(np.float32), compression='gzip')
        grp_grid.create_dataset('x_mesh', data=XX.astype(np.float32), compression='gzip')
        grp_grid.create_dataset('z_mesh', data=ZZ.astype(np.float32), compression='gzip')
        grp_grid['X_over_D'].attrs['units'] = 'non-dimensional (X/D, rotated by -20 deg)'
        grp_grid['Z_over_D'].attrs['units'] = 'non-dimensional (Z/D, rotated by -20 deg)'

        # Group 2: Time
        grp_time = hf.create_group('time')
        grp_time.create_dataset('timesteps', data=timesteps_arr)

        # Group 3: Time-Averaged Mean Velocities
        grp_mean = hf.create_group('mean')
        grp_mean.create_dataset('U_mean', data=U_mean, compression='gzip')
        grp_mean.create_dataset('W_mean', data=W_mean, compression='gzip')
        grp_mean.create_dataset('U_mean_norm', data=U_mean_norm, compression='gzip')
        grp_mean.create_dataset('W_mean_norm', data=W_mean_norm, compression='gzip')
        grp_mean['U_mean_norm'].attrs['units'] = '<U_bar> / U_inf'
        grp_mean['W_mean_norm'].attrs['units'] = '<W_bar> / U_inf'

        # Group 4: Unsteady Velocities (3D arrays: [N_time, Nx, Nz])
        grp_unsteady = hf.create_group('unsteady')
        grp_unsteady.create_dataset('u_prime', data=u_prime, compression='gzip')
        grp_unsteady.create_dataset('w_prime', data=w_prime, compression='gzip')
        grp_unsteady.create_dataset('u_prime_norm', data=u_prime_norm, compression='gzip')
        grp_unsteady.create_dataset('w_prime_norm', data=w_prime_norm, compression='gzip')
        grp_unsteady['u_prime'].attrs['dims'] = '(time, nx, nz)'
        grp_unsteady['w_prime'].attrs['dims'] = '(time, nx, nz)'
        grp_unsteady['u_prime_norm'].attrs['units'] = "u' / U_inf"
        grp_unsteady['w_prime_norm'].attrs['units'] = "w' / U_inf"

        # Group 5: Velocity Variances (Reynolds Normal Stresses)
        grp_var = hf.create_group('variance')
        grp_var.create_dataset('u_variance', data=u_variance, compression='gzip')
        grp_var.create_dataset('w_variance', data=w_variance, compression='gzip')
        grp_var.create_dataset('u_variance_norm', data=u_variance_norm, compression='gzip')
        grp_var.create_dataset('w_variance_norm', data=w_variance_norm, compression='gzip')
        grp_var['u_variance_norm'].attrs['units'] = "<u'^2> / U_inf^2"
        grp_var['w_variance_norm'].attrs['units'] = "<w'^2> / U_inf^2"

        # Group 6: Q-criterion
        if Q_arr is not None:
            grp_q = hf.create_group('q_criterion')
            grp_q.create_dataset('Q_mean', data=np.nanmean(Q_arr, axis=0), compression='gzip')
            grp_q.create_dataset('Q_snapshots', data=Q_arr, compression='gzip')
            grp_q['Q_mean'].attrs['units'] = '1 / s^2'

        # Group 7: Vortex Trajectories
        grp_vort = hf.create_group('vortices')
        for tr_idx, tr in enumerate(tracks):
            tr_name = f'track_{tr_idx+1:02d}'
            gtr = grp_vort.create_group(tr_name)
            gtr.create_dataset('time', data=np.array([d['time'] for d in tr]))
            gtr.create_dataset('X_over_D', data=np.array([d['center'][0] for d in tr]))
            gtr.create_dataset('Z_over_D', data=np.array([d['center'][1] for d in tr]))
            gtr.create_dataset('mesh_x', data=np.array([d['mesh_center'][0] for d in tr]))
            gtr.create_dataset('mesh_z', data=np.array([d['mesh_center'][1] for d in tr]))
            gtr.create_dataset('semi_major', data=np.array([d['semi_major'] for d in tr]))
            gtr.create_dataset('semi_minor', data=np.array([d['semi_minor'] for d in tr]))
            gtr.create_dataset('angle_deg', data=np.array([d['angle_deg'] for d in tr]))
            gtr.create_dataset('q_max', data=np.array([d['q_max'] for d in tr]))
            if 'bnd_pts' in tr[0]:
                gtr.create_dataset('bnd_pts_step0', data=tr[0]['bnd_pts'])

    print(f"Processing complete in {time.time() - t_start:.2f}s! Output saved to {output_h5}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mean, unsteady velocities, and tip vortex trajectories from .fvuns planar slices.")
    parser.add_argument("--input", type=str, default="coviz/pivplane_*.fvuns", help="Glob pattern for input .fvuns files")
    parser.add_argument("--output_h5", type=str, default="velocity_and_vortex_data.h5", help="Output HDF5 filename")
    parser.add_argument("--dx", type=float, default=0.078125, help="Grid resolution in X")
    parser.add_argument("--dz", type=float, default=None, help="Grid resolution in Z (defaults to dx)")
    parser.add_argument("--start", type=int, default=None, help="Start timestep filter")
    parser.add_argument("--end", type=int, default=None, help="End timestep filter")
    parser.add_argument("--step", type=int, default=None, help="Timestep stride filter")
    parser.add_argument("--theta", type=float, default=-20.0, help="Rotation angle in degrees (default -20.0)")
    parser.add_argument("--diameter", type=float, default=24.0, help="Rotor diameter in inches (default 24.0)")
    parser.add_argument("--box_limit_factor", type=float, default=0.75, help="Domain box half-width factor in rotor diameters [+/- factor * D] (default 0.75)")
    parser.add_argument("--q_thresh", type=float, default=1.0e6, help="Q-criterion threshold for vortex core detection")

    args = parser.parse_args()
    process_unsteady_and_vortices(
        input_pattern=args.input,
        output_h5=args.output_h5,
        dx=args.dx,
        dz=args.dz,
        start=args.start,
        end=args.end,
        step=args.step,
        theta_deg=args.theta,
        rotor_diam=args.diameter,
        box_limit_factor=args.box_limit_factor,
        q_threshold=args.q_thresh
    )

