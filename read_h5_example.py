import h5py
import numpy as np

import os
import sys

h5_path = sys.argv[1] if len(sys.argv) > 1 else "velocity_and_vortex_data.h5"
if not os.path.exists(h5_path):
    fallback = os.path.join(os.path.dirname(__file__), "velocity_and_vortex_data.h5")
    if os.path.exists(fallback):
        h5_path = fallback

print(f"Opening {h5_path}...")
with h5py.File(h5_path, 'r') as f:
    print("\n--- Root Attributes ---")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")
        
    print("\n--- Datasets in HDF5 File ---")
    def print_dataset_info(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  /{name:<35} shape={str(obj.shape):<18} dtype={obj.dtype}")
    f.visititems(print_dataset_info)

    # Example: Loading Coordinates
    X_over_D = f['grid/X_over_D'][:]
    Z_over_D = f['grid/Z_over_D'][:]

    # Example: Loading Time-Mean Normalized Velocities
    U_mean_norm = f['mean/U_mean_norm'][:]
    W_mean_norm = f['mean/W_mean_norm'][:]
    print(f"\nMean U/U_inf range: [{np.nanmin(U_mean_norm):.3f}, {np.nanmax(U_mean_norm):.3f}]")
    print(f"Mean W/U_inf range: [{np.nanmin(W_mean_norm):.3f}, {np.nanmax(W_mean_norm):.3f}]")

    # Example: Loading 3D Unsteady Velocity Arrays [N_time, Nx, Nz]
    u_prime_norm = f['unsteady/u_prime_norm'][:]
    w_prime_norm = f['unsteady/w_prime_norm'][:]
    print(f"Unsteady u'/U_inf shape: {u_prime_norm.shape}")

    # Example: Loading Vortex Tracks
    print("\n--- Tip Vortex Trajectories ---")
    if 'vortices' in f:
        for tr_name in sorted(f['vortices'].keys()):
            gtr = f['vortices'][tr_name]
            t = gtr['time'][:]
            x = gtr['X_over_D'][:]
            z = gtr['Z_over_D'][:]
            print(f"  {tr_name}: {len(t)} points, X/D [{x[0]:.3f} -> {x[-1]:.3f}], Z/D [{z[0]:.3f} -> {z[-1]:.3f}]")

print("\nHDF5 verification completed successfully.")

