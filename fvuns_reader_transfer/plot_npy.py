import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def plot_npy(npy_file, png_file, vmin=None, vmax=None, component='W'):
    print(f"Loading {npy_file}...")
    try:
        data = np.load(npy_file, allow_pickle=True).item()
        print("Data loaded.")
        X = -data['x'] /24
        Z = data['z'] / 24
        variables = data['variables']
        # dict_keys(['density [nondim]', 'rhou [nondim]; momentum [nondim]', 'rhov [nondim]', 'rhow [nondim]', 'energy [nondim]', 'turb1 [nondim]', 'bodyTag', 'QQ [nondim]', 'Vorticity Magnitude [nondim]'])
        density_name = list(variables.keys())[0]
        rhou_name = list(variables.keys())[1]
        rhow_name = list(variables.keys())[3]
        
        # Calculate original velocities. Since X was inverted, we invert U 
        # to keep the velocity vector consistent with the flipped geometry.
        U_orig = -variables[rhou_name] / variables[density_name]
        W_orig = variables[rhow_name] / variables[density_name]
        
        # Apply 20 degrees clockwise rotation
        theta = np.radians(-20)
        
        X_rot = X * np.cos(theta) - Z * np.sin(theta)
        Z_rot = X * np.sin(theta) + Z * np.cos(theta)
        
        U_rot = U_orig * np.cos(theta) - W_orig * np.sin(theta)
        W_rot = U_orig * np.sin(theta) + W_orig * np.cos(theta)
        
        if component.upper() == 'U':
            pname = "U/Uinf"
            V = U_rot / (10/340)
        else:
            pname = "W/Uinf"
            V = W_rot / (10/340)
        print(f"Shapes - X: {X_rot.shape}, Z: {Z_rot.shape}, V: {V.shape}")
        
        V_plot = np.where(V == 0.0, np.nan, V)
        print("Creating figure...")
        plt.figure(figsize=(10, 8))
        print("Running pcolormesh...")
        plt.pcolormesh(X_rot, Z_rot, V_plot, shading='auto', cmap='Spectral_r', vmin=vmin, vmax=vmax)
        print("Adding colorbar...")
        cbar=plt.colorbar()
        cbar.set_label(pname,fontsize=18)
        
        plt.xlabel("X/D", fontsize=18)
        plt.ylabel("Z/D", fontsize=18)
        #plt.xlim((-0.32,0.1))
        #plt.ylim((-0.12,0.4))
        
        print("Saving figure...")
        plt.savefig(png_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {png_file}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D data from the exported .npy file.")
    parser.add_argument("npy_file", help="Path to the .npy file")
    parser.add_argument("png_file", help="Path to save the resulting .png image")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum limit for the colorbar (optional)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum limit for the colorbar (optional)")
    parser.add_argument("--component", choices=["U", "W"], default="W", help="Velocity component to plot: U or W (default W)")
    
    args = parser.parse_args()
    plot_npy(args.npy_file, args.png_file, vmin=args.vmin, vmax=args.vmax, component=args.component)
