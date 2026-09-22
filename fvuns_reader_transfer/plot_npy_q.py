import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def plot_q(npy_file, png_file, vmin=None, vmax=None):
    print(f"Loading {npy_file}...")
    try:
        data = np.load(npy_file, allow_pickle=True).item()
        print("Data loaded.")
        X = -data['x'] / 24
        Z = data['z'] / 24
        variables = data['variables']
        
        # Find QQ in variables
        qq_name = None
        for k in variables.keys():
            if 'QQ' in k:
                qq_name = k
                break
        if qq_name is None:
            # Fallback to default name
            qq_name = 'QQ [nondim]'
            
        if qq_name not in variables:
            raise KeyError(f"Q criterion key '{qq_name}' not found in variables. Available keys: {list(variables.keys())}")
            
        V = variables[qq_name] *340*340/0.0254/0.0254
        print(np.nanmin(V), np.nanmax(V))
        pname = "Q criterion"
        
        # Apply 20 degrees clockwise rotation to coordinates
        theta = np.radians(-20)
        
        X_rot = X * np.cos(theta) - Z * np.sin(theta)
        Z_rot = X * np.sin(theta) + Z * np.cos(theta)
        
        print(f"Shapes - X: {X_rot.shape}, Z: {Z_rot.shape}, V: {V.shape}")
        
        V_plot = np.where(V == 0.0, np.nan, V)
        print("Creating figure...")
        plt.figure(figsize=(10, 8))
        print("Running pcolormesh...")
        plt.pcolormesh(X_rot, Z_rot, V_plot, shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        print("Adding colorbar...")
        cbar = plt.colorbar()
        cbar.set_label(pname, fontsize=18)
        
        plt.xlabel("X/D", fontsize=18)
        plt.ylabel("Z/D", fontsize=18)
        
        print("Saving figure...")
        plt.savefig(png_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {png_file}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Q criterion from the exported .npy file.")
    parser.add_argument("npy_file", help="Path to the .npy file")
    parser.add_argument("png_file", help="Path to save the resulting .png image")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum limit for the colorbar (optional)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum limit for the colorbar (optional)")
    
    args = parser.parse_args()
    plot_q(args.npy_file, args.png_file, vmin=args.vmin, vmax=args.vmax)

