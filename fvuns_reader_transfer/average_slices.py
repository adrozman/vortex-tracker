import fvuns_reader
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
import glob
import os
import argparse

def process_slices(input_pattern, output_file, dx=0.15625, dz=None, radius_limit=None, max_hole_dist=3.0, npy_output=None):
    """
    Reads multiple fvuns slices, interpolates them onto a common Cartesian grid,
    averages the results (excluding holes), and writes to a new fvuns file or numpy file.
    """
    if dz is None:
        dz = dx

    files = glob.glob(input_pattern)
    if not files:
        print(f"No files found matching {input_pattern}")
        return

    print(f"Found {len(files)} files to process.")

    # 1. Read the first file to get metadata and bounds
    print("Reading template file for metadata...")
    f0 = fvuns_reader.read_fvuns(files[0])
    
    all_x = np.concatenate([g['x'] for g in f0['grids']])
    all_z = np.concatenate([g['z'] for g in f0['grids']])
    constant_y = f0['grids'][0]['y'][0]
    
    min_x, max_x = all_x.min(), all_x.max()
    min_z, max_z = all_z.min(), all_z.max()
    
    orig_min_x, orig_min_z = min_x, min_z
    
    if radius_limit is not None:
        print(f"Applying radius limit to bounding box: {radius_limit}")
        min_x = max(min_x, -radius_limit)
        max_x = min(max_x, radius_limit)
        min_z = max(min_z, -radius_limit)
        max_z = min(max_z, radius_limit)

    # Align min and max to multiples of dx and dz relative to the original grid phase
    min_x = orig_min_x + np.floor((min_x - orig_min_x) / dx) * dx
    max_x = orig_min_x + np.ceil((max_x - orig_min_x) / dx) * dx
    min_z = orig_min_z + np.floor((min_z - orig_min_z) / dz) * dz
    max_z = orig_min_z + np.ceil((max_z - orig_min_z) / dz) * dz

    print(f"Bounding box X: [{min_x:.2f}, {max_x:.2f}], Z: [{min_z:.2f}, {max_z:.2f}]")
    
    # 2. Generate common Cartesian grid
    x_lin = np.arange(min_x, max_x + dx * 0.1, dx)
    z_lin = np.arange(min_z, max_z + dz * 0.1, dz)
    nx = len(x_lin)
    nz = len(z_lin)
    
    XX, ZZ = np.meshgrid(x_lin, z_lin, indexing='ij')
    
    query_points = np.column_stack((XX.flatten(), ZZ.flatten()))
    num_query_pts = len(query_points)

    # Optional Limiter
    if radius_limit is not None:
        print(f"Applying radius limit: {radius_limit}")
        dists_to_origin = np.sqrt(query_points[:, 0]**2 + query_points[:, 1]**2)
        valid_query_mask = dists_to_origin <= radius_limit
    else:
        valid_query_mask = np.ones(num_query_pts, dtype=bool)

    variable_names = f0['variable_names']
    
    # Accumulators for averaging
    # shape: (num_vars, num_query_pts)
    sum_vars = np.zeros((len(variable_names), num_query_pts), dtype=np.float32)
    count_vars = np.zeros((len(variable_names), num_query_pts), dtype=np.int32)

    # 3. Process each file
    for fidx, filepath in enumerate(files):
        print(f"[{fidx+1}/{len(files)}] Processing {os.path.basename(filepath)}...")
        data = fvuns_reader.read_fvuns(filepath)
        
        all_valid_pts = []
        all_valid_vars = {var_name: [] for var_name in variable_names}
        
        for g in data['grids']:
            bndry_nodes = set()
            for b in g.get('boundary_faces', []):
                if len(b['connectivity']) > 0:
                    bndry_nodes.update(b['connectivity'].flatten())
            if 0 in bndry_nodes:
                bndry_nodes.remove(0)
                
            # Convert to 0-based indices and sort
            valid_indices = np.array(sorted(list(bndry_nodes))) - 1
            
            # Extract points
            pts_x = g['x'][valid_indices]
            pts_z = g['z'][valid_indices]
            all_valid_pts.append(np.column_stack((pts_x, pts_z)))
            
            for var_name in variable_names:
                all_valid_vars[var_name].append(g['variables'][var_name][valid_indices])
                
        pts = np.concatenate(all_valid_pts)
        
        tree = cKDTree(pts)
        dists, _ = tree.query(query_points)
        valid_mask = (dists <= max_hole_dist) & valid_query_mask
        
        for v_idx, var_name in enumerate(variable_names):
            vals = np.concatenate(all_valid_vars[var_name])
            
            # Interpolate
            interp = LinearNDInterpolator(pts, vals)
            interp_vals = interp(query_points)
            
            # Apply masks (holes and convex hull NaNs)
            mask = valid_mask & ~np.isnan(interp_vals)
            
            sum_vars[v_idx, mask] += interp_vals[mask]
            count_vars[v_idx, mask] += 1
            
    # 4. Compute Average
    print("Computing averages...")
    avg_vars = np.zeros_like(sum_vars)
    with np.errstate(invalid='ignore'):
        avg_vars = sum_vars / count_vars
        
    # Create a mask of nodes that actually received data from at least one file
    # We use the first variable's count as a proxy for all.
    has_data_mask = count_vars[0] > 0
    
    # We leave them as np.nan where there is no data for the .npy file,
    # but for fvuns we want 0.0 so we don't write NaNs to binary.
    avg_vars_fvuns = np.copy(avg_vars)
    avg_vars_fvuns[count_vars == 0] = 0.0

    # ADD DUMMY HEX TO SATISFY FIELDVIEW UNSTRUCTURED FORMAT REQUIREMENT
    # FieldView requires at least one 3D volume element to consider an unstructured file valid.
    dummy_x = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float32) * dx + query_points[0, 0]
    dummy_z = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.float32) * dz + query_points[0, 1]
    dummy_y = np.full(8, constant_y + dx, dtype=np.float32)
    
    final_x = np.concatenate([query_points[:, 0], dummy_x])
    final_y = np.concatenate([np.full(num_query_pts, constant_y, dtype=np.float32), dummy_y])
    final_z = np.concatenate([query_points[:, 1], dummy_z])
    
    final_vars = []
    for v_idx in range(len(variable_names)):
        final_vars.append(np.concatenate([avg_vars_fvuns[v_idx], np.zeros(8, dtype=np.float32)]))
        
    num_final_pts = num_query_pts + 8
    
    dummy_hex_nodes = np.arange(num_query_pts + 1, num_query_pts + 9, dtype=np.int32)
    dummy_hex = {
        'headers': np.array([1310719], dtype=np.int32),
        'nodes': np.array([dummy_hex_nodes], dtype=np.int32)
    }

    print("Building Cartesian output grid...")
    
    # We will build Quads (Boundary faces type 1)
    # FieldView nodes are 1-indexed.
    faces = []
    # To avoid rendering elements outside the radius, we skip those quads
    # We also skip any quad that touches a node with NO data (the hole)
    global_valid_mask = valid_query_mask & has_data_mask
    
    for i in range(nx - 1):
        for j in range(nz - 1):
            n1 = i * nz + j
            n2 = (i + 1) * nz + j
            n3 = (i + 1) * nz + (j + 1)
            n4 = i * nz + (j + 1)
            
            # Only add quad if all 4 nodes are within the valid mask
            if global_valid_mask[n1] and global_valid_mask[n2] and global_valid_mask[n3] and global_valid_mask[n4]:
                # 1-based indexing for fvuns
                faces.append([n1 + 1, n2 + 1, n3 + 1, n4 + 1])
                
    faces = np.array(faces, dtype=np.int32)
    print(f"Created {len(faces)} quad faces.")

    # Reconstruct the metadata dictionary
    out_data = {
        'header': "Averaged Cartesian Interpolation",
        'version': f0['version'],
        'file_type': f0.get('file_type', 1441),
        'reserved': f0.get('reserved', 0),
        'constants': f0['constants'],
        'boundary_types': [{'flags': np.array([0, 0], dtype=np.int32), 'name': 'Averaged_Slice'}],
        'variable_names': variable_names,
        'bvariable_names': [],
        'grids': [{
            'nnodes': num_final_pts,
            'x': final_x,
            'y': final_y,
            'z': final_z,
            'boundary_faces': [{
                'bndry_type_idx': 1,
                'num_faces': len(faces),
                'connectivity': faces
            }] if len(faces) > 0 else [],
            'elements': {'tets': [], 'hexes': [dummy_hex], 'prisms': [], 'pyramids': []},
            'variables': {var_name: final_vars[v_idx] for v_idx, var_name in enumerate(variable_names)}
        }]
    }

    # 6. Write to file
    if output_file:
        print(f"Writing output to {output_file}...")
        fvuns_reader.write_fvuns(output_file, out_data)

    if npy_output:
        print(f"Saving numpy data to {npy_output}...")
        npy_data = {
            'x': XX,
            'z': ZZ,
            'variables': {var_name: avg_vars[v_idx].reshape((nx, nz)) for v_idx, var_name in enumerate(variable_names)}
        }
        np.save(npy_output, npy_data)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate and average fvuns slices.")
    parser.add_argument("--input", type=str, default="extracts/*.fvuns", help="Glob pattern for input files")
    parser.add_argument("--output", type=str, default="averaged_output.fvuns", help="Output filename")
    parser.add_argument("--npy_output", type=str, default=None, help="Optional output filename for numpy dictionary (.npy)")
    parser.add_argument("--dx", type=float, default=0.15625, help="Grid resolution in X")
    parser.add_argument("--dz", type=float, default=None, help="Grid resolution in Z (defaults to dx)")
    parser.add_argument("--radius", type=float, default=None, help="Optional radius limiter from origin")
    parser.add_argument("--hole_dist", type=float, default=3.0, help="Max distance to nearest point to not be considered a hole")
    
    args = parser.parse_args()
    process_slices(args.input, args.output, args.dx, args.dz, args.radius, args.hole_dist, args.npy_output)

