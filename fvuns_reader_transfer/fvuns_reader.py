import numpy as np
import struct
import os

def read_fvuns(filepath):
    """
    Reads a binary FieldView unstructured (.fvuns) file.
    
    Ported from C logic in fvavg.c.
    
    Args:
        filepath (str): Path to the .fvuns file.
        
    Returns:
        dict: Parsed data containing header, version, constants, boundary types, 
              variable names, and a list of grids.
    """
    
    FV_MAGIC = 66051
    FV_NODES = 1001
    FV_FACES = 1002
    FV_ELEMENTS = 1003
    FV_VARIABLES = 1004
    FV_BNDRY_VARS = 1006

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        # 1. Determine Endianness
        magic_bytes = f.read(4)
        if not magic_bytes:
            raise ValueError("Empty file.")
            
        # Try little-endian first
        magic = struct.unpack('<I', magic_bytes)[0]
        if magic == FV_MAGIC:
            endian = '<'
        else:
            # Try big-endian
            magic = struct.unpack('>I', magic_bytes)[0]
            if magic == FV_MAGIC:
                endian = '>'
            else:
                raise ValueError(f"Invalid magic number: {magic}. Expected {FV_MAGIC}.")

        # Helper functions for reading binary data
        def read_ints(n):
            if n == 0: return np.array([], dtype=f'{endian}i4')
            return np.fromfile(f, dtype=f'{endian}i4', count=n)
            
        def read_floats(n):
            if n == 0: return np.array([], dtype=f'{endian}f4')
            return np.fromfile(f, dtype=f'{endian}f4', count=n)
            
        def read_str80():
            s_bytes = f.read(80)
            if not s_bytes: return ""
            return s_bytes.decode('ascii', errors='ignore').split('\0', 1)[0].strip()

        # 2. Metadata Parsing
        header_str = read_str80()
        version = read_ints(2)
        v_major, v_minor = version[0], version[1]
        
        file_type = None
        if (v_major == 2 and v_minor >= 7) or v_major >= 3:
            file_type = read_ints(1)[0]
        
        reserved = None
        if (v_major == 2 and v_minor >= 6) or v_major >= 3:
            reserved = read_ints(1)[0]
            
        constants = read_floats(4) # [time, fsmach, alpha, re]
        
        num_grids = read_ints(1)[0]
        
        num_bndry_types = read_ints(1)[0]
        boundary_types = []
        for _ in range(num_bndry_types):
            flags = read_ints(2)
            name = read_str80()
            boundary_types.append({'flags': flags, 'name': name})
            
        num_vars = read_ints(1)[0]
        variable_names = []
        for _ in range(num_vars):
            variable_names.append(read_str80())
            
        num_bvars = read_ints(1)[0]
        bvariable_names = []
        for _ in range(num_bvars):
            bvariable_names.append(read_str80())
            
        # 3. Grid Parsing Loop
        grids = []
        for grid_idx in range(num_grids):
            grid_data = {}
            
            # FV_NODES
            node_header = read_ints(2) # [FV_NODES, nnodes]
            if node_header[0] != FV_NODES:
                raise ValueError(f"Expected FV_NODES tag (1001), got {node_header[0]} at grid {grid_idx}")
            nnodes = node_header[1]
            
            grid_data['nnodes'] = nnodes
            grid_data['x'] = read_floats(nnodes)
            grid_data['y'] = read_floats(nnodes)
            grid_data['z'] = read_floats(nnodes)
            
            # FV_FACES (Boundary Faces)
            faces = []
            tag = read_ints(1)[0]
            while tag == FV_FACES:
                ibt, nf = read_ints(2)
                if nf > 0:
                    # faces[ibt-1] = malloc(4*nf * sizeof(int));
                    face_connectivity = read_ints(4 * nf).reshape(nf, 4)
                    faces.append({
                        'bndry_type_idx': ibt, 
                        'num_faces': nf, 
                        'connectivity': face_connectivity
                    })
                tag = read_ints(1)[0]
            grid_data['boundary_faces'] = faces
            
            # FV_ELEMENTS
            elements = {
                'tets': [],
                'hexes': [],
                'prisms': [],
                'pyramids': []
            }
            while tag == FV_ELEMENTS:
                # [ntet, nhex, nprism, npyramid]
                counts = read_ints(4)
                ntet, nhex, nprism, npyramid = counts
                
                if ntet > 0:
                    # header (1) + nodes (4) = 5 ints per element
                    data = read_ints(ntet * 5).reshape(ntet, 5)
                    elements['tets'].append({
                        'headers': data[:, 0],
                        'nodes': data[:, 1:]
                    })
                    
                if nhex > 0:
                    # header (1) + nodes (8) = 9 ints per element
                    data = read_ints(nhex * 9).reshape(nhex, 9)
                    elements['hexes'].append({
                        'headers': data[:, 0],
                        'nodes': data[:, 1:]
                    })
                    
                if nprism > 0:
                    # header (1) + nodes (6) = 7 ints per element
                    data = read_ints(nprism * 7).reshape(nprism, 7)
                    elements['prisms'].append({
                        'headers': data[:, 0],
                        'nodes': data[:, 1:]
                    })
                    
                if npyramid > 0:
                    # header (1) + nodes (5) = 6 ints per element
                    data = read_ints(npyramid * 6).reshape(npyramid, 6)
                    elements['pyramids'].append({
                        'headers': data[:, 0],
                        'nodes': data[:, 1:]
                    })
                
                tag = read_ints(1)[0]
            grid_data['elements'] = elements
            
            # FV_VARIABLES
            if tag == FV_VARIABLES:
                variables = {}
                for var_name in variable_names:
                    variables[var_name] = read_floats(nnodes)
                grid_data['variables'] = variables
            elif num_vars > 0:
                 raise ValueError(f"Expected FV_VARIABLES tag (1004), got {tag} at grid {grid_idx}")

            # FV_BNDRY_VARS
            if num_bvars > 0:
                tag = read_ints(1)[0]
                if tag == FV_BNDRY_VARS:
                    raise NotImplementedError("Boundary variables (FV_BNDRY_VARS) are not yet implemented as the source C code skips them.")
                else:
                    raise ValueError(f"Expected FV_BNDRY_VARS tag (1006), got {tag}")

            grids.append(grid_data)

    return {
        'header': header_str,
        'version': {'major': v_major, 'minor': v_minor},
        'file_type': file_type,
        'reserved': reserved,
        'constants': {
            'time': constants[0],
            'fsmach': constants[1],
            'alpha': constants[2],
            're': constants[3]
        },
        'boundary_types': boundary_types,
        'variable_names': variable_names,
        'bvariable_names': bvariable_names,
        'grids': grids
    }

def write_fvuns(filepath, data, endian='<'):
    """
    Writes a dictionary of data to a binary FieldView unstructured (.fvuns) file.
    
    Args:
        filepath (str): Path to write the .fvuns file.
        data (dict): Data structure as returned by read_fvuns.
        endian (str): Endianness marker ('<' for little, '>' for big).
    """
    FV_MAGIC = 66051
    FV_NODES = 1001
    FV_FACES = 1002
    FV_ELEMENTS = 1003
    FV_VARIABLES = 1004
    FV_BNDRY_VARS = 1006

    with open(filepath, 'wb') as f:
        def write_ints(arr):
            np.array(arr, dtype=f'{endian}i4').tofile(f)

        def write_floats(arr):
            np.array(arr, dtype=f'{endian}f4').tofile(f)

        def write_str80(s):
            b = s.encode('ascii', errors='ignore')[:80]
            f.write(b.ljust(80, b'\0'))

        # 1. Magic
        f.write(struct.pack(f'{endian}I', FV_MAGIC))

        # 2. Metadata
        write_str80(data['header'])
        v_major = data['version']['major']
        v_minor = data['version']['minor']
        write_ints([v_major, v_minor])

        if (v_major == 2 and v_minor >= 7) or v_major >= 3:
            write_ints([data.get('file_type', 1441)]) # Default to FV_COMBINED_FILE
        
        if (v_major == 2 and v_minor >= 6) or v_major >= 3:
            write_ints([data.get('reserved', 0)])

        c = data['constants']
        write_floats([c['time'], c['fsmach'], c['alpha'], c['re']])

        write_ints([len(data['grids'])])

        write_ints([len(data['boundary_types'])])
        for bt in data['boundary_types']:
            write_ints(bt['flags'])
            write_str80(bt['name'])

        write_ints([len(data['variable_names'])])
        for name in data['variable_names']:
            write_str80(name)

        write_ints([len(data['bvariable_names'])])
        for name in data['bvariable_names']:
            write_str80(name)

        # 3. Grids
        for grid in data['grids']:
            # FV_NODES
            nnodes = grid['nnodes']
            write_ints([FV_NODES, nnodes])
            write_floats(grid['x'])
            write_floats(grid['y'])
            write_floats(grid['z'])

            # FV_FACES
            for bf in grid.get('boundary_faces', []):
                write_ints([FV_FACES, bf['bndry_type_idx'], bf['num_faces']])
                write_ints(bf['connectivity'].flatten())

            # FV_ELEMENTS
            elems = grid['elements']
            ntet = sum(len(b['nodes']) for b in elems['tets'])
            nhex = sum(len(b['nodes']) for b in elems['hexes'])
            nprism = sum(len(b['nodes']) for b in elems['prisms'])
            npyramid = sum(len(b['nodes']) for b in elems['pyramids'])
            
            if ntet > 0 or nhex > 0 or nprism > 0 or npyramid > 0:
                write_ints([FV_ELEMENTS, ntet, nhex, nprism, npyramid])
                
                for b in elems['tets']:
                    combined = np.column_stack((b['headers'], b['nodes'])).flatten()
                    write_ints(combined)
                for b in elems['hexes']:
                    combined = np.column_stack((b['headers'], b['nodes'])).flatten()
                    write_ints(combined)
                for b in elems['prisms']:
                    combined = np.column_stack((b['headers'], b['nodes'])).flatten()
                    write_ints(combined)
                for b in elems['pyramids']:
                    combined = np.column_stack((b['headers'], b['nodes'])).flatten()
                    write_ints(combined)

            # FV_VARIABLES
            if len(data['variable_names']) > 0:
                write_ints([FV_VARIABLES])
                for var_name in data['variable_names']:
                    write_floats(grid['variables'][var_name])
            
            # Note: FV_BNDRY_VARS are skipped as they are not yet supported by the reader.

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        data = read_fvuns(input_file)
        print(f"Header: {data['header']}")
        print(f"Version: {data['version']}")
        print(f"Variables: {data['variable_names']}")
        print(f"Number of grids: {len(data['grids'])}")
        
        # Test write
        output_file = "test_output.fvuns"
        print(f"Testing write to {output_file}...")
        write_fvuns(output_file, data)
        
        # Verify read back
        print("Verifying read back...")
        data2 = read_fvuns(output_file)
        if len(data['grids']) == len(data2['grids']):
            print("Round-trip SUCCESS: Grid counts match.")
        else:
            print("Round-trip FAILURE: Grid counts do not match.")

