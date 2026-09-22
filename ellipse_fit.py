import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

def fit_ellipse_moments(x_coords, z_coords, weights):
    """Computes 2-sigma ellipse from weighted spatial moments of Q-criterion."""
    m00 = np.sum(weights)
    if m00 <= 0:
        return None
    xc = np.sum(x_coords * weights) / m00
    zc = np.sum(z_coords * weights) / m00
    
    dx = x_coords - xc
    dz = z_coords - zc
    
    u20 = np.sum(dx**2 * weights) / m00
    u02 = np.sum(dz**2 * weights) / m00
    u11 = np.sum(dx * dz * weights) / m00
    
    cov = np.array([[u20, u11], [u11, u02]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    semi_major = 2.0 * np.sqrt(max(eigvals[0], 0))
    semi_minor = 2.0 * np.sqrt(max(eigvals[1], 0))
    angle_deg = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    
    return {
        'center': (xc, zc),
        'semi_major': semi_major,
        'semi_minor': semi_minor,
        'angle_deg': angle_deg,
        'cov': cov
    }

def fit_ellipse_to_points(points):
    """Fits an ellipse (center, semi-axes a, b, orientation angle) to 2D boundary points."""
    pts = np.array(points)
    xc0 = np.mean(pts[:, 0])
    zc0 = np.mean(pts[:, 1])
    cov = np.cov(pts[:, 0] - xc0, pts[:, 1] - zc0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    a0 = np.sqrt(2 * max(eigvals[order[0]], 1e-4))
    b0 = np.sqrt(2 * max(eigvals[order[1]], 1e-4))
    ang0 = np.degrees(np.arctan2(eigvecs[1, order[0]], eigvecs[0, order[0]]))

    def loss(params):
        xc, zc, a, b, ang_deg = params
        if a <= 0.005 or b <= 0.005 or a > 0.15 or b > 0.15:
            return 1e10
        th = np.radians(ang_deg)
        x_shift = pts[:, 0] - xc
        z_shift = pts[:, 1] - zc
        x_rot = x_shift * np.cos(th) + z_shift * np.sin(th)
        z_rot = -x_shift * np.sin(th) + z_shift * np.cos(th)
        return np.sum(((x_rot / a)**2 + (z_rot / b)**2 - 1.0)**2)

    res = minimize(loss, [xc0, zc0, a0, b0, ang0], method='Nelder-Mead')
    xc_f, zc_f, a_f, b_f, ang_f = res.x
    if a_f < b_f:
        a_f, b_f = b_f, a_f
        ang_f += 90.0
    return xc_f, zc_f, a_f, b_f, ang_f % 180.0

def extract_vortex_swirl_ellipse(interp_U, interp_W, xc_init, zc_init, rot_to_mesh_fn,
                                 rotor_diam=24.0, s_max_inches=1.2, num_angles=8):
    """
    Extracts vortex core boundary from swirl velocity extrema along radial spokes
    and fits an ellipse defining the core boundary and center.
    """
    angles = np.linspace(0, 180, num_angles, endpoint=False)
    boundary_pts = []
    swirl_strengths = []
    s = np.linspace(-s_max_inches, s_max_inches, 200)

    for phi_deg in angles:
        phi = np.radians(phi_deg)
        xr_pts = xc_init + (s / rotor_diam) * np.cos(phi)
        zr_pts = zc_init + (s / rotor_diam) * np.sin(phi)
        xm_pts, zm_pts = rot_to_mesh_fn(xr_pts, zr_pts)
        pts = np.column_stack((xm_pts, zm_pts))
        v_swirl = -interp_U(pts) * np.sin(phi) + interp_W(pts) * np.cos(phi)

        cs = CubicSpline(s, v_swirl)
        s_fine = np.linspace(-s_max_inches * 0.85, s_max_inches * 0.85, 400)
        v_fine = cs(s_fine)
        i_min = np.argmin(v_fine)
        i_max = np.argmax(v_fine)
        s_min = s_fine[i_min]
        s_max = s_fine[i_max]
        dv = abs(v_fine[i_max] - v_fine[i_min]) / 2.0
        swirl_strengths.append(dv)

        p1_x = xc_init + (s_min / rotor_diam) * np.cos(phi)
        p1_z = zc_init + (s_min / rotor_diam) * np.sin(phi)
        p2_x = xc_init + (s_max / rotor_diam) * np.cos(phi)
        p2_z = zc_init + (s_max / rotor_diam) * np.sin(phi)
        boundary_pts.append((p1_x, p1_z))
        boundary_pts.append((p2_x, p2_z))

    mean_swirl = float(np.mean(swirl_strengths))
    xc_fit, zc_fit, a_fit, b_fit, ang_fit = fit_ellipse_to_points(boundary_pts)
    aspect = a_fit / b_fit if b_fit > 0 else 99.0

    return {
        'center': (float(xc_fit), float(zc_fit)),
        'semi_major': float(a_fit),
        'semi_minor': float(b_fit),
        'angle_deg': float(ang_fit),
        'aspect_ratio': float(aspect),
        'mean_swirl': mean_swirl,
        'boundary_pts': boundary_pts
    }
