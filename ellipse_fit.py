import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Given central moments of an isolated Q region:
# M00 = sum(Q)
# M10 = sum(x*Q), M01 = sum(z*Q)
# xc = M10/M00, zc = M01/M00
# u20 = sum((x-xc)^2 * Q) / M00
# u02 = sum((z-zc)^2 * Q) / M00
# u11 = sum((x-xc)*(z-zc) * Q) / M00
#
# The eigenvalues of the covariance matrix [u20, u11; u11, u02] give the squared semi-axes (for a 2-sigma ellipse, factor of 2 or sqrt(2)):
# lambda1, lambda2 = eigenvalues
# a = 2 * sqrt(lambda1), b = 2 * sqrt(lambda2)
# angle = 0.5 * atan2(2*u11, u20 - u02)

def fit_ellipse_moments(x_coords, z_coords, weights):
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
    
    # Covariance matrix
    cov = np.array([[u20, u11], [u11, u02]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues descending
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    # 2-sigma equivalent
    semi_major = 2.0 * np.sqrt(max(eigvals[0], 0))
    semi_minor = 2.0 * np.sqrt(max(eigvals[1], 0))
    
    # Angle of major axis with x-axis in degrees
    angle_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    angle_deg = np.degrees(angle_rad)
    
    return {
        'center': (xc, zc),
        'semi_major': semi_major,
        'semi_minor': semi_minor,
        'angle_deg': angle_deg,
        'cov': cov
    }

print("Ellipse moment fitting function ready.")
