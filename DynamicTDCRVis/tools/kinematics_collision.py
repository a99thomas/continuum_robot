import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.spatial.transform import Rotation as R

from tools.constants import object_pos, object_rad, object_height, path_height
from tools.constants import kappa_init, phi_init, ell_init
from tools.constants import pos_error_penalty, ori_error_penalty, z_error_penalty, collision_vio_penalty, path_error_penalty

def lengths_to_q(type = "threesegtdcr", lengths = np.zeros((3,1)), radius = [0.0254, 0.0254, 0.0254]):
    """
    Convert lengths of each string to kappa, phi, and ell values
    """
    kappa= []
    phi = []
    ell = []
    if type == "threesegtdcr":
        for segment in range(len(lengths[0])):
            l1 = lengths[0, segment]
            l2 = lengths[1, segment]
            l3 = lengths[2, segment]
            g = l1**2 + l2**2 + l3**2 - l1*l2 - l2*l3 - l1*l3

            ell.append((l1 + l2 + l3) / 3)
            kappa.append(2 * g**(0.5) / (radius[segment] * (l1+l2+l3)))
            phi.append(np.arctan2((np.sqrt(3)*(l2+l3-2*l1)+0.00000000001), 3*(l2 - l3)))

        return np.array(kappa), np.array(phi), np.array(ell)

def robotindependentmapping(kappa: np.ndarray[float], phi: np.ndarray[float], ell: np.ndarray[float], pts_per_seg: np.ndarray[int]) -> np.ndarray[float]:
    """
    ROBOTINDEPENDENTMAPPING creates a framed curve for given configuration parameters

    Example
    -------
    g = robotindependentmapping([1/40e-3;1/10e-3],[0,pi],[25e-3,20e-3],10)
    creates a 2-segment curve with radius of curvatures 1/40 and 1/10
    and segment lengths 25 and 20, where the second segment is rotated by pi rad.

    Parameters
    ------
    kappa: ndarray
        (nx1) segment curvatures
    phi: ndarray
        (nx1) segment bending plane angles
    l: ndarray
        (nx1) segment lengths
    pts_per_seg: ndarray
        (nx1) number of points per segment
            if n=1 all segments with equal number of points

    Returns
    -------
    g : ndarray
        (mx16) backbone curve with m 4x4 transformation matrices, where m is
            total number of points, reshaped into 1x16 vector (columnwise)

    Author: Jessica Burgner-Kahrs <jbk@cs.toronto.edu>
    Date: 2022/02/16
    Version: 0.2
    Copyright: 2023 Continuum Robotics Laboratory, University of Toronto
    """

    if kappa.shape != phi.shape or kappa.shape != ell.shape:
        raise ValueError("Dimension mismatch.")

    numseg = kappa.size
    if pts_per_seg.size == 1 and numseg > 1: # If same number of points per segment
        pts_per_seg = np.tile(pts_per_seg, numseg)  # Create an array that is numseg long with the num points repeated

    g = np.zeros((np.sum(pts_per_seg+1), 16))  # Stores the transformation matrices of all the points in all the segments as rows

    p_count = 0  # Points counter
    R_y = R.from_euler('y', 90, degrees=True).as_matrix() #rotate around y by -90
    R_x = R.from_euler('x', 0, degrees=True).as_matrix() #rotate around y by -90
    R_base = R_y @ R_x
    T_base = np.eye(4)  # base starts off as identity
    T_base[:3, :3] = R_base # apply rotation
    T_base[:3, 3] = np.array([0, 0, 0])
    for i in range(numseg):
        c_p = np.cos(phi[i])
        s_p = np.sin(phi[i])

        for j in range(pts_per_seg[i]+1):
            c_ks = np.cos(kappa[i] * j * (ell[i]/pts_per_seg[i]))
            s_ks = np.sin(kappa[i] * j * (ell[i]/pts_per_seg[i]))

            T_temp = np.array([
                [ c_p*c_p*(c_ks-1) + 1, s_p*c_p*(c_ks-1),        c_p*s_ks, 0],
                [ s_p*c_p*(c_ks-1),     c_p*c_p*(1-c_ks) + c_ks, s_p*s_ks, 0],
                [-c_p*s_ks,            -s_p*s_ks,                c_ks,     0],
                [ 0,                    0,                       0,        0]
            ])

            if kappa[i] != 0:
                T_temp[:, 3] = [(c_p*(1-c_ks))/kappa[i], (s_p*(1-c_ks))/kappa[i], s_ks/kappa[i], 1]
            else:  # To avoid division by zero
                T_temp[:, 3] = [0, 0, j*(ell[i]/pts_per_seg[i]), 1]

            g[p_count, :] = (T_base @ T_temp).T.reshape((1, 16))  # A matlab reshape is column-wise and not row-wise
            p_count += 1

        T_base = g[p_count - 1, :].reshape(4, 4).T  # lastmost point's transformation matrix is the new base

    return g

def collision_penalty(params, kappa, phi, ell, g, object_pos, object_rad):

    penalty = 0

    for point in g[:, 12:15]: #extracted x, y, z positions for entire curve
        x, y, z = point
        dist_to_cylinder = np.sqrt((x - object_pos[0])**2 + (y - object_pos[1])**2)
        if dist_to_cylinder < object_rad:
            penalty += (object_rad - dist_to_cylinder)**2 #quadratic penalty for extra uh oh
    
    return penalty


def inverse_kinematics(target_pose, initial_guess, pts_per_seg, previous_params=None, kappa_limits = [[-15,15], [-15,15], [-15,15]], phi_limits = [[-4,4],[-4,4], [-4,4]], ell_limits = [[0.15, 1.2],[0.15,1.2], [0.15,1.2]], z_flag = 0):
    """
    Solves inverse kinematics (IK) for a continuum robot with joint limits.
    
    Parameters:
    - target_pose: 4x4 desired transformation matrix.
    - initial_guess: Initial guess for [kappa1, ..., kappaN, phi1,..., phiN, ell1, ..., ellN].
    - pts_per_seg: Number of points per segment.
    - kappa_limits: (N,2) array with min/max values for kappa.
    - phi_limits: (N,2) array with min/max values for phi.
    - ell_limits: (N,2) array with min/max values for ell.

    Returns:
    - Optimal segment parameters within limits.
    """

    def pose_error(params):
        # print(f"pose_error function received z_flag: {z_flag}")  # Debugging
        """Objective function to minimize position & orientation error."""

        # extract robot params
        num_segments = 3
        kappa = np.array(params[0:num_segments])
        phi = np.array(params[num_segments:2*num_segments])
        ell = np.array(params[2*num_segments:3*num_segments])

        # get curvature
        g = robotindependentmapping(kappa, phi, ell, pts_per_seg)
        T_actual = g[-1, :].reshape(4, 4).T

        pos_error = np.linalg.norm(T_actual[:3, 3] - target_pose[:3, 3])
        ori_error = np.linalg.norm(T_actual[:3, :3] - target_pose[:3, :3], 'fro')**2

        # Penalize longer paths
        partial_ell = np.array(ell[:2])
        #print("partial ell: ", partial_ell)
        path_length = np.sum(partial_ell)  # Sum of first two segment lengths
        path_penalty = path_length 

        # Penalize 1st segment curvature
        first_kappa = kappa[0]
        second_kappa = kappa[1]
        #print(f"first kappa: {first_kappa:.4f}, second kappa: {second_kappa:.4f}")

        # Z-level penalty
        z_values = g[:, 14]  # Extract all z coordinates
        partial_idx = len(z_values) // 2 + 4  # Find the halfway index
        z_values_second_half = z_values[partial_idx:]  # Slice to get the second half
        z_penalty = np.sum((z_values_second_half - path_height)**2)  # Penalize large height variations

        # collision penalty
        collision_vio = collision_penalty(params, kappa, phi, ell, g, object_pos, object_rad)

        # print statement on penalties
        #print(f"POS: {pos_error:.4f}, ORI: {ori_error:.4f}, collision: {collision_vio:.4f}, z flag: {z_flag:.4f}, z: {10 * z_penalty * z_flag:.4f}, length reward: {path_penalty:.4f}")

        # Smoothness penalty: Encourage small incremental changes
        smoothness_penalty = 0
        delta = 0.2
        if previous_params is not None:
            change = np.abs(params - previous_params)
            excess_change = np.maximum(0, change - delta)  # Penalize only changes exceeding `delta`
            smoothness_penalty = np.sum(excess_change**2) * 1e3  # Scale penalty factor

        total_error = pos_error_penalty * pos_error + ori_error_penalty * ori_error + z_error_penalty * z_penalty * z_flag + collision_vio_penalty * collision_vio + path_error_penalty * path_penalty

        return total_error

    # Construct bounds for each parameter
    num_segments = len(initial_guess) // 3
    bounds = (
        np.vstack(kappa_limits).tolist() +  # Kappa bounds
        np.vstack(phi_limits).tolist() +    # Phi bounds
        np.vstack(ell_limits).tolist()      # Ell bounds
    )

    result = minimize(pose_error, initial_guess, method='L-BFGS-B', bounds=bounds, tol=1e-5, options={'maxiter': 100})

    num_segments = len(result.x) // 3
    result.x[num_segments:2*num_segments] = (result.x[num_segments:2*num_segments] + np.pi) % (2 * np.pi) - np.pi
    # print(result.success)
    return result.x

    

def equations(vars, ell, kappa, phi, r):
    # Unpack variables
    l1, l2, l3 = vars
    
    # Equation 1: l1 + l2 + l3 = 3 * ell
    eq1 = l1 + l2 + l3 - 3 * ell
    
    # Equation 2: g = (kappa^2 * r^2 * (3 * ell)^2) / 4
    g = (kappa**2 * r**2 * (3 * ell)**2) / 4
    eq2 = l1**2 + l2**2 + l3**2 - l1*l2 - l2*l3 - l1*l3 - g
    
    # Equation 3: phi = arctan(3 * (l3 - l2) / (sqrt(3) * (l2 + l3 - 2 * l1)))
    # eq3 = np.arctan2(3 * (l3 - l2) , (np.sqrt(3) * (l2 + l3 - 2 * l1))) - phi
    eq3 = np.arctan2((np.sqrt(3)*(l2+l3-2*l1)+0.00000000001), 3*(l2 - l3)) - phi
    
    # Return the equations as a numpy array
    return np.reshape(np.array([eq1, eq2, eq3]),3)

def q_to_lengths(kappa, phi, ell, r):
    """
    Calculate the lengths of three tendons for multiple continuum robot segments.

    Parameters:
    kappa (float or array-like): Curvature(s) of the segments (1/m).
    phi (float or array-like): Orientation(s) of the curvature planes (radians).
    ell (float or array-like): Arc length(s) of the segments (m).
    r (float or array-like): Radius/radii of tendon attachment points (m).

    Returns:
    list of lists: A list where each element is a list of lengths of the three tendons [l1, l2, l3] (m) for a segment.
    """
    # Ensure all inputs are numpy arrays and at least 1D
    # kappa = np.atleast_1d(kappa)
    # phi = np.atleast_1d(phi)
    # ell = np.atleast_1d(ell)
    # r = np.atleast_1d(r)

    tendon_lengths = []

    # Angular positions of the tendons around the segment cross-section
    theta = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])

    # Initial guesses for tendon lengths
    initial_guesses = [0.8, 1.1, 0.4]
    # print("KAPPS",kappa)
    # Iterate over each segment and solve the system of equations
    for segment in range(len(kappa)):
        # print(kappa[segment])
        if kappa[segment] == 0:
            # Straight configuration: all tendons have the same length
            tendon_lengths.append([ell[segment]] * 3)
        else:
            # Solve the system of equations for each segment using fsolve
            # print(ell[segment], kappa[segment], phi[segment], r[segment])
            l1, l2, l3 = fsolve(equations, initial_guesses, args=(ell[segment], kappa[segment], phi[segment], r[segment]))
            tendon_lengths.append([l1, l2, l3])
    
    return np.array(tendon_lengths)

if __name__ == "__main__":
    # Define target pose (4x4 transformation matrix)
    target_pose = np.eye(4)
    target_pose[:3, 3] = [0.1, 0.2, 0.3]  
    target_pose[:3, :3] = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()

    # Initial guess for [kappa1, phi1, ell1, kappa2, phi2, ell2]
    # initial_guess = np.array([1/40e-3, 0, 25e-3, 1/10e-3, np.pi, 20e-3])
    initial_guess = np.array([0.84141899, 0., 0., 0.98696486, -0.01547761, -0.01547761, 0.34352859, 0.392, 0.392])

    # Number of points per segment
    pts_per_seg = np.array([10, 10, 10])

    # Solve IK
    optimal_params = inverse_kinematics(target_pose, initial_guess, pts_per_seg, z_flag=0)
    #print("Optimal IK Parameters:", optimal_params)
    num_segments = len(optimal_params) // 3
    kappa = np.array(optimal_params[:num_segments])
    phi = np.array(optimal_params[num_segments:2*num_segments])
    ell = np.array(optimal_params[2*num_segments:])
