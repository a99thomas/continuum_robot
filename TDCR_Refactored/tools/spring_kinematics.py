import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.spatial.transform import Rotation as R

def lengths_to_config(type = "threesegtdcr", lengths = np.zeros((3,1)), radius = [0.0254]):
    """
    Convert lengths of each string to kappa, phi, and ell values
    """
    config = np.empty((len(lengths[0])))
    kappa= []
    phi = []
    ell = []
    if type == "threesegtdcr":
        for segment in range(len(lengths[0])):
            l1 = lengths[0, segment]
            l2 = lengths[1, segment]
            l3 = lengths[2, segment]
            ell.append((l1 + l2 + l3) / 3)
            g = l1**2 + l2**2 + l3**2 - l1*l2 - l2*l3 - l1*l3
            kappa.append(2 * g**(0.5) / (radius[segment] * (l1+l2+l3)))

            # phi.append(180 / np.pi * np.arctan(3**0.5 / 3 * (l3 + l2 - 2*l1) / (l2-l3 -(0.000000001))))
            phi.append(np.arctan2((np.sqrt(3)*(l2+l3-2*l1)+0.00000000001), 3*(l2 - l3)))
            # kappa.append((l2 + l3 - 2*l1)/((l1+l2+l3) * radius * np.sin(np.pi - phi[-1])))
        config[0,:] = kappa
        config[1,:] = phi
        config[2,:] = ell

        return config
    

def config_to_task(config, pts_per_seg):
    """
    Generates backbone curve using constant curvature assumption.

    Parameters
    ----------
    kappa, phi, ell : ndarray
        Segment curvature, bending plane angles, and lengths (shape: (n,))
    pts_per_seg : ndarray or int
        Number of points per segment (uniform if scalar)

    Returns
    -------
    g : ndarray
        (m, 16) transformation matrices along the backbone
    """
    kappa = config[0,:]
    phi = config[1,:]
    ell = config[2,:]
    if kappa.shape != phi.shape or kappa.shape != ell.shape:
        raise ValueError("Dimension mismatch.")

    numseg = kappa.size
    # print("NUMSEG", numseg)
    # print(kappa)
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


from scipy.optimize import minimize

def inverse_kinematics(target_pose, initial_guess, pts_per_seg, kappa_limits, phi_limits, ell_limits, k_values):
    """
    Solves inverse kinematics (IK) for a continuum robot with joint limits.
    
    Parameters:
    - target_pose: 4x4 desired transformation matrix.
    - initial_guess: Initial guess for [kappa1, phi1, ell1, ..., kappaN, phiN, ellN].
    - pts_per_seg: Number of points per segment.
    - kappa_limits: (N,2) array with min/max values for kappa.
    - phi_limits: (N,2) array with min/max values for phi.
    - ell_limits: (N,2) array with min/max values for ell.

    Returns:
    - Optimal segment parameters within limits.
    """

    def pose_error(params):
        """Objective function to minimize position & orientation error."""
        num_segments = len(params) // 3
        config = np.empty((3, num_segments))
        kappa = np.array(params[0:num_segments])
        # phi = np.array(params[num_segments:2*num_segments])
        phi = np.array(params[num_segments:2*num_segments])
        ell = np.array(params[2*num_segments:3*num_segments])

        config[0,:] = kappa
        config[1,:] = phi
        config[2,:] = ell

        g = config_to_task(config, pts_per_seg)
        T_actual = g[-1, :].reshape(4, 4).T

        pos_error = np.linalg.norm(T_actual[:3, 3] - target_pose[:3, 3])

        R_actual = R.from_matrix(T_actual[:3, :3])
        R_target = R.from_matrix(target_pose[:3, :3])
        ori_error = 1 - np.dot(R_actual.as_quat(), R_target.as_quat()) ** 2
        # print(pos_error + 0.75 * ori_error)
        # print("ORI",ori_error)
        # print(pos_error)
        return pos_error + 2 * ori_error

    # Construct bounds for each parameter
    num_segments = len(initial_guess) // 3
    bounds = (
        np.vstack(kappa_limits).tolist() +  # Kappa bounds
        np.vstack(phi_limits).tolist() +    # Phi bounds
        np.vstack(ell_limits).tolist()      # Ell bounds
    )

    def spring_constraint(params):
        """
        Returns an array of constraint values for a continuum robot with multiple segments.
        
        For each adjacent segment pair (i and i+1), the constraint is:
            k_values[i]*(ell_limits[i][1] - ell[i]) - k_values[i+1]*(ell_limits[i+1][1] - ell[i+1]) - margin >= 0
        If all constraints are >= 0, the inequality holds for every adjacent pair.
        
        Parameters:
            params : array-like
                Optimization vector containing parameters for all segments arranged as 
                [kappa1, kappa2, ..., phi1, phi2, ..., ell1, ell2, ...].
        
        Returns:
            np.array: Array of constraint values (one per adjacent segment pair).
        """
        num_segments = len(params) // 3
        ell = np.array(params[2*num_segments:3*num_segments])
        
        # If only one segment, no constraint needs to be enforced.
        if num_segments < 2:
            return np.array([1.0])
        
        margin = .0  # Adjust this margin as needed.
        constraints = []
        # For each adjacent segment pair, enforce the inequality constraint.
        for i in range(num_segments - 1):
            # print(i)
            l_max_i = ell_limits[i][1]
            l_max_next = ell_limits[i+1][1]
            k_i = k_values[i]
            k_next = k_values[i+1]
            # Compute constraint value: should be >= 0 to be feasible.
            constraint_value = k_i * (l_max_i - ell[i]) - k_next * (l_max_next - ell[i+1]) - margin
            constraints.append(constraint_value)
        return np.array(constraints)


    constraints = [{'type': 'ineq', 'fun': spring_constraint}]

    # Use SLSQP method (supports nonlinear constraints)
    result = minimize(pose_error, initial_guess, method='SLSQP', bounds=bounds, 
                      constraints=constraints, tol=1e-5)
    num_segments = len(result.x) // 3
    # Normalize phi values to be between -pi and pi
    result.x[num_segments:2*num_segments] = (result.x[num_segments:2*num_segments] + np.pi) % (2 * np.pi) - np.pi

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

def config_to_lengths(config, r):
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

    kappa = config[0,:]
    phi = config[1,:]
    ell = config[2,:]

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
    initial_guess = np.array([1/40e-3, 0, 25e-3, 1/10e-3, np.pi, 20e-3])

    # Number of points per segment
    pts_per_seg = np.array([10, 10])

    # Solve IK
    optimal_params = inverse_kinematics(target_pose, initial_guess, pts_per_seg)
    print("Optimal IK Parameters:", optimal_params)
    num_segments = len(optimal_params) // 3
    kappa = np.array(optimal_params[:num_segments])
    phi = np.array(optimal_params[num_segments:2*num_segments])
    ell = np.array(optimal_params[2*num_segments:])

