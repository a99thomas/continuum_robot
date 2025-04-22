# constants.py

import numpy as np

# --- Object and environment properties ---
object_pos = np.array([0.7, 0.1, 0.0])  # Object position (x, y, z)
object_rad = 0.2                         # Cylinder radius
object_height = 2.5                      # Cylinder height
path_height = object_pos[2] + object_height / 2

# --- Initial robot parameters ---
kappa_init = np.array([0.001, 0.001, 0.001])
phi_init = np.array([0.001, 0.001, 0.001])
ell_init = np.array([0.1, 0.15, 0.185])
kappa_limits_init = [[-25, 40], [-25, 40], [-25, 40]]
phi_limits_init = [[-6, 6],[-6, 6],[-6,6]]
ell_limits_init = [[0.06, 0.12], [0.08, 0.17], [0.08, 0.2]]
num_segments_init = np.array([3,3,3])
seg_end_init = np.array([4, 8, 12])
k_values = [5, 3, 2]


# kappa_init = np.array([3.0, 1.25])
# phi_init = np.array([1.3, -1.5])
# ell_init = np.array([0.1, 0.15])
# kappa_limits_init = [[0, 25], [0, 25]]
# phi_limits_init = [[-4, 4],[-4, 4]]
# ell_limits_init = [[0.06, 0.12], [0.08, 0.17]]
# num_segments_init = np.array([3,3])
# seg_end_init = np.array([4, 8])
# k_values = [5, 3]


# kappa_init = np.array([1.0])
# phi_init = np.array([1.])
# ell_init = np.array([0.1])
# kappa_limits_init = [[0, 25]]
# phi_limits_init = [[-4, 4]]
# ell_limits_init = [[0.06, 0.12]]
# num_segments_init = np.array([3])
# seg_end_init = np.array([4])
# k_values = [5]


# --- Animation/trajectory settings ---
phase_duration = 30
total_phases = 5

# --- Penalty weights for inverse kinematics cost function ---
pos_error_penalty = 1
ori_error_penalty = 4
z_error_penalty = 10
collision_vio_penalty = 1
path_error_penalty = 0.25
