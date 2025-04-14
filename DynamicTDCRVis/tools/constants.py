# constants.py

import numpy as np

# --- Object and environment properties ---
object_pos = np.array([0.7, 0.1, 0.0])  # Object position (x, y, z)
object_rad = 0.2                         # Cylinder radius
object_height = 2.5                      # Cylinder height
path_height = object_pos[2] + object_height / 2

# --- Initial robot parameters ---
kappa_init = np.array([1.0, 1.0, 1.0])
phi_init = np.array([0.0, 0.0, 0.0])
ell_init = np.array([0.392, 0.392, 0.392])

# --- Animation/trajectory settings ---
phase_duration = 30
total_phases = 5

# --- Penalty weights for inverse kinematics cost function ---
pos_error_penalty = 1
ori_error_penalty = 4
z_error_penalty = 10
collision_vio_penalty = 1
path_error_penalty = 0.25
