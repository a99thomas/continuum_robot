### This is the code for Dynamic Tendon Driven Robot simulation
This is a dynamic visualization, adapted from CRVisToolkit from the Continuum Robotics Lab. https://github.com/ContinuumRoboticsLab/CRVisToolkit

Current work includes:
- Implementing basic kinematics and inverse kinematics
- Plotting the path given end effector pose or kappa, phi, and ell values for each stage (see ik_path.py and k_phi_ell_path.py)

Future work includes:
- Setting up RRT star path planning algorithms
- Controlling pose of multiple tendons for manipulation
- Implementing gamepad control for manual manipulation
- Adding kinematics for differing spring constants for different stages
