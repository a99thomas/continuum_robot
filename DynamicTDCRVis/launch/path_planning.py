import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation

# Add parent directory to system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tools.kinematics_collision import lengths_to_q, robotindependentmapping, q_to_lengths, inverse_kinematics, collision_penalty
from tools.plotting import setupfigure, plot_tf, update_plot, draw_tdcr
from tools.constants import object_pos, object_rad, object_height, path_height
from tools.constants import kappa_init, phi_init, ell_init
from tools.constants import phase_duration, total_phases

# Global Variables
object_pos, object_rad, object_height, path_height = object_pos, object_rad, object_height, path_height
z_flag = 0
lock = 0
previous_params = None

def clear_plot(plot_elements):
    """ Clears all plotted elements from the figure without resetting axis limits. """
    for element in plot_elements.values():
        if isinstance(element, list):
            for e in element:
                e.remove()
        else:
            element.remove()

# Trajectory functions
def x_path_func(t): return object_pos[0] + object_rad * np.cos(t-np.pi/2)
def y_path_func(t): return object_pos[1] + object_rad * np.sin(t-np.pi/2)
def z_path_func(t): return path_height

def orientation(t):
    """ Compute rotation matrix so that the tip is perpendicular to the cylindrical path. """
    tangent = np.array([-np.sin(t), np.cos(t), 0])
    binormal = np.array([0, 0, 1])
    normal = np.array([np.cos(t), np.sin(t), 0])
    return np.column_stack([binormal, tangent, normal])

def animate():
    """ Animates the continuum robot movement. """
    global previous_params

    # Initialize robot parameters
    kappa, phi, ell = kappa_init, phi_init, ell_init
    g0 = robotindependentmapping(kappa, phi, ell, np.array([10]))
    fig, ax = setupfigure(g0=g0)

    # Plot trajectory
    theta_values = np.linspace(0, np.pi, 5)
    ax.plot(x_path_func(theta_values), y_path_func(theta_values), z_path_func(theta_values), label="Trajectory")

    # Visualization parameters
    seg_end = np.array([11, 22, 33])
    clearance = 0.1
    curvelength = np.sum(np.linalg.norm(g0[1:, 12:15] - g0[:-1, 12:15], axis=1))

    def frame_update(frame):
        """ Updates the robot configuration at each animation frame. """
        global previous_params, z_flag, lock
        nonlocal kappa, phi, ell

        # Phase definitions
        phase = frame // phase_duration
        if phase >= total_phases:
            return

        # Define target pose
        target_pose = np.eye(4)

        if phase == 0:  # Move to start position
            z_flag = frame*0.02
            t_transition = (frame % phase_duration) / phase_duration
            print("frame: ", frame)
            start_pos = np.array([0.6, 0.0, 1.5])
            end_pos = np.array([
                x_path_func(0),
                y_path_func(0),
                z_path_func(0)
            ])

            target_pose[:3, 3] = (1 - t_transition) * start_pos + t_transition * end_pos
            target_pose[:3, :3] = (1 - t_transition) * np.eye(3) + t_transition * orientation(0)
            print("moving to start pos")
        elif phase == 1:  # Wrap around cylinder
            z_flag = 1
            t = (frame % phase_duration) / phase_duration * np.pi
            print("t: ", t)
            print("frame: ", frame)
            target_pose[:3, 3] = [x_path_func(t), y_path_func(t), z_path_func(t)]
            target_pose[:3, :3] = orientation(t)
            print("wrapping around cyl")
        elif phase == 2:  # Move object up
            print(" ######## MOVING OBJECT UP ######## ")
            print("frame: ", frame)
            z_flag = 0
            t_transition = ((frame-60) % phase_duration) / phase_duration

            start_pos = np.array([
                x_path_func(3.036872898470133), 
                y_path_func(3.036872898470133), 
                z_path_func(3.036872898470133)
                ])
            # Target position: to the right of the object
            end_pos = np.array([
                object_pos[0],        # same x
                object_pos[1],  # shift right in y
                path_height + 0.2         # same z
            ])

            # Define target pose
            target_pose[:3, 3] = (1 - t_transition) * start_pos + t_transition * end_pos
            target_pose[:3, :3] = orientation(3.036872898470133)
        elif phase == 3:  # move object over
            print(" ######## MOVING OBJECT OVER ######## ")
            print("frame: ", frame)
            z_flag = 0
            t_transition = ((frame-60) % phase_duration) / phase_duration

            start_pos = np.array([
                object_pos[0], 
                object_pos[1], 
                path_height + 0.2
                ])
            # Target position: to the right of the object
            end_pos = np.array([
                object_pos[0],        # same x
                object_pos[1] + 0.5,  # shift right in y
                path_height         # same z
            ])

            # Define target pose
            target_pose[:3, 3] = (1 - t_transition) * start_pos + t_transition * end_pos
            target_pose[:3, :3] = orientation(3.036872898470133)
        elif phase == 4:  # unwind back to straight
            z_flag = 0
            t_transition = ((frame - 120) % phase_duration) / phase_duration
            # Start pose: where we were at end of phase 3
            start_pos = np.array([
                object_pos[0],
                object_pos[1] + 0.5,
                path_height
            ])

            # Target position: unwind
            end_pos = np.array([
                object_pos[0] + 0.2,        # right shift x
                object_pos[1] + 0.2,  # shift left in y
                path_height + 0.3         # up z
            ])

            # Define target pose
            target_pose[:3, 3] = (1 - t_transition) * start_pos + t_transition * end_pos
            target_pose[:3, :3] = orientation(3.036872898470133)


        initial_guess = np.concatenate([kappa, phi, ell]) if previous_params is None else previous_params
        optimal_params = inverse_kinematics(target_pose, initial_guess, np.array([10, 10, 10]), previous_params=previous_params, z_flag=z_flag)
        previous_params = optimal_params.copy()

        # Extract updated values
        num_segments = len(optimal_params) // 3
        kappa, phi, ell = optimal_params[:num_segments], optimal_params[num_segments:2*num_segments], optimal_params[2*num_segments:]
        print(f"kappa: {kappa}, phi: {phi}, ell: {ell}")
        # Update visualization
        ax.clear()
        ax.plot(x_path_func(theta_values), y_path_func(theta_values), z_path_func(theta_values), label="Trajectory")

        # Set plot limits
        max_val_x, max_val_y = np.max(np.abs(g0[:, 12])) + clearance, np.max(np.abs(g0[:, 13])) + clearance
        ax.set_xlim(0, max_val_x)
        ax.set_ylim(-1.0, max_val_y)
        ax.set_zlim(0, curvelength + clearance)

        # Compute FK and update plot
        g = robotindependentmapping(kappa, phi, ell, np.array([10, 10, 10]))
        plot_elements = plot_tf(ax, g, seg_end, tipframe=True, segframe=False, baseframe=True, projections=True, baseplate=True)
        plot_elements2 = draw_tdcr(ax,
            g, 
            seg_end, 
            r_disk=2.5*1e-2, 
            r_height=1.5*1e-3, 
            tipframe=True, 
            segframe=False, 
            baseframe=True, 
            projections=False, 
            baseplate=False
        )
        update_plot(plot_elements, g)

        # Plot target position
        ax.scatter(*target_pose[:3, 3], color='red', s=100, label="Target Position")

        # Plot tip reference frame
        scale = 0.05
        R_mat = target_pose[:3, :3]
        pos = target_pose[:3, 3]
        ax.quiver(pos[0], pos[1], pos[2], R_mat[0, 0], R_mat[1, 0], R_mat[2, 0], color='r', length=scale)
        ax.quiver(pos[0], pos[1], pos[2], R_mat[0, 1], R_mat[1, 1], R_mat[2, 1], color='g', length=scale)
        ax.quiver(pos[0], pos[1], pos[2], R_mat[0, 2], R_mat[1, 2], R_mat[2, 2], color='b', length=scale)

    # Start animation
    ani = FuncAnimation(fig, frame_update, frames=200, interval=10)
    plt.show()

if __name__ == "__main__":
    animate()
