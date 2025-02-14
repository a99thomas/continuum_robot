import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path


# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tools.kinematics import robotindependentmapping, inverse_kinematics
from tools.plotting import setupfigure, plot_tf, update_plot, draw_tdcr
from tools.rrt_star import RRTStar  # Assuming you have an RRT* implementation

# Define workspace boundaries
workspace_bounds = [[-0.3, 0.3], [-0.3, 0.3], [0.3, 0.9]]

# Define start and goal positions
start = np.array([0.0, 0.0, 0.4])
goal = np.array([0.25, 0.28, 0.6])

# Plan path using RRT*
rrt_star = RRTStar(start, goal)
path = rrt_star.plan()

# Visualize planned path
fig, ax = setupfigure()
ax.plot(*zip(*path), marker='o', color='blue', label="Planned Path")

# Initialize robot parameters
radius = [0.0254, 0.0254]
initial_guess = np.array([0, 0, 0.3, 0, 0, 0.3])
seg_end = np.array([11,22])

# Animate robot following the path
def animate():
    # Define initial robot parameters
    radius = [0.0254, 0.0254]
    kappa, phi, ell = np.array([1,1]), np.array([0, 0]), np.array([0.392, 0.392])
    g = robotindependentmapping(np.array(kappa), np.array(phi), np.array(ell), np.array([10]))
    g0 = g
    fig, ax = setupfigure(g0=g)

    ## Trajectory Plotting
    t_values = np.linspace(0, 100, 500)
    x_values = 0.25 * np.sin(t_values * np.pi / 20)
    y_values = 0.28 * np.cos(t_values * np.pi / 20)
    z_values = 0.6 + 0.1 * np.cos(t_values * np.pi / 2)

    ax.plot(x_values, y_values, z_values, label="Trajectory")
    
    seg_end = np.array([11,22])  
    clearance = 0.03
    curvelength = np.sum(np.linalg.norm(g[1:, 12:15] - g[:-1, 12:15], axis=1))
    
    # Define initial_guess in the same function
    initial_guess = np.array([0, 0, 0.3, 0, 0, 0.3])

    def frame_update(frame):
        # Use initial_guess directly (no need for nonlocal)
        nonlocal initial_guess  # This is now valid
        
        ax.clear()
        ax.plot(x_values, y_values, z_values, label="Trajectory")
        
        max_val_x = np.max(np.abs(g0[:, 12])) + clearance
        max_val_y = np.max(np.abs(g0[:, 13])) + clearance
        ax.set_xlim(-max_val_x, max_val_x)
        ax.set_ylim(-max_val_y, max_val_y)
        ax.set_zlim(0, curvelength + clearance)
        
        target_pose = np.eye(4)
        target_pose[:3, 3] = [
            0.25 * np.sin(frame * np.pi / 20), 
            0.28 * np.cos(frame * np.pi / 20), 
            0.6 + 0.1 * np.cos(frame * np.pi / 2)
        ]  
        target_pose[:3, :3] = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
        
        target_position = target_pose[:3, 3]
        
        # Perform inverse kinematics
        optimal_params = inverse_kinematics(target_pose, initial_guess, pts_per_seg=np.array([10, 10]))
        initial_guess = optimal_params  # Update guess for the next frame

        kappa = np.array(optimal_params[:2])
        phi = np.array(optimal_params[2:4])
        ell = np.array(optimal_params[4:])

        g = robotindependentmapping(kappa, phi, ell, np.array([10]))
        
        plot_elements = plot_tf(ax, g, seg_end, tipframe=True, segframe=False, baseframe=True, projections=True, baseplate=True)
        update_plot(plot_elements, g)

        draw_tdcr(ax, g, seg_end, r_disk=2.5e-2, r_height=1.5e-3, tipframe=True, segframe=False, baseframe=True, projections=False, baseplate=False)
        
        ax.scatter(*target_position, color='red', s=100, label="Target Position")

        plt.draw()

    ani = FuncAnimation(fig, frame_update, frames=200, interval=10)
    plt.show()

animate()
