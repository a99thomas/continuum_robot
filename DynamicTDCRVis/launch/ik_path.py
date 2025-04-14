import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from testing.aruco_tracking import get_aruco_transform_matrix


from matplotlib.animation import FuncAnimation
from tools.kinematics import lengths_to_q, robotindependentmapping, q_to_lengths, inverse_kinematics
from tools.plotting import setupfigure, plot_tf, update_plot, draw_tdcr
        

current_path = Path(__file__).resolve().parent.parent
print(f"Current script path: {current_path}")

# Global trajectory parameters
x_path_freq = np.pi / 40  
y_path_freq = np.pi / 40  
z_path_freq = np.pi / 4

def clear_plot(plot_elements):
    """
    Clears all the plotted elements from the figure without resetting the axis limits.
    """
    for element in plot_elements.values():
        if isinstance(element, list):
            # For segment frames, clear each one
            for e in element:
                e.remove()
        else:
            element.remove()

# Global trajectory functions
def x_path_func(t):
    return 0.25 * np.sin(t * x_path_freq)

def y_path_func(t):
    return 0.28 * np.cos(t * y_path_freq)

def z_path_func(t):
    return 0.6 + 0.1 * np.cos(t * z_path_freq)

def animate():
    # Define initial robot parameters
    radius = [0.0254, 0.0254]
    kappa, phi, ell = np.array([1,1]), np.array([0, 0]), np.array([0.392, 0.392])
    g = robotindependentmapping(np.array(kappa), np.array(phi), np.array(ell), np.array([10]))
    g0 = g
    fig, ax = setupfigure(g0=g)

    ##Trajectory Plotting
    t_values = np.linspace(0, 100, 500)

    # Calculate the trajectory for each time value
    x_values = x_path_func(t_values)
    y_values = y_path_func(t_values)
    z_values = z_path_func(t_values)

    ax.plot(x_values, y_values, z_values, label="Trajectory")
    
    seg_end = np.array([11,22])  # Example segment indices
    clearance = 0.03
    curvelength = np.sum(np.linalg.norm(g[1:, 12:15] - g[:-1, 12:15], axis=1))
    initial_guess = np.array([0.84141899, 0., 0.98696486, -0.01547761,  0.34352859, 0.392     ])
    optimal_params = initial_guess
    
    def frame_update(frame):
        nonlocal initial_guess
        # Update the plot
        ax.clear()
        
        ax.plot(x_values, y_values, z_values, label="Trajectory")
        # Set plot limits
        max_val_x = np.max(np.abs(g0[:, 12])) + clearance
        max_val_y = np.max(np.abs(g0[:, 13])) + clearance
        ax.set_xlim(-max_val_x, max_val_x)
        ax.set_ylim(-max_val_y, max_val_y)
        ax.set_zlim(0, curvelength + clearance)
        
        # Target pose (update based on the frame)
        target_pose = np.eye(4)
        target_pose[:3, 3] = [x_path_func(frame),
                              y_path_func(frame),
                              z_path_func(frame)]
        target_pose[:3, :3] = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
        
        target_position = target_pose[:3, 3]
        
        # Perform inverse kinematics to find the optimal parameters
        optimal_params = inverse_kinematics(target_pose, initial_guess, pts_per_seg=np.array([10, 10]))
        # print("Optimal IK Parameters:", optimal_params)

        # Extract kappa, phi, ell from optimal_params
        num_segments = len(optimal_params) // 3
        kappa = np.array(optimal_params[:num_segments])
        phi = np.array(optimal_params[num_segments:2*num_segments])
        ell = np.array(optimal_params[2*num_segments:])
        # print(kappa,phi,ell)
        print(q_to_lengths(kappa, phi, ell, radius))
        
        # Update initial_guess with the optimal_params (this makes it the new initial guess for the next frame)
        initial_guess = optimal_params
        
        # Recalculate the robot configuration using updated kappa, phi, ell
        g = robotindependentmapping(np.array(kappa), np.array(phi), np.array(ell), np.array([10]))
        
        # Plot the updated robot configuration
        plot_elements = plot_tf(ax, g, seg_end, tipframe=True, segframe=False, baseframe=True, projections=True, baseplate=True)
        update_plot(plot_elements, g)
        
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
        
        # Plot the target position (no need for orientation here)
        ax.scatter(*target_position, color='red', s=100, label="Target Position")

        # Redraw the plot for the new frame
        plt.draw()

    # Start the animation
    ani = FuncAnimation(fig, frame_update, frames=200, interval=10)
    plt.show()

animate()
# length_to_q(lengths = np.array([[5,6],[4,6],[5,6]]))
