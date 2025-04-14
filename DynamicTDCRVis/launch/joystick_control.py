import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import serial
# Replace with the port you want: 
# ser = serial.Serial('/dev/ttyUSB0', 115200)  # Adjust the port and baud rate as needed
time.sleep(2)  # Give it time to connect

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from matplotlib.animation import FuncAnimation
from tools.joystick_command import Joystick
from tools.raw_kinematics import lengths_to_q, robotindependentmapping, q_to_lengths, inverse_kinematics
from tools.raw_plotting import setupfigure, plot_tf, update_plot, draw_tdcr
        

current_path = Path(__file__).resolve().parent.parent
print(f"Current script path: {current_path}")

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

def animate():
    # Define initial robot parameters

    joystick = Joystick()
    radius = [0.0254, 0.0254]
    kappa, phi, ell = np.array([1,1]), np.array([0, 0]), np.array([0.392, 0.392])
    g = robotindependentmapping(np.array(kappa), np.array(phi), np.array(ell), np.array([10]))
    g0 = g
    fig, ax = setupfigure(g0=g)

    target_pose = np.eye(4)
    target_pose[:3, 3] = [0.1, 0.1, 0.7]
    target_pose[:3, :3] = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()  # Target orientation
    ax.scatter(target_pose[0, 3], target_pose[1, 3], target_pose[2,3], label="Trajectory")

    
    seg_end = np.array([11,22])  # Example segment indices
    clearance = 0.03
    curvelength = np.sum(np.linalg.norm(g[1:, 12:15] - g[:-1, 12:15], axis=1))
    initial_guess = np.array([1.21463777, 0.95008398, 0.98696486, -1.04791395,  0.32951001, 0.3903954 ])
    optimal_params = initial_guess
    prev_time = time.time()
    
    def frame_update(frame):
        nonlocal initial_guess
        nonlocal prev_time
        # Update the plot
        ax.clear()
        
        axes, buttons = joystick.get_transmitter_values()
        # print(axes,buttons)
        # Set plot limits
        max_val_x = np.max(np.abs(g0[:, 12])) + clearance
        max_val_y = np.max(np.abs(g0[:, 13])) + clearance
        ax.set_xlim(-max_val_x, max_val_x)
        ax.set_ylim(-max_val_y, max_val_y)
        ax.set_zlim(0, curvelength + clearance)
        
        # Target pose (update based on the controller commands)
        dt = time.time()-prev_time
        prev_time = time.time()
        dt = dt/20
        target_pose[:3, 3] += [axes["L1"]*dt, axes["L2"]*dt, (buttons["Y"]-buttons["A"])*dt]  # Target position
        
        pitch_change = axes["R1"] * dt*300  # Pitch from right stick Y-axis
        yaw_change   = axes["R2"] * dt*300  # Yaw from triggers
        print("Pitch", pitch_change)

        current_rotation = R.from_matrix(target_pose[:3, :3])
        delta_rotation = R.from_euler('xyz', [0.0, pitch_change, yaw_change], degrees=True)
        target_pose[:3, :3] = (delta_rotation*current_rotation).as_matrix()

        
        ax.scatter(target_pose[0, 3], target_pose[1, 3], target_pose[2,3], label="Trajectory")
        target_position = target_pose[:3, 3]
        
        # Perform inverse kinematics to find the optimal parameters
        optimal_params = inverse_kinematics(target_pose, initial_guess, pts_per_seg=np.array([10, 10]))

        # Extract kappa, phi, ell from optimal_params
        num_segments = len(optimal_params) // 3
        kappa = np.array(optimal_params[:num_segments])
        phi = np.array(optimal_params[num_segments:2*num_segments])
        ell = np.array(optimal_params[2*num_segments:])
        print(kappa,phi, ell)
        
        # Update initial_guess with the optimal_params (this makes it the new initial guess for the next frame)
        initial_guess = optimal_params

        ### Send new lengths to robot ###

        # Convert to tendon lengths
        lengths = q_to_lengths(kappa, phi, ell)

        # Flatten to a 1D list
        flat_lengths = [l for segment in lengths for l in segment]

        # If this is the first frame, initialize prev_lengths
        if 'prev_lengths' not in locals():
            prev_lengths = flat_lengths

        # Compute change in lengths for each motor
        delta_lengths = [new - old for new, old in zip(flat_lengths, prev_lengths)]

        # Convert delta length m to degrees & format for serial
        motor_commands = ", ".join([f"{i} {(delta * 1000 / 25 * 360):.2f}" for i, delta in enumerate(delta_lengths)])

        # Send to robot
        # ser.write((motor_commands + "\n").encode())

        # Update previous lengths for next frame
        prev_lengths = flat_lengths
        
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
        ax.scatter(0.1, 0.1, 0.6, color="green", s=100)

        # Redraw the plot for the new frame
        plt.draw()

    # Start the animation
    ani = FuncAnimation(fig, frame_update, frames=200, interval=10)
    plt.show()

animate()