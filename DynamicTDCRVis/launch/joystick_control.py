import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import serial
# Replace with the port you want: 
use_serial = 0
if use_serial == 1:
    ser = serial.Serial('/dev/cu.usbserial-0001', 115200)  # Adjust the port and baud rate as needed
time.sleep(2)  # Give it time to connect
# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from matplotlib.animation import FuncAnimation
from tools.joystick_command import Joystick
from tools.kinematics_collision import lengths_to_q, robotindependentmapping, q_to_lengths, inverse_kinematics
from tools.plotting import setupfigure, plot_tf, update_plot, draw_tdcr
from tools.constants import object_pos, object_rad, object_height, path_height
from tools.constants import kappa_init, phi_init, ell_init, ell_limits_init
from tools.constants import phase_duration, total_phases

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
    radius = [0.0254, 0.0254, 0.0254]
    kappa, phi, ell = kappa_init, phi_init, ell_init
    g = robotindependentmapping(np.array(kappa), np.array(phi), np.array(ell), np.array([3,3,3]))
    g0 = g
    fig, ax = setupfigure(g0=g)
    prev_lengths = None


    target_pose = np.eye(4)
    target_pose[:3, 3] = [ell_init[0] + ell_init[1] + ell_init[2], 0.0, 0.0]
    target_pose[:3, :3] = R.from_euler('xyz', [0, 91, 0], degrees=True).as_matrix()  # Target orientation
    ax.scatter(target_pose[0, 3], target_pose[1, 3], target_pose[2,3], label="Trajectory")

    seg_end = np.array([4,8,12])  # Example segment indices
    clearance = 0.03
    curvelength = np.sum(np.linalg.norm(g[1:, 12:15] - g[:-1, 12:15], axis=1))
    initial_guess = np.concatenate([kappa, phi, ell])
    optimal_params = initial_guess
    prev_time = time.time()
    
    def frame_update(frame):
        nonlocal initial_guess
        nonlocal prev_time
        nonlocal prev_lengths

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
        dt = dt/40
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
        optimal_params = inverse_kinematics(target_pose, initial_guess, pts_per_seg=np.array([3, 3, 3]), ell_limits=ell_limits_init)

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
        lengths = q_to_lengths(kappa, phi, ell, radius)
        print("lengths: ", lengths)

        # Flatten to a 1D list
        flat_lengths = [l for segment in lengths for l in segment]

        # If this is the first frame, initialize prev_lengths
        if prev_lengths is None:
            prev_lengths = flat_lengths
            print("Initialized prev_lengths:", prev_lengths)
            return  # Exit early — skip sending deltas this frame

        # Compute change in lengths for each motor
        custom_motor_order = [2, 3, 8, 0, 5, 4, 1, 6, 7]
        init_ell = np.repeat(ell_init,3)

        delta_lengths = np.round([old-new for new, old in zip(flat_lengths, init_ell)],4)
        
        # Reorder the delta_lengths to match the motor wiring

        print("deltas: ", delta_lengths)
        # Convert delta length m to degrees & format for serial
       # Convert to degrees and generate motor command string
        motor_commands = ", ".join([f"{motor} {(delta /  0.0125 * 180 / np.pi):.2f}" 
                            for motor, delta in zip(custom_motor_order, delta_lengths)])
        print(motor_commands)

        # Send to robot
        if use_serial == 1:
            ser.write((motor_commands + "\n").encode())

        # Update previous lengths for next frame
        prev_lengths = flat_lengths
        
        # Recalculate the robot configuration using updated kappa, phi, ell
        g = robotindependentmapping(np.array(kappa), np.array(phi), np.array(ell), pts_per_seg=np.array([3,3,3]))
        
        # Plot the updated robot configuration
        plot_elements = plot_tf(ax, g, seg_end, tipframe=True, segframe=False, baseframe=True, projections=True, baseplate=True)
        update_plot(plot_elements, g)
        
        plot_elements2 = draw_tdcr(ax,
            g, 
            seg_end, 
            r_disk=1.5*1e-2, 
            r_height=1.5*1e-3, 
            tipframe=True, 
            segframe=False, 
            baseframe=True, 
            projections=False, 
            baseplate=False
        )
        
        # Plot the target position (no need for orientation here)
        ax.scatter(*target_position, color='red', s=100, label="Target Position")
        # ax.scatter(0.1, 0.1, 0.6, color="green", s=100)

        # Redraw the plot for the new frame
        plt.draw()

    # Start the animation
    ani = FuncAnimation(fig, frame_update, frames=200, interval=10)
    plt.show()

animate()