import sys
import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation

# ==== CONFIGURATION ====
USE_JOYSTICK = True   # Set to True to use joystick input
USE_SERIAL = False     # Set to True to send motor commands via serial

# Serial setup
if USE_SERIAL:
    ser = serial.Serial('/dev/cu.usbserial-0001', 115200)
    time.sleep(2)  # Allow serial connection to establish

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ==== MODULE IMPORTS ====
if USE_JOYSTICK:
    from tools.joystick_command import Joystick
    controller = Joystick()
else:
    from tools.keyboard_commands import KeyboardController
    controller = KeyboardController()

from tools.spring_kinematics import (
    lengths_to_config, config_to_task, config_to_lengths, inverse_kinematics
)
from tools.plotting import setupfigure, plot_tf, update_plot, draw_tdcr
from tools.constants import (
    object_pos, object_rad, object_height, path_height,
    kappa_init, phi_init, ell_init, kappa_limits_init, phi_limits_init, ell_limits_init, k_values,
    num_segments_init, seg_end_init, phase_duration, total_phases
)

print(f"Current script path: {Path(__file__).resolve().parent.parent}")

# ==== ANIMATION FUNCTION ====
def animate():
    # Initialize configuration space parameters (Config = [kappa; phi; ell])
    config = np.vstack([kappa_init, phi_init, ell_init])
    num_segments = num_segments_init
    # Compute initial backbone (task space) coordinates
    task_coords = config_to_task(config, pts_per_seg=num_segments)
    fig, ax = setupfigure(g0=task_coords)
    seg_end = seg_end_init
    selected_segment = 0

    # Set up the target pose
    target_pose = np.eye(4)
    target_pose[:3, 3] = [sum(ell_init), 0.005, -0.005]
    target_pose[:3, :3] = R.from_euler('xyz', [0, 91, 0], degrees=True).as_matrix()
    ax.scatter(*target_pose[:3, 3], label="Target Trajectory")


    num_segments_extra = np.array([3,3,3])
    seg_end_extra = np.array([4, 8, 12])
    config_extra = np.array([[0], [0], [0.185]])
    

    # Prepare initial guess for inverse kinematics and a placeholder for tendon length changes
    initial_guess = np.concatenate([kappa_init, phi_init, ell_init])
    prev_tendon_lengths = None
    prev_time = time.time()

    def frame_update(frame):
        nonlocal initial_guess, prev_time, prev_tendon_lengths, config, selected_segment, config_extra

        # Clear the plot and update limits
        ax.clear()
        ax.set_xlim(-0.2, 0.6)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0, 0.4)
        ax.view_init(elev=30, azim=60)

        # Get controller values (axes and buttons)
        axes, buttons, seg_key = controller.get_transmitter_values()

        # Update target pose based on input
        dt = (time.time() - prev_time) / 40
        prev_time = time.time()
        kappa_change = (axes["TL"] - axes["TR"]) * dt * 100
        phi_change = (buttons["KL"] - buttons["KR"]) * dt * 50
        ell_change = (buttons["LB"] - buttons["RB"]) * dt

        target_pose[:3, 3] += [(buttons["Y"] - buttons["A"]) * dt, axes["L2"] * dt, axes["L1"] * dt]
        pitch_change = axes["R1"] * dt * 250
        yaw_change   = axes["R2"] * dt * 250
        
        config_extra += np.array([[kappa_change], [phi_change], [ell_change]])

        min_limits = np.array([[-40], [-40], [0.08]])
        max_limits = np.array([[40], [40], [0.185]])

        # Clamp each value to its min/max
        config_extra = np.clip(config_extra, min_limits, max_limits)


        current_rot = R.from_matrix(target_pose[:3, :3])
        delta_rot = R.from_euler('xyz', [0, pitch_change, yaw_change], degrees=True)
        target_pose[:3, :3] = (delta_rot * current_rot).as_matrix()

        ax.scatter(*target_pose[:3, 3], label="Target Trajectory")
        target_position = target_pose[:3, 3]



        # Perform inverse kinematics to update the configuration
        optimal_params = inverse_kinematics(
            target_pose, initial_guess, pts_per_seg=num_segments,
            kappa_limits=kappa_limits_init, phi_limits=phi_limits_init, ell_limits=ell_limits_init, k_values=k_values
        )
        if optimal_params is None:
            return

        initial_guess = optimal_params + np.random.normal(loc=0.0, scale=0.001, size=optimal_params.shape)
        appended_params = np.hstack((optimal_params.reshape((3,2)), config_extra))
        print(appended_params)

        # initial_guess = optimal_params
        num_seg = (len(optimal_params)) // 3
        
        config[0, :] = optimal_params[:num_seg]
        config[1, :] = optimal_params[num_seg:2*num_seg]
        config[2, :] = optimal_params[2*num_seg:]
        print(config)
        

        # Convert Config to tendon lengths
        radius = [0.0254] * (len(ell_init) + 1)
        tendon_lengths = config_to_lengths(config=appended_params, r=radius)
        flat_lengths = [l for seg in tendon_lengths for l in seg]

        # Send motor commands if previous lengths exist
        if prev_tendon_lengths is not None:
            custom_motor_order = [2, 3, 8, 0, 5, 4, 1, 6, 7]
            init_tendon = np.repeat(ell_init, 3)
            delta_lengths = np.round([prev - curr for curr, prev in zip(flat_lengths, init_tendon)], 4)
            motor_commands = ", ".join([
                f"{motor} {(delta / 0.0125 * 180 / np.pi):.2f}"
                for motor, delta in zip(custom_motor_order, delta_lengths)
            ])
            if USE_SERIAL:
                ser.write((motor_commands + "\n").encode())
        prev_tendon_lengths = flat_lengths

        # Recompute backbone coordinates from updated configuration
        task_coords = config_to_task(appended_params, pts_per_seg=num_segments_extra)
        plot_elements = plot_tf(ax, task_coords, seg_end_extra, tipframe=True, segframe=False,
                                baseframe=True, projections=True, baseplate=True)
        update_plot(plot_elements, task_coords)
        # print(config[1,:])
        draw_tdcr(ax, task_coords, seg_end_extra, r_disk=1.5e-2, r_height=1.5e-3,
                  tipframe=True, segframe=False, baseframe=True, projections=False, baseplate=False)
        ax.scatter(*target_position, color='red', s=100, label="Target Position")
        ax.scatter(0.25, 0.1, 0, color='blue', s=60, label="Object Point")
        ax.plot(
            [.25, 0.25],
            [-.05, 0.05],
            [0, 0],
            color='purple', linewidth=2, linestyle='--', label='Target Line'
        )
        plt.draw()

    ani = FuncAnimation(fig, frame_update, frames=200, interval=50)
    plt.show()

if __name__ == "__main__":
    animate()
