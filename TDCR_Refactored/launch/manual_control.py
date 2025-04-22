import sys
import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation

# ==== CONFIGURATION ====
USE_JOYSTICK = True   # switch to True to read from your Joystick class
USE_SERIAL   = False   # switch to True to actually send to motors

if USE_SERIAL:
    ser = serial.Serial('/dev/cu.usbserial-0001', 115200)
    time.sleep(2)  # let it boot

# add project root so imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ==== INPUT HANDLER ====
if USE_JOYSTICK:
    from tools.joystick_command import Joystick
    controller = Joystick()
else:
    from tools.keyboard_commands import KeyboardController
    controller = KeyboardController()

# ==== KINEMATICS & PLOTTING ====
from tools.spring_kinematics import config_to_task, config_to_lengths
from tools.plotting         import setupfigure, plot_tf, update_plot, draw_tdcr
from tools.constants        import (
    kappa_init, phi_init, ell_init,
    kappa_limits_init, phi_limits_init, ell_limits_init,
    num_segments_init, seg_end_init
)

def animate():
    # --- initialize config space: 3 × N_segments
    num_segments = num_segments_init
    config = np.vstack([kappa_init, phi_init, ell_init])

    # compute an initial backbone so we can size the axes
    init_coords = config_to_task(config, pts_per_seg=num_segments)
    total_length = np.sum(ell_init)

    fig, ax = setupfigure(g0=init_coords)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, total_length + 0.05)

    seg_end   = seg_end_init
    prev_tend = None
    prev_time = time.time()
    selected_segment = 0  # 0,1,2 for segments 1-3

    custom_motor_order = [2, 3, 8, 0, 5, 4, 1, 6, 7]  # your mapping

    def frame_update(frame):
        nonlocal config, prev_tend, prev_time, selected_segment

        
        ax.clear()
        ax.set_xlim(-0.2, 0.6)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0, 0.4)
        ax.view_init(elev=30, azim=60)

        # read inputs: axes, buttons, and (optionally) segment key
        axes, buttons, seg_key = controller.get_transmitter_values()
        if seg_key is not None:
            selected_segment = seg_key
        # selected_segment = 0

        # timedelta for smooth steps
        dt = (time.time() - prev_time) / 40
        prev_time = time.time()

        # Δ‑step size (tweak as you like)
        STEP = 0.01

        i = selected_segment
        # map L1 / L2 to κ / φ
        config[0, i] += (axes["TR"]-axes["TL"]) * STEP*20
        config[1, i] += (buttons["KL"] - buttons["KR"]) * STEP*5
        # buttons Y/A to length ℓ
        config[2, i] += (buttons["Y"] - buttons["A"]) * STEP/2

        # clip within your defined limits
        config[0, i] = np.clip(config[0, i], *kappa_limits_init[i])
        config[1, i] = np.clip(config[1, i], *phi_limits_init[i])
        config[2, i] = np.clip(config[2, i], *ell_limits_init[i])

        # compute tendon lengths for *all* segs (you could slice out just three if you prefer)
        radius = [0.0254] * config.shape[1]
        tendons = config_to_lengths(config=config, r=radius)
        flat_L  = [L for seg in tendons for L in seg]

        # if we’ve run once before, send just the 3 tendons of this segment
        if USE_SERIAL and prev_tend is not None:
            start = i*3
            prev_slice = prev_tend[start:start+3]
            curr_slice = flat_L[start:start+3]
            deltas = [(c - p) for c, p in zip(curr_slice, prev_slice)]
            cmds = []
            for mtr, Δ in zip(custom_motor_order[start:start+3], deltas):
                # convert Δ‑length → degrees  
                deg = Δ / 0.0125 * 180 / np.pi
                cmds.append(f"{mtr} {deg:.2f}")
            ser.write((", ".join(cmds) + "\n").encode())

        prev_tend = flat_L

        # replot the backbone
        coords = config_to_task(config, pts_per_seg=num_segments)
        
        plot_elems = plot_tf(ax, coords, seg_end,
                             tipframe=True, segframe=False,
                             baseframe=True, projections=True,
                             baseplate=True)
        update_plot(plot_elems, coords)
        draw_tdcr(ax, coords, seg_end,
                  r_disk=1.5e-2, r_height=1.5e-3,
                  tipframe=True, segframe=False,
                  baseframe=True, projections=False,
                  baseplate=False)

        ax.set_title(f"Manual Config Control — Segment #{selected_segment+1}")
        ax.scatter(0.25, 0.1, 0, color='blue', s=60, label="Object Point")
        ax.plot(
            [.25, 0.25],
            [-.05, 0.05],
            [0, 0],
            color='purple', linewidth=2, linestyle='--', label='Target Line'
        )
        plt.draw()

    ani = FuncAnimation(fig, frame_update, frames=2000, interval=20)
    plt.show()

if __name__ == "__main__":
    animate()
