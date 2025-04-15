import matplotlib.pyplot as plt
import numpy as np
import cv2
import cv2.aruco as aruco
import pickle
import threading
import time
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path



sys.path.append(str(Path(__file__).resolve().parent.parent))
# --- Import your robotics functions ---
from tools.kinematics import robotindependentmapping, inverse_kinematics, q_to_lengths
from tools.plotting import setupfigure, plot_tf, update_plot, draw_tdcr

# =======================
# Trajectory Definitions
# =======================
x_path_freq = np.pi / 40  
y_path_freq = np.pi / 40  
z_path_freq = np.pi / 4

def x_path_func(t):
    return 0.25 * np.sin(t * x_path_freq)

def y_path_func(t):
    return 0.28 * np.cos(t * y_path_freq)

def z_path_func(t):
    return 0.6 + 0.1 * np.cos(t * z_path_freq)

# ================================
# ArUco Pose Estimation Function
# ================================
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    Estimate rvec and tvec for each detected marker.
    '''
    marker_points = np.array([
        [-marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)
    
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        # The function below returns a flag (unused), rvec and tvec.
        retval, rvec, tvec = cv2.solvePnP(marker_points, c, mtx, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(rvec)
        tvecs.append(tvec)
        trash.append(retval)
    return rvecs, tvecs, trash

# ========================================
# ArUco Tracking Thread (Background Task)
# ========================================
def aruco_tracking_thread(camera_matrix, dist_coeffs, marker_size, stop_event, marker_dict):
    """
    Open a video stream, detect ArUco markers using cv2.aruco and update the provided dictionary.
    """
    cap = cv2.VideoCapture(1)  # change index if needed
    # Create ArUco detector with our dictionary and parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Clear any old marker data
        marker_dict.clear()

        if ids is not None and len(corners) > 0:
            # Use your provided estimation function
            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                # Create a 4x4 transformation matrix from the estimated rvec and tvec
                rmat, _ = cv2.Rodrigues(rvecs[i])
                tf_matrix = np.eye(4)
                tf_matrix[:3, :3] = rmat
                tf_matrix[:3, 3] = tvecs[i].reshape(3)
                marker_dict[marker_id] = tf_matrix

        time.sleep(0.05)  # small delay to reduce CPU usage

    cap.release()

# ================================
# Utility: Draw an ArUco marker
# ================================
def draw_marker(ax, tf_matrix, label="", length=0.03):
    """
    Draws the coordinate axes and a scatter point for the ArUco marker.
    """
    origin = tf_matrix[:3, 3]
    # Compute axis vectors for plotting
    x_axis = tf_matrix[:3, 0] * length
    y_axis = tf_matrix[:3, 1] * length
    z_axis = tf_matrix[:3, 2] * length

    # Draw axes using quiver
    ax.quiver(*origin, *x_axis, color='magenta', arrow_length_ratio=0.1)
    ax.quiver(*origin, *y_axis, color='cyan', arrow_length_ratio=0.1)
    ax.quiver(*origin, *z_axis, color='yellow', arrow_length_ratio=0.1)
    # Mark the origin
    ax.scatter(*origin, color='magenta', s=50)
    ax.text(origin[0], origin[1], origin[2], label, color='magenta')

# ============================================
# Animation Function (Robot & ArUco plotting)
# ============================================
def animate():
    # ----- Robot Initialization -----
    radius = [0.0254, 0.0254]
    kappa, phi, ell = np.array([1, 1]), np.array([0, 0]), np.array([0.392, 0.392])
    g = robotindependentmapping(np.array(kappa), np.array(phi), np.array(ell), np.array([10]))
    g0 = g

    # Setup figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Precompute a full trajectory for display
    t_values = np.linspace(0, 100, 500)
    x_values = x_path_func(t_values)
    y_values = y_path_func(t_values)
    z_values = z_path_func(t_values)
    ax.plot(x_values, y_values, z_values, label="Trajectory", color='blue')

    seg_end = np.array([11, 22])  # example segment indices
    clearance = 0.03
    # (Note: In your original script, curvelength was computed using g's robot coordinates.)
    curvelength = 1  # using a fixed value for z-limit; adjust as needed.
    initial_guess = np.array([0.84141899, 0., 0.98696486, -0.01547761, 0.34352859, 0.392])

    # -----------------------------------------
    # Global dictionary for ArUco marker data.
    # This dictionary will be updated by the tracking thread.
    global aruco_markers
    aruco_markers = {}

    def frame_update(frame):
        nonlocal initial_guess
        ax.clear()
        # Replot the robot trajectory
        ax.plot(x_values, y_values, z_values, label="Trajectory", color='blue')

        # Define axis limits (update as needed)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # ---- Robot Motion update (using inverse kinematics) ----
        target_pose = np.eye(4)
        target_pose[:3, 3] = [x_path_func(frame), y_path_func(frame), z_path_func(frame)]
        target_pose[:3, :3] = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
        target_position = target_pose[:3, 3]

        optimal_params = inverse_kinematics(target_pose, initial_guess, pts_per_seg=np.array([10, 10]))
        num_segments = len(optimal_params) // 3
        kappa = np.array(optimal_params[:num_segments])
        phi = np.array(optimal_params[num_segments:2*num_segments])
        ell = np.array(optimal_params[2*num_segments:])
        initial_guess = optimal_params

        g = robotindependentmapping(kappa, phi, ell, np.array([10]))

        # Plot the robot configuration using your helper functions
        plot_elements = plot_tf(ax, g, seg_end, tipframe=True, segframe=False, baseframe=True, projections=True, baseplate=True)
        update_plot(plot_elements, g)
        draw_tdcr(ax, g, seg_end,
                  r_disk=2.5*1e-2, r_height=1.5*1e-3,
                  tipframe=True, segframe=False, baseframe=True,
                  projections=False, baseplate=False)
        ax.scatter(*target_position, color='red', s=100, label="Target Position")

        # ---- Plot ArUco Markers (if detected) ----
        for marker_id, tf_matrix in aruco_markers.items():
            draw_marker(ax, tf_matrix, label=f"ID {marker_id}")

        ax.legend()
        plt.draw()

    ani = FuncAnimation(fig, frame_update, frames=200, interval=10)
    plt.show()

# ============================
# Main
# ============================
if __name__ == "__main__":
    # -------- Load Camera Calibration --------
    try:
        with open("DynamicTDCRVis/tools/calibration2.pkl", "rb") as f:
            camera_matrix, dist_coeffs = pickle.load(f)
        print("Loaded camera calibration data successfully.")
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        camera_matrix = np.array([[800, 0, 320],
                                  [0, 800, 240],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    marker_size = 0.1  # Adjust based on your marker's physical size (in meters)
    aruco_markers = {}

    # Start ArUco tracking in a separate thread.
    stop_event = threading.Event()
    aruco_thread = threading.Thread(target=aruco_tracking_thread, args=(camera_matrix, dist_coeffs, marker_size, stop_event, aruco_markers))
    aruco_thread.daemon = True
    aruco_thread.start()

    # Run the animation (this includes your robot motion and ArUco marker points)
    animate()

    # When the plot is closed, signal the tracking thread to stop.
    stop_event.set()
    aruco_thread.join()
