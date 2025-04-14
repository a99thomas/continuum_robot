import cv2
import cv2.aruco as aruco
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_aruco_transform_matrix(camera_matrix, dist_coeffs, marker_size=0.0035):
    """
    Continuously track ArUco markers and display their transformation matrices in live video.
    
    Args:
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Camera distortion coefficients (1x5)
        marker_size: Size of the ArUco marker in meters
    """
    # Initialize video capture
    cap = cv2.VideoCapture(1)
    
    # Set a larger window size
    cv2.namedWindow('ArUco Marker Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ArUco Marker Detection', 1280, 720)
    
    # Create ArUco detector with the new API
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    # Setup matplotlib 3D plot
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers using the new API
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Clear previous transformation matrices
        transformation_matrices = {}
        
        if ids is not None:
            # Estimate pose for each detected marker - NEW API
            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, dist_coeffs)
            
            # Draw axis and info for each marker
            for i in range(len(ids)):
                frame = aruco.drawDetectedMarkers(frame, corners, ids)
                frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                                        rvecs[i], tvecs[i], marker_size/2)
                
                # Convert rotation vector to rotation matrix
                rmat, _ = cv2.Rodrigues(rvecs[i])
                
                # Create 4x4 transformation matrix
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rmat
                transformation_matrix[:3, 3] = tvecs[i].reshape(3)
                
                # Store the matrix with marker ID as key
                marker_id = ids[i][0]
                transformation_matrices[marker_id] = transformation_matrix
                
                # Display position information on the frame
                position = tvecs[i][0]
                rotation = rvecs[i][0]
                # text = f"ID {marker_id}: Pos({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
                # cv2.putText(frame, text, (10, 30 + i*30), 
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            # Clear the previous plot
        # ax.cla()
        # ax.set_title("ArUco Marker Positions (Camera Frame)")
        # ax.set_xlabel("X (m)")
        # ax.set_ylabel("Y (m)")
        # ax.set_zlabel("Z (m)")
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(0, 2)
        
        # for marker_id, matrix in transformation_matrices.items():
        #     pos = matrix[:3, 3]
        #     ax.scatter(pos[0], pos[1], pos[2], label=f"ID {marker_id}")
        #     ax.text(pos[0], pos[1], pos[2], f"{marker_id}", fontsize=8)
        
        # ax.legend()
        # plt.draw()
        # plt.pause(0.001)

        
        # Display the frame
        cv2.imshow('ArUco Marker Detection', frame)
        
        # Print transformation matrices to console if any markers detected
        if transformation_matrices:
            print("\n" + "="*50)
            for marker_id, matrix in transformation_matrices.items():
                print(f"\nMarker ID {marker_id} transformation matrix (camera to marker):")
                print(matrix)
            print("="*50 + "\n")
        
        # Exit on 'q' key press
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the current frame
            cv2.imwrite(f"aruco_detection_{cv2.getTickCount()}.png", frame)
            print("Frame saved!")
    
    cap.release()
    cv2.destroyAllWindows()

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

if __name__ == "__main__":
    # Load camera calibration
    try:
        with open("DynamicTDCRVis/tools/calibration2.pkl", "rb") as f:
            camera_matrix, dist_coeffs = pickle.load(f)
        print("Successfully loaded camera calibration data")
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        print("Using default values as fallback")
        camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    # Start tracking
    print("Starting ArUco marker tracking...")
    print("Press 'q' to quit, 's' to save current frame")
    get_aruco_transform_matrix(camera_matrix, dist_coeffs, marker_size=0.1)