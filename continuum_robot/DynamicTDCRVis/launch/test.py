import numpy as np
import matplotlib.pyplot as plt

# Robot state
class RobotState:
    def __init__(self):
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.f_ext = np.zeros(3)

# Impedance control parameters
class ImpedanceControlParams:
    def __init__(self, K, C, M):
        self.K = K
        self.C = C
        self.M = M

# Impedance control function
def impedance_control(robot_state, impedance_params, desired_position, desired_velocity):
    position_error = desired_position - robot_state.x
    velocity_error = desired_velocity - robot_state.v

    force = (np.dot(impedance_params.M, robot_state.a) + 
             np.dot(impedance_params.C, velocity_error) + 
             np.dot(impedance_params.K, position_error))

    # Limit force impact
    force = np.clip(force, -50, 50)
    
    # Compute acceleration
    acceleration = np.dot(np.linalg.inv(impedance_params.M), force + robot_state.f_ext)
    
    return acceleration

# Main loop
def control_loop():
    robot_state = RobotState()
    
    K = np.diag([100, 100, 100])  # Adjusted stiffness
    C = np.diag([30, 30, 30])     # Increased damping
    M = np.diag([1, 1, 1])        # Inertia

    impedance_params = ImpedanceControlParams(K, C, M)

    desired_position = np.array([1.0, 0.75, 0.5])
    desired_velocity = np.array([0.0, 0.0, 0.0])

    positions = []
    times = []
    
    dt = 0.001  # Smaller time step
    for t in range(5000):  
        acceleration = impedance_control(robot_state, impedance_params, desired_position, desired_velocity)
        
        # Apply semi-implicit Euler integration
        robot_state.v += acceleration * dt
        robot_state.x += robot_state.v * dt

        positions.append(robot_state.x.copy())
        times.append(t * dt)

        # External force changes smoothly
        robot_state.f_ext = np.array([0.01 * t, -0.01*t, 40.0])

    # Convert to numpy array
    positions = np.array(positions)

    # Plot position over time
    plt.figure(figsize=(10, 5))
    plt.plot(times, positions[:, 0], label="X Position")
    plt.plot(times, positions[:, 1], label="Y Position")
    plt.plot(times, positions[:, 2], label="Z Position")
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Robot End-Effector Position Over Time')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    control_loop()
