import numpy as np
import matplotlib.pyplot as plt

# Robot state
class RobotState:
    def __init__(self):
        self.x = np.array([0.0, 0.0, 0.0])  # Position
        self.v = np.array([0.0, 0.0, 0.0])  # Velocity
        self.a = np.array([0.0, 0.0, 0.0])  # Acceleration
        self.f_ext = np.array([0.0, 0.0, 0.0])  # External force

# Admittance control parameters
class AdmittanceControlParams:
    def __init__(self, M, C, K):
        self.M = M  # Inertia matrix
        self.C = C  # Damping matrix
        self.K = K  # Stiffness matrix

# Admittance control function
def admittance_control(robot_state, admittance_params, desired_position):
    position_error = robot_state.x - desired_position
    force = np.dot(admittance_params.K, position_error)  # Spring force

    # Compute acceleration from admittance equation
    acceleration = np.dot(np.linalg.inv(admittance_params.M),
                          -np.dot(admittance_params.C, robot_state.v) - force + robot_state.f_ext)
    
    return acceleration

# Main loop
def control_loop():
    robot_state = RobotState()

    # Define admittance parameters (adjust for stability)
    K = np.diag([100, 100, 100])  # Adjusted stiffness
    C = np.diag([30, 30, 30])     # Increased damping
    M = np.diag([1, 1, 1])        # Inertia

    admittance_params = AdmittanceControlParams(M, C, K)

    desired_position = np.array([1.0, 0.75, 0])  # Target position
    positions = []
    times = []

    dt = 0.001  # Small time step for stability

    for t in range(2000):  
        acceleration = admittance_control(robot_state, admittance_params, desired_position)

        # Update motion using semi-implicit Euler integration
        robot_state.v += acceleration * dt
        robot_state.x += robot_state.v * dt

        # Store position for plotting
        positions.append(robot_state.x.copy())
        times.append(t * dt)

        # Simulate an external force (changing over time)
        # robot_state.f_ext = np.array([5.0 * np.sin(t * 0.002), 0.0, 0.0])  # Force applied in X-direction
        robot_state.f_ext = np.array([0.01 * t, -0.01*t, 40.0])

    positions = np.array(positions)

    # Plot movement over time
    plt.figure(figsize=(10, 5))
    plt.plot(times, positions[:, 0], label="X Position")
    plt.plot(times, positions[:, 1], label="Y Position")
    plt.plot(times, positions[:, 2], label="Z Position")
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Admittance Control: Robot End-Effector Position Over Time')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    control_loop()
