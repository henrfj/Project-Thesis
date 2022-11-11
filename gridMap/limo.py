import numpy as np

def distance(point1, point2):
    """ Euclidean distance between two points in the grid"""
    return np.linalg.norm(np.asarray(point1)-np.asarray(point2))


class LIMO:
    """ A general Ackerman driven robot platform.
     Modelled using simple coordinated turm motion. """
    
    def __init__(self, dt, gamma) -> None:
        # Intrinsics
        self.maxspeed = 2 # m/s
        self.max_wheel_angle = 1.2 # About 66 degrees
        self.d = 0.5 # Distance between front and back axels

        # Extrinsics
        self.X = np.array([[0], [0], [0], [0]])
        self.psi = 0    # Heading
        self.alpha = 0  # Wheel angle. Throttle level can be derived from X[2:,:]
        self.omega = 0  # Current turning rate

        # Tunable parameters
        self.K_a = 0.1 # Wheel angle update rate
        self.K_v = 0.1 # Wheel 
        self.dt = dt # Time between discrete simulations
        self.gamma = gamma

    def one_step_algorithm(self, alpha_ref, v_ref):
        """
        Running one step of the update algorithm.
        """
        # Step 1: update u, alpha, find B_k
        self.alpha = self.alpha + self.K_a * (alpha_ref - self.alpha)
        v_k = self.X[2:,:] # Current speed from state matrix
        u_k = self.K_v * (np.linalg.norm(v_ref)-np.linalg.norm(v_k))
        B_k = np.array([[0],
                        [0], 
                        [np.cos(self.psi)], 
                        [np.sin(self.psi)]])
    
        # Step 2: Choose A_k
        if np.abs(self.omega) > self.gamma:
            A_k = np.array([[1, 0, np.sin(self.omega*self.dt)/self.omega, -(1-np.cos(self.omega*self.dt))/self.omega],
                          [0, 1, (1-np.cos(self.omega*self.dt))/self.omega, np.sin(self.omega*self.dt)/self.omega],
                          [0, 0, np.cos(self.omega*self.dt), -np.sin(self.omega*self.dt)], 
                          [0, 0, np.sin(self.omega*self.dt), np.cos(self.omega*self.dt)]], dtype=np.ndarray)
        else: # In case omega ~ 0
            A_k = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0], 
                          [0, 0, 0, 1]])
        
        # Step 3: Update state matrix
        self.X = A_k@self.X + B_k*u_k

        # Step 4: Update omega nad psi (based on theta)
        theta_next = np.linalg.norm(v_k)*self.dt*np.tan(self.alpha) / self.d # Turn angle
        self.psi = self.psi + theta_next # Heading in WC
        self.omega = np.linalg.norm(v_k)*np.tan(self.alpha) / self.d


if __name__ == "__main__":
    ### Testing single steps
    robot = LIMO(dt=0.1, gamma=5e-7)
    steps = 1000
    v_ref = 1
    alpha_ref = 0
    states = np.zeros((4, 1, steps))
    alphas = np.zeros((steps,))

    for i in range(100):
        alphas[i] = robot.alpha
        states[:, :, i] = robot.X
        robot.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)