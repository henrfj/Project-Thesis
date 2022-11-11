import numpy as np

def distance(point1, point2):
    """ Euclidean distance between two points in the grid"""
    return np.linalg.norm(np.asarray(point1)-np.asarray(point2))


class LIMO:
    """ A general Ackerman driven robot platform """
    
    def __init__(self, start_state=np.array([[0], [0], [0], [0]])) -> None:
        # Intrinsics
        self.m2p = 4000 # Pixels per meter
        self.maxspeed = 2 # m/s
        self.length = 0.5
        self.width = 0.2 # Width in meters

        # Extrinsics
        self.min_obs_dist = 100 # pixels
        self.count_down = 5 # seconds
        #self.dt = dt # Time between discrete simulations
        self.X = start_state
        self.heading = self.get_heading(self.X[2:])
        self.v = np.sqrt(self.X[2]**2 + self.X[3]**2)

    def one_step_alpha(self, alpha, dt):
        """ NB: Extending this code
                - Take in (a, alpha) instead, and make |v| variable.
                - Needs to keep a heading, and use it when |v| goes from zero to nonzero.
            Also:
                - Make code more efficient by not having to recalculate omega, A for repeating value (v, alpha)
        """
        # Step 1: calculate turn rate
        omega = self.v * np.tan(alpha)/self.length

        # Step 2: update state vector
        if omega != 0:
            A = np.array([[1, 0, np.sin(omega*dt)/omega, -(1-np.cos(omega*dt))/omega],
                          [0, 1, (1-np.cos(omega*dt))/omega, np.sin(omega*dt)/omega],
                          [0, 0, np.cos(omega*dt), -np.sin(omega*dt)], 
                          [0, 0, np.sin(omega*dt), np.cos(omega*dt)]], dtype=np.ndarray)
        else: # In case omega == 0; straight line driving
        # NB! If |v| was zero, need to find new heading?

            A = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0], 
                          [0, 0, 0, 1]])
        
        # Step 3: run one step
        self.X = A@self.X

        self.heading = self.get_heading(self.X[2:])

    def kinematic_series(self, alpha, dt, steps):
        # Store the trajectory
        trajectory = np.zeros((4, 1, steps))
        # Step 1: calculate turn rate
        omega = self.v * np.tan(alpha)/self.length
        # Step 2: update state vector
        if omega != 0:
            A = np.array([[1, 0, np.sin(omega*dt)/omega, -(1-np.cos(omega*dt))/omega],
                          [0, 1, (1-np.cos(omega*dt))/omega, np.sin(omega*dt)/omega],
                          [0, 0, np.cos(omega*dt), -np.sin(omega*dt)], 
                          [0, 0, np.sin(omega*dt), np.cos(omega*dt)]], dtype=np.ndarray)
        else: # In case omega == 0; straight line driving
        # NB! If |v| was zero, need to find new heading?

            A = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0], 
                          [0, 0, 0, 1]])
        trajectory[:, :, 0] = self.X

        # Step 3: run a number of updates
        for i in range(steps):
            self.X = A@self.X
            trajectory[:, :, i] = self.X

        return trajectory

    def get_heading(self, v):
        v_u = np.reshape((v / np.linalg.norm(v)), (2,))
        x_u = np.array([1, 0])
        return np.arccos(np.clip(np.dot(v_u, x_u), -1.0, 1.0))


if __name__ == "__main__":
    ### Heading test
    test_heading=True
    if test_heading==True:
        X = np.array([[0], [0], [0], [0.05]])
        heading_test_robot = LIMO(X)
        print(np.rad2deg(heading_test_robot.heading))
        print(heading_test_robot.X)

        X = np.array([[0], [0], [0.5], [0.5]])
        heading_test_robot = LIMO(X)
        print(np.rad2deg(heading_test_robot.heading))
        print(heading_test_robot.X)

        X = np.array([[0], [0], [0.5], [0]])
        heading_test_robot = LIMO(X)
        print(np.rad2deg(heading_test_robot.heading))
        print(heading_test_robot.X)
    
    X = np.array([[0], [0], [0.5], [0.5]])
    robot = LIMO()