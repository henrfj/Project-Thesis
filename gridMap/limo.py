import numpy as np
import matplotlib.pyplot as plt
import time

def distance(point1, point2):
    """ Euclidean distance between two points in the grid"""
    return np.linalg.norm(np.asarray(point1)-np.asarray(point2))


# TODO: 
# 1. When speed approaches zero, the turn rate explodes to meet the desired turn  


class LIMO:
    """ A general Ackerman driven robot platform.
     Modelled using simple coordinated turm motion. """
    
    def __init__(self, dt, gamma=5e-7, alpha_max=1.2, v_max=8, d=0.5, var_alpha=0.01, var_vel=0.5) -> None:
        # Intrinsics
        self.v_max = v_max
        self.alpha_max = alpha_max
        self.var_alpha=var_alpha
        self.var_vel=var_vel
        self.d = d # Distance between front and back axels

        # Extrinsics
        self.X = np.array([[0], [0], [0], [0]]) # State vector
        self.psi = 0    # Heading
        self.alpha = 0  # Wheel angle. Throttle level can be derived from X[2:,:]
        self.omega = 0  # Current turning rate

        # Tunable parameters
        self.K_a = 0.2 # Wheel angle update rate
        self.K_v = 0.2 # Wheel 
        self.dt = dt # Time between discrete simulations
        self.gamma = gamma # division tolerance

        #self.Tau_v = (1/5)
        #self.K_v = self.dt / (self.dt + self.Tau_v)

        """ NB! 1. order system
            - Right now, the steering angle as well as the speed is modelled as 1. order systems
            - The consequence is that K = dt/(dt +Tau) ~ dt/Tau in case Tau is at least 10x bigger than dt.
            - Tau is the system time constant (=m/b for mass friction systems), 5xTau is the time before 99.3% of value is reached.
        """

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

    def brownian_motion(self, steps = 1000, v_ref = 3, alpha_ref = 0.0, r_factor=0.02):
        """
        Random walk algorithm.
        """
        # Book keeping
        states = np.zeros((4, 1, steps))
        alphas = np.zeros((steps,))
        v_refs = np.ones((steps,))*v_ref
        alpha_refs = np.ones((steps,))*alpha_ref
        psis = np.zeros((steps,))

        for i in range(steps):
            # Random driver:
            if (i*self.dt).is_integer: # Checks only on whole seconds
                if np.random.choice(a=[0,1], p=[1-r_factor, r_factor]): # Random motion has occured

                    """
                    # NB! using this method is bugged,
                    #  and its to big to be added one way, it can still break it in the other direction...
                    v_rand = np.random.normal(loc=0, scale=self.var_vel)
                    alpha_rand = np.random.normal(loc=0, scale=self.var_alpha)
                    """
                    choices = np.random.randint(0, 2, 2)

                    if choices[0]:
                        v_rand = -self.var_vel
                    else: 
                        v_rand = self.var_vel

                    if choices[1]:
                        alpha_rand = -self.var_alpha
                    else:
                        alpha_rand = self.var_alpha

                    if (v_ref+v_rand) > self.v_max or (v_ref+v_rand) <= 0:
                        v_ref = v_ref - v_rand
                    else:
                        v_ref = v_ref + v_rand
                    
                    if (alpha_ref + alpha_rand) > self.alpha_max or (alpha_ref + alpha_rand) <= -self.alpha_max:
                        alpha_ref = alpha_ref - alpha_rand
                    else:
                        alpha_ref = alpha_ref + alpha_rand
                    
                    
            # Book-keeping
            alphas[i] = self.alpha
            alpha_refs[i] = alpha_ref
            v_refs[i]= v_ref
            states[:, :, i] = self.X
            psis[i] = self.psi
            self.one_step_algorithm(alpha_ref=alpha_ref, v_ref=v_ref)


        return states, alphas, v_refs, alpha_refs, psis

if __name__ == "__main__":
    ###############################
    ### Testing Brownian motion ###
    ###############################
    # Robot
    dt = 0.1
    gamma=5e-7
    alpha_max = 0.8
    v_max = 5
    d=10
    var_alpha= 0.2 # 0.3
    var_vel= 0.5 # 0.5
    robot = LIMO(dt=dt, gamma=gamma, d=d, alpha_max=alpha_max, v_max=v_max, var_alpha=var_alpha, var_vel=var_vel)

    # Brownian motion
    steps = 10000
    v_ref = 0       # Initial
    alpha_ref = 0   # Initial
    #robot.K_a = 1.5
    before_time = time.time()
    states, alphas, v_refs, alpha_refs, psis = robot.brownian_motion(steps=steps, v_ref=v_ref, alpha_ref=alpha_ref, r_factor=0.01)
    print("Time:", time.time()-before_time)


    # Plotting
    time_axis = np.linspace(0, steps*dt-dt, steps)
    Xs = states[0, :].T
    Ys = states[1, :].T
    Vs = states[2:, :]
    abs_speeds = np.linalg.norm(Vs.T, axis=2)

    plt.title("Speeds")
    plt.plot(time_axis, abs_speeds, label="Real")
    plt.plot(time_axis, v_refs, label="Reference")
    plt.xlabel("Time[s]")
    plt.ylabel("Speed [m/s]")
    plt.legend()
    plt.show()

    plt.title("Steering angles (alpha)")
    plt.xlabel("Time[s]")
    plt.ylabel("Steering angle [rad]")
    plt.plot(time_axis, alphas, label="Real")
    plt.plot(time_axis, alpha_refs, label="Reference")
    plt.legend()
    plt.show()

    plt.plot(time_axis, Xs)
    plt.title("X pos over time") 
    plt.xlabel("Time[s]")
    plt.ylabel("X pos[m]")
    plt.show()

    plt.plot(time_axis, Ys)
    plt.title("Y pos over time") 
    plt.xlabel("Time[s]")
    plt.ylabel("Y pos[m]")
    plt.show()

    plt.plot(Xs, Ys)
    plt.title("Trajectory in the plane") 
    plt.xlabel("X pos [m]")
    plt.ylabel("Y pos [m]")
    plt.show()

    plt.plot(time_axis, psis)
    plt.title("Heading over time") 
    plt.xlabel("Time [s]")
    plt.ylabel("Heading[rad]")
    plt.show()
