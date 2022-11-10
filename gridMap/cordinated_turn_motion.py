import numpy as np
import matplotlib.pyplot as plt

class CT:
    def __init__(self) -> None:
        self.X = np.array([[0], [0], [0], [0]])   # State
        
    def discrete_step(self, A):
        self.X = A@self.X

    def omega_known(self, omega, T, steps):
        """ Omega is either given or derived from speed and steering angle """
        states = np.zeros((4, 1, steps))

        if omega != 0:
            A = np.array([[1, 0, np.sin(omega*T)/omega, -(1-np.cos(omega*T))/omega],
                          [0, 1, (1-np.cos(omega*T))/omega, np.sin(omega*T)/omega],
                          [0, 0, np.cos(omega*T), -np.sin(omega*T)], 
                          [0, 0, np.sin(omega*T), np.cos(omega*T)]])
        else: # In case omega == 0; straight line driving
            A = np.array([[1, 0, T, 0],
                          [0, 1, 0, T],
                          [0, 0, 1, 0], 
                          [0, 0, 0, 1]])
        states[:, :, 0] = self.X
        for i in range(steps):
            self.discrete_step(A)
            states[:, :, i] = self.X

        return states


if __name__ == "__main__":
    
    """ If omega = 0, the A matrix described in the paper will devide by zero... """
    omega = 0.2
    T = 1
    A = np.array([[1, 0, np.sin(omega*T)/omega, -(1-np.cos(omega*T))/omega],
                      [0, 1, (1-np.cos(omega*T))/omega, np.sin(omega*T)/omega],
                      [0, 0, np.cos(omega*T), -np.sin(omega*T)], 
                      [0, 0, np.sin(omega*T), np.cos(omega*T)]])
    X = np.array([[0], [0], [1], [1]])
    #print(A)
    #print(X)
    #print(A@X)
    """ Testing for single turn angle """
    omega = 0.4
    T = 0.1
    steps = 100

    coordinated_turn = CT()
    coordinated_turn.X = np.array([[0], [0], [1], [1]])
    states = coordinated_turn.omega_known(omega=omega, T=T, steps=steps)
    print("states:", states.shape)

    plt.plot(states[0, 0, :], states[1, 0, :])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory using omega="+str(omega)+", T="+str(T)+", steps="+str(steps))
    plt.show()