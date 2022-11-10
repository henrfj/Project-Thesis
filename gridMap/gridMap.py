import numpy as np




class CT:
    def __init__(self) -> None:
        self.X = [0, 0, 0, 0]   # State
        self.omega = 0          # Turn rate: given / derived
        


    def discrete_step(self, A):
        self.X = A@self.X

    def omega_known(self):

        A = ...







if __name__ == "__main__":
    coordinated_motion = CT()