''' Contains the functinality to produce a coarse coding of the mountain car problem 

- This is used as input to train a NN, solving the problem.
- the system is two dimentional. position (x) and speed (x')
'''

import numpy as np


class Coarse:
    '''Contains methods for coarse coding the mountain car problem. Uses uniform tiling to partition the state space.'''
    def __init__(self, pos_range=(-1.2, 0.6), vel_range=(-0.07, 0.07), tilings=4, tiling_dims=10):
        '''initializes the coarse coded "map"'''
        #
        self.pos_range = pos_range  # (min, max) values of position
        self.vel_range = vel_range  # (min, max) values of velocity
        self.n = tilings            # Number of tilings
        self.dims = tiling_dims     # Dimensions of a single tiling

        # The size of the feature_vector
        self.feature_size = tilings * tiling_dims * tiling_dims

        # Create the tilings themselves.
        self.create_partitions()

    def create_partitions(self):
        '''Partitions the 2D space into multiple tilings
        '''
        # X-direction (x-pos)
        D_1 = abs(self.pos_range[0]-self.pos_range[1]) # Range of domain
        O_1 = 0 # Offset of first tiling
        if self.n>1: # If n = 1, then we wont need this variable
            # Using this offset makes sure every single tile covers at least one possible value of the state.
            O_1 = D_1 / self.dims # Slightly less than the width of a tiling.
            step_1 = 1/(self.n-1) * O_1 # The shift from one tiling to the next, in the x-direction

        # Y-direction (x-vel)
        D_2 = abs(self.vel_range[0]-self.vel_range[1]) # range of domain
        O_2 = 0 # Offset of first tiling
        if self.n>1: # If n = 1, then we wont need this variable
            # Using this offset makes sure every single tile covers at least one possible value of the state.
            O_2 = D_2 / self.dims # Slightly less than the width of a tiling.
            step_2 = 1/(self.n-1) * O_2 # The shift from one tiling to the next, in the x-direction

        # Create the first tiling. The others are just shifted copies.
        self.partitions = np.empty((self.n, 4))
        x_lower = self.pos_range[0]
        x_upper = self.pos_range[1]+O_1
        y_lower = self.vel_range[0]
        y_upper = self.vel_range[1]+O_2
        self.partitions[0] = [x_lower, x_upper, y_lower,y_upper]
        
        # Create and add the shifted tilings
        for i in range(1,self.n):
            x_lower-=step_1
            x_upper-=step_1
            y_lower-=step_2
            y_upper-=step_2
            self.partitions[i] = [x_lower, x_upper, y_lower,y_upper]

    def get_coarse_code(self, pos, vel):
        '''Transforms input into a coarse coding of the problem'''
        if pos < self.pos_range[0] or pos > self.pos_range[1] or vel < self.vel_range[0] or vel > self.vel_range[1]:
            raise Exception("Values are outside range.")
        
        # Feature vector to be returned
        feature_vec = np.zeros((self.n, self.dims, self.dims))  # Size is tilings x tiling_dims x tiling_dims
        
        for i in range(len(self.partitions)):
            # Setup for the tile. Variable sized tilings can be used.
            tiling = self.partitions[i] # [x_lower, x_upper, y_lower,y_upper]
            d_1 = abs(tiling[0]-tiling[1]) / self.dims # Size of one tile in x-direction
            d_2 = abs(tiling[2]-tiling[3]) / self.dims # Size of one tile in y-direction
            
            # x-direction
            s = tiling[0] + d_1/100 # Add small offset 
            k = 0
            while s < tiling[1]:
                s += d_1
                if pos <= s:
                    x_index = k
                    break # Exactly one hit per tiling
                k+=1
            
            # y-direction
            s = tiling[2] + d_2/100
            k = 0
            while s < tiling[3]:
                s += d_2
                if vel <= s:
                    y_index = k
                    break # Exactly one hit per tiling
                k+=1

            feature_vec[i][y_index][x_index] = 1

        # print(feature_vec)
        return feature_vec.flatten()

def partition_test():
    pos_range=(-1.2, 0.6)
    vel_range=(-0.07, 0.07)
    tilings=4
    tiling_dims=10

    cc = Coarse(pos_range, vel_range, tilings, tiling_dims)
    print(cc.partitions)

def feature_vector_test():
    pos_range=(-1.2, 0.6)
    vel_range=(-0.07, 0.07)
    tilings=1
    tiling_dims=10

    cc = Coarse(pos_range, vel_range, tilings, tiling_dims)
    
    pos = -0.31
    vel = 0.01
    vec = cc.get_coarse_code(pos, vel)
    print(vec)

if __name__ == '__main__':
    partition_test()
    #feature_vector_test()