import pygame
import math
import numpy as np
from vehicle import Vehicle
import copy
import matplotlib.pyplot as plt
import time

class Visualization:
    """

    """

    def __init__(self, dimentions, robot_img_path, map_img_path) -> None:
        pygame.init()

        # COLORS
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)

        # Map
        self.robot = pygame.image.load(robot_img_path)
        self.map_img = pygame.image.load(map_img_path)

        # Dimensions
        self.height, self.width = dimentions

        # Window settings
        pygame.display.set_caption("Cordinated turn motion model: Vehicle")    # Title
        self.map = pygame.display.set_mode((self.width, self.height))       # Canvas
        self.map.blit(self.map_img, (0, 0)) 

    def draw_robot(self, state, heading, alpha, d, robot : Vehicle = None):
        """  """
        # Extract from state
        x = state[0, 0]
        y = state[1, 0]
        v_x = state[2, 0]
        v_y = state[3, 0]
        v = np.sqrt(v_x**2 + v_y**2)

        rotated = pygame.transform.rotozoom(self.robot, math.degrees(heading), 1) # Rotate robot image to heading
        
        x, y = self.to_pygame(x, y)
        rect = rotated.get_rect(center=(int(x), int(y))) # Bounding rectangle in world coordinates
        self.map.blit(rotated, rect) # Draw roboto onto the map

        heading = -heading # World is flipped aboud x-axis.
        # TODO: Does not rotate about center, but about back axel of vehicle.  
        real_x = x - (d/2)*np.cos(heading)
        real_y = y - (d/2)*np.sin(heading)
        #pygame.draw.circle(self.map, self.red, (real_x, real_y), 3, 0)
        # Draw heading / other useful vectors
        end_x = real_x + 5*v*np.cos(heading)
        end_y = real_y + 5*v*np.sin(heading)
        pygame.draw.line(self.map, self.red, (real_x, real_y), (end_x, end_y), width=3)

        # Draw turn radius
        if np.abs(np.tan(alpha))>1e-7:
            end_x = real_x - d/np.tan(alpha)*np.cos(np.pi/2 + heading)
            end_y = real_y - d/np.tan(alpha)*np.sin(np.pi/2 + heading)
            pygame.draw.line(self.map, self.green, (real_x, real_y), (end_x, end_y), width=3)
        else:
            pass

        # Draw trajectory?
        if robot:
            trajectory_robot = copy.deepcopy(robot) # To use the same parameters
            # Make sure that the state is equal
            trajectory_robot.X = state 
            trajectory_robot.psi = heading
            trajectory_robot.alpha = alpha

            n = 100
            for i in range(n):
                x = trajectory_robot.X[0, 0]
                y = trajectory_robot.X[1, 0]
                traj_x, traj_y = self.to_pygame(x, y)
                pygame.draw.circle(self.map, self.blue, (traj_x, traj_y), 1, 0)
                trajectory_robot.one_step_algorithm(alpha_ref=alpha, v_ref=v)
                
    def draw_static_circogram_data(self, cirgo, objects):
        pass


    def draw_sensor_data(self, point_cloud):
        for point in point_cloud:
            pygame.draw.circle(self.map, self.red, point, 3, 0)

    def to_pygame(self, x, y):
        """Convert coordinates into pygame coordinates (lower-left => top left)."""
        return (x, self.height - y)

if __name__ == "__main__":
    """
    Offline simulator.
    """
    
    # Spawn in 4 cars    
    car1 = Vehicle(np.array([25, 25]), 2, 4, np.pi/2)
    car2 = Vehicle(np.array([20, 28]), 2, 4, np.pi/42)
    car3 = Vehicle(np.array([30, 20]), 2, 4, np.pi/8)
    car4 = Vehicle(np.array([30, 30]), 2, 4, np.pi/5)
    objects = [car1, car2, car3, car4]
    
    # Testing Circogram
    N = 15
    horizon = 50
    x = np.linspace(0, 2*np.pi, N)
    #
    circog = list(car1.static_circogram(N, [car2, car3, car4], horizon))
    d1, d2, P1, P2 = zip(*circog)
    

    MAP_DIMENSIONS = (800, 800)
    gfx = Visualization(MAP_DIMENSIONS, 'small_robot.png', 'test_map_2.png') # Also initializes the display
    # Draw on empty canvas
    gfx.map.blit(gfx.map_img, (0,0))
    pixels_per_unit = 20
    # First draw all vehicles
    for obj in objects:
        for side in obj.sides:
            start_x = side[0][0]*pixels_per_unit
            start_y = side[0][1]*pixels_per_unit
            end_x = side[1][0]*pixels_per_unit
            end_y = side[1][1]*pixels_per_unit
            pygame.draw.line(gfx.map, gfx.red, (start_x, start_y), (end_x, end_y), width=3)

    # Mark the center vehicle
    pygame.draw.circle(gfx.map, gfx.blue, (car1.center[0]*pixels_per_unit, car1.center[1]*pixels_per_unit), 4, 0)
    print(P2)
    for i, ego_points in enumerate(P1):
        # Draw lines between P1 and P2
        if P2[i] is not None:
            start_x = car1.center[0]*pixels_per_unit
            start_y = car1.center[1]*pixels_per_unit
            end_x = P2[i][0]*pixels_per_unit
            end_y = P2[i][1]*pixels_per_unit 

            pygame.draw.line(gfx.map, gfx.blue, (start_x, start_y), (end_x, end_y), width=2)
        #else:
        #    start_x = car1.center[0]*pixels_per_unit
        #    start_y = car1.center[1]*pixels_per_unit
        #    end_x = P1[i][0]*pixels_per_unit
        #    end_y = P1[i][1]*pixels_per_unit 
        #    pygame.draw.line(gfx.map, gfx.red, (start_x, start_y), (end_x, end_y), width=1)

        pygame.draw.circle(gfx.map, gfx.blue, (ego_points[0]*pixels_per_unit, ego_points[1]*pixels_per_unit), 2, 0)
        pygame.draw.line(gfx.map, gfx.red, (car1.center[0]*pixels_per_unit, car1.center[1]*pixels_per_unit), (ego_points[0]*pixels_per_unit, ego_points[1]*pixels_per_unit), width=2)

    # Lines
    #pygame.draw.line(gfx.map, gfx.red, (real_x, real_y), (end_x, end_y), width=3)
    #pygame.draw.circle(gfx.map, gfx.blue, (traj_x, traj_y), 1, 0)

    #
    x = np.linspace(0, 2*np.pi, N)
    plt.title("Car_1")
    plt.scatter(x, d1, c='b', label='Ego perimeter')
    plt.scatter(x, d2, c='r', label='Objects surrounding')
    plt.legend(loc='upper left')
    #plt.savefig('figures/Circogram_graph.pdf')
    #plt.savefig('figures/Circogram_Car_1.png', dpi=300)
    plt.show()

    # Update    
    pygame.display.update()
    time.sleep(1000)