import pygame
import math
from limo import LIMO
import numpy as np
import time
import copy

class Graphics:
    """
    NB! Remember how the rendered coordinate frame is!
    
    
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
        pygame.display.set_caption("Cordinated turn motion model: LIMO")    # Title
        self.map = pygame.display.set_mode((self.width, self.height))       # Canvas
        self.map.blit(self.map_img, (0, 0)) 

    def draw_robot(self, state, heading, alpha, d, robot : LIMO = None):
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
                



    def draw_sensor_data(self, point_cloud):
        for point in point_cloud:
            pygame.draw.circle(self.map, self.red, point, 3, 0)

    def to_pygame(self, x, y):
        """Convert coordinates into pygame coordinates (lower-left => top left)."""
        return (x, self.height - y)


class Ultrasonic:
    """ Example of how simple sensors can also be implemented """
    def __init__(self, sensor_range, map) -> None:
        """
        Params:
            - sensor_range: list holding [range meters, range angle]

        
        """
        self.sensor_range = sensor_range
        self.map_width, self.map_height = pygame.display.get_surface().get_size()
        self.map = map
        self.resolution = 10

    def sense_obstacle(self, x, y, heading):
        heading = -heading
        obstacles = []
        x1, y1 = x, y
        # Cone shaped sense areas.
        start_angle = heading - self.sensor_range[1]
        finish_angle = heading + self.sensor_range[1]
        # For all lines
        for angle in np.linspace(start_angle, finish_angle, self.resolution, False):
            x2 = x1 + self.sensor_range[0] * math.cos(angle)
            y2 = y1 - self.sensor_range[0] * math.sin(angle)
            # Sample along the line segment:
            for i in range(0, 100):
                u = i/100
                x = int(x2 * u * x1 * (1-u))
                y = int(y2 * u * y1 * (1-u))
                if 0 < x < self.map_width and 0 < y < self.map_height: # Within map
                    color = self.map.get_at((x, y)) # Get color of this pixel
                    self.map.set_at((x, y), (0, 208, 255))
                    if (color[0], color[1], color[2]) == (0, 0, 0): # Wall detected
                        obstacles.append([x, y])
                        break

        return obstacles


def step_by_step_animation():
    """
    A useful example
    """
    # Robot
    dt = 0.1
    gamma=5e-7
    alpha_max = 1.2
    v_max = 8
    limo = LIMO(dt=dt, gamma=gamma, alpha_max=alpha_max, v_max=v_max)
    # Sensor
    sensor_range = (250, math.radians(40))
    ultra_sensor = Ultrasonic(sensor_range, gfx.map)
    # Game loop
    last_time = pygame.time.get_ticks()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Press x button
                running = False

        # Get timediff
        dt = (pygame.time.get_ticks()-last_time)/1000
        last_time = pygame.time.get_ticks()

        # Draw on empty canvas
        gfx.map.blit(gfx.map_img, (0,0))
        limo.kinematics(alpha=0.01, dt=dt) #Turn in a circle
        gfx.draw_robot(limo.X[0,0], limo.X[1,0], limo.heading)

        point_cloud = ultra_sensor.sense_obstacle(limo.X[0], limo.X[1], limo.heading)
        gfx.draw_sensor_data(point_cloud)

        pygame.display.update()

if __name__ == "__main__":
    """
    Offline simulator.
    """
    # Robot
    dt = 0.1
    gamma=5e-7
    alpha_max = 0.6
    v_max = 45
    d = 50 #Width of vehicle in pixels
    var_alpha = 0.2 #0.4
    var_vel = 10
    robot = LIMO(dt=dt, gamma=gamma, d=d, alpha_max=alpha_max, v_max=v_max, var_alpha=var_alpha, var_vel=var_vel)
    robot.K_a = 0.2


    # Graphics
    MAP_DIMENSIONS = (2000, 2000)
    gfx = Graphics(MAP_DIMENSIONS, 'small_robot.png', 'test_map_2.png') # Also initializes the display

    # Sensor
    sensor_range = (250, math.radians(40))
    ultra_sensor = Ultrasonic(sensor_range, gfx.map)

    # Brownian motion
    steps = 5000
    v_ref = 30       # Initial
    alpha_ref = -0.3    # Initial
    robot.X = np.array([[MAP_DIMENSIONS[0]/4], [MAP_DIMENSIONS[1]/1.5], [0], [0]]) # Move robot to middle
    states, alphas, v_refs, alpha_refs, psis = robot.brownian_motion(steps=steps, v_ref=v_ref, alpha_ref=alpha_ref, r_factor=0.02)
    
    last_time = pygame.time.get_ticks()
    for i in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Press x button
                exit()

        # Get timediff
        delta_t = (pygame.time.get_ticks()-last_time)/1000
        last_time = pygame.time.get_ticks()

        # Draw on empty canvas
        gfx.map.blit(gfx.map_img, (0,0))

        # Get robot pose
        gfx.draw_robot(state=states[:, :, i], heading=psis[i], alpha=alphas[i], d=robot.d, robot=robot)

        #point_cloud = ultra_sensor.sense_obstacle(states[0, 0, i], states[1, 0, i], psis[i])
        #gfx.draw_sensor_data(point_cloud)

        # Apply to display
        pygame.display.update()

        # Add sleep
        # Not optimal, as we do not know how long the processing took above. Assume 0 time...
        # time.sleep(0.005)