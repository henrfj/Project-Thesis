import pygame
import math
from limo import LIMO
import numpy as np

class Graphics:

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

    def draw_robot(self, x, y, heading):
        """  """
        rotated = pygame.transform.rotozoom(self.robot, math.degrees(heading), 1) # Rotate robot image to heading
        rect = rotated.get_rect(center=(int(x), int(y))) # Bounding rectangle in world coordinates
        self.map.blit(rotated, rect) # Draw roboto onto the map

    def draw_sensor_data(self, point_cloud):
        for point in point_cloud:
            pygame.draw.circle(self.map, self.red, point, 3, 0)


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


if __name__ == "__main__":
    MAP_DIMENSIONS = (1000, 1000)

    # ROBOT
    X_0 = np.array([[200], [20], [0], [15]]) # Initial state
    limo = LIMO(start_state=X_0)
    # Graphics
    gfx = Graphics(MAP_DIMENSIONS, 'small_robot.png', 'test_map_1.png')

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