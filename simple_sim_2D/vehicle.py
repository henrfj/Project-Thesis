from object import *
import matplotlib.pyplot as plt

# TODO: The dt should be a global parameter...



class Vehicle(Object):
    def __init__(self, center: np.array, width: int = 1, length: int = 1, # Superparameters
                heading=0, dt = 0.1, gamma=5e-7, alpha_max=1.2, v_max=8):     # Specific parameters
        
        ##########################################################################
        self.heading = heading
        self.length = length
        self.width = width
        self.center = center
        self.vertices_in_VCF = self.verticesVCF(self.length, self.width)
        
        self.originVCF = self.CCTtoWCF(np.array([0.0, -length / 2]))
        self.vertices = self.vertices_VCTtoWCF(self.vertices_in_VCF)


        super().__init__(center, self.vertices)

        ##########################################################################
        """ Current state """
        self.X = np.array([[self.originVCF[0]], [self.originVCF[0]], [0], [0]]) # State vector
        #self.psi = 0    # Heading
        self.alpha = 0  # Wheel angle. Throttle level can be derived from X[2:,:]
        self.omega = 0  # Current turning rate
        self.steering = 0.0
        self.throttle = 0.0

        """ vehicle structure """
        self.d = length # Keep it simple!

        """Dynamics parameters"""
        self.dt = dt
        self.tau_throttle = 1.0 # Parameter used in the throttle model
        self.tau_steering = 1.0 # Parameter used in the steering model
        self.K_a = self.dt / self.tau_steering # Gain parameter
        self.K_v = self.dt / self.tau_throttle # Gain parameter
        self.v_max = v_max      # Parameter used in the throttle model
        self.alpha_max = alpha_max
        self.k_max = 3.0        # Velocity signal parameter
        self.c_max = 0.5        # Steering signal parameter

        """Others"""
        self.gamma = gamma
        self.c = self.alpha_to_inverse_curve()

    @staticmethod
    def verticesVCF(length: float, width: float) -> np.array:

        verticesVCF = np.array([[width/2, 0],
                                [width/2, length],
                                [-width/2, length],
                                [-width/2, 0]])
        return verticesVCF
    
    def alpha_to_inverse_curve(self):
        # Mapping from steering angle alpha, to inverse curve radius c = 1/R.
        return np.tan(self.alpha)/self.d

    def __str__(self):
        # For printing vehicle info
        return ("#########################################\n"
                +"Center: " + str(self.center) 
                + "\nVertices: " + str(self.vertices)
                +"\nLength: " + str(self.length)
                +"\nlines: " + str(self.lines)
                +"\nwidth: " +str(self.width)
                +"\noriginVCF: "+str(self.originVCF)
                +"\n#########################################\n")

    #################################
    ##### Coordinate transforms #####
    #################################
    def CCTtoWCF(self, point: np.array) -> np.array:
        R_w_c = np.array([[np.cos(self.heading), -np.sin(self.heading)],
                          [np.sin(self.heading),  np.cos(self.heading)]])

        return R_w_c@point + self.center
        
    def VCTtoWCF(self, point: np.array) -> np.array:
        R_w_v = np.array([[np.cos(self.heading), -np.sin(self.heading)],
                          [np.sin(self.heading),  np.cos(self.heading)]])

        return R_w_v@point + self.originVCF
    
    def CCTtoVCF(self, point: np.array) -> np.array:
        return point + np.array([0.0, -self.length/2])
     
    def vertices_VCTtoWCF(self, verticesVCF) -> np.array:
        verticesWCF = []
        for vertex in verticesVCF:
            verticesWCF.append(self.VCTtoWCF(vertex))
        return np.asarray(verticesWCF)
    #################################
    #################################
   


    "Methods for the motion model"
    def throttle_model(self, actual_speed: float, throttle: float, tau_throttle: float, deltaT: float) -> float:
        """
        PSEUDOCODE
        Apply the mathematical model written in the notes
        """
        new_speed = 0.0
        return new_speed

    def braking_model(self, actual_speed: float, throttle: float, k_max: float, deltaT: float) -> float:
        """
        PSEUDOCODE
        Apply the mathematical model written in the notes
        """
        new_speed = 0.0
        return new_speed

    # The speed model manage the two cases in which the throttle is bigger (throttle) than zero
    # or smaller (braking) than zero.
    def speed_model(self, actual_speed: float, throttle: float, tau_throttle: float,
                    k_max: float, deltaT: float) -> float:

        """
        PSEUDOCODE
        IF throttle >= 0
            return result from the method "throttle_model()"
        ELSE
            return result from the method "braking_model()"
        """
        if self.throttle >= 0:
            return self.throttle_model(actual_speed, throttle, tau_throttle, deltaT)
        else:
            return self.braking_model(actual_speed, throttle, k_max, deltaT)

    def inverse_curve_radius_model(self, actual_inv_curve_radius: float, steering: float, tau_steering: float,
                                   c_max : float, deltaT: float) -> float:
        """
        PSEUDOCODE
        Apply the mathematical model written in the notes
        """
        new_inv_curve_radius = 0.7
        return new_inv_curve_radius

    def coordinated_turn_model(self, speed: float, steering: float):
        # It modifies the internal variables of the vehicle
        # More specifically the state vector x formed by (originVCF,norm_velocity)

        # Having the speed and the curve radius, we evaluate the new state vector

        """
        PSEUDOCODE
        Apply the mathematical model written in the notes
        """
        new_x = 0.0
        return new_x


    def motion_model(self, actual_speed: float, throttle: float, tau_throttle: float,
                    k_max: float, actual_inv_curve_radius: float, steering: float, tau_steering: float,
                                   c_max : float):
        # It modifies the internal variables of the vehicle
        # More specifically the state vector x formed by (originVCF,norm_velocity)

        """
        PSEUDOCODE
        Evaluate the new speed with the method "speed_model()"
        Evaluate the new inverse radius with the method "inverse_curve_radius_model()"
        Evaluate the new state vector with the method "coordinated_turn_model()", inserting speed and inverse radius
        return the new state vector
        """

        new_x = 0.0
        return new_x

    # Evaluate line equation in a point
    def eval_value_line_eq(self, line: np.array, point: np.array) -> float:
        """
        PSEUDOCODE
        Evaluate the value of the line equation by substituting a point as x and y
        """
        return line[0]*point[0] + line[1]*point[1] + line[2]

    # METHOD 8: Check the intersection between a line and an entity (it works both for an object and a segment)
    def is_line_intersect_entity(self, line: np.array, vertices: np.array) -> bool:

        same_side = []
        reference = self.eval_value_line_eq(line, vertices[0]) >= 0

        for vertex in vertices[1:]:
            # We are going to compare every vertex with the initial one, if they are on the same side of the line,
            # we add a True to the list same_side. Same side it is initialized as a list of one element True, because
            # of course the first element is on the same side of itself.
            same_side.append((self.eval_value_line_eq(line, vertex) >= 0) == reference)

        # if all the vertex_values are positive (or negative), there is NO intersection with the ray, return False.
        # Otherwise return True
        return not all(same_side)

    # METHOD 9: find the point of intersection between two lines, knowing that there is intersection
    def find_intersection_line_line(self, line1: np.array, line2: np.array) -> np.array:
        """
        PSEUDOCODE
        return the point of intersection, that can be found by solving the system of the two equations
        """
        a1 = line1[0]
        b1 = line1[1]
        c1 = line1[2]
        a2 = line2[0]
        b2 = line2[1]
        c2 = line2[2]

        den = a1*b2 - a2*b1
        point_inters = np.array([(b1*c2 - b2*c1)/den, (a2*c1 - a1*c2)/den])
        return point_inters

    def is_up(self, point1: np.array, center: np.array) -> bool:
        return point1[1] > center[1]


    def is_right(self, point1: np.array, center: np.array) -> bool:
        return point1[0] > center[0]


    # METHOD 10: find the point of intersection between a line and a object obj and the disnp.tance between them
    def find_intersection_ray_object(self, line: np.array, obj: Object, angle: float):
        """
        PSEUDOCODE
        IF there is intersection with the object (METHOD 8)
            FOR every side of the object
                IF there is intersection of the line with the side (METHOD 8)
                    Find point of intersection (METHOD 9)
            Keep the closest point of intersection to the vehicle by checking disnp.tance point-point (METHOD 1)
        Return point
        """
        cent = self.center
        inters_points = []
        dist = []
        up = None
        right = None
        if self.is_line_intersect_entity(line, obj.vertices):
            for i in range(len(obj.lines)):
                if self.is_line_intersect_entity(line, obj.sides[i]):
                    loc_point = self.find_intersection_line_line(line, obj.lines[i])

                    # TODO: quadrant problem fix
                    #if (0 < angle < np.pi / 2 and self.is_up(loc_point, cent) and self.is_right(loc_point, cent) or
                    #    np.pi / 2 < angle < np.pi and self.is_up(loc_point, cent) and not self.is_right(loc_point, cent) or
                    #    np.pi < angle < 3 / 2 * np.pi and not self.is_up(loc_point, cent) and not self.is_right(loc_point, cent) or
                    #    3 / 2 * np.pi < angle < 2 * np.pi and not self.is_up(loc_point, cent) and self.is_right(loc_point, cent) or
                    #    angle == 0 and self.is_right(loc_point, cent) or
                    #    angle == np.pi / 2 and self.is_up(loc_point, cent) or
                    #    angle == np.pi and not self.is_right(loc_point, cent) or
                    #    angle == 3 * np.pi / 2 and not self.is_up(loc_point, cent)):

                    inters_points.append(loc_point)
                    dist.append(self.dist_point_point(loc_point, self.center))

        if inters_points:  # if it not empty
            pos_min = dist.index(min(dist))  # find position of the minimum disnp.tance in the list
            hit_point = [inters_points[pos_min], dist[pos_min]]
        else:
            hit_point = None
        return hit_point
        

    # METHOD 11: find the point of intersection between a the circogram ray and the ego vehicle
    def find_intersection_line_ego(self, angle: float, length: float, width: float) -> np.array:
        """
        PSEUDOCODE
        with the angle, evaluate the point via a formula in CCF(center coordinate frame)
        Transform te point from CCF to WCF
        Return the point
        """
        # There could be problems with angle = np.pi/2, check
        if width * abs(np.sin(angle)) < length * abs(np.cos(angle)):
            x = np.sign(np.cos(angle)) * width/2
            hit_pointCCF = np.array([x, x * np.tan(angle)])
        else:
            y = np.sign(np.sin(angle)) * length / 2
            hit_pointCCF = np.array([y / np.tan(angle), y])

        hit_pointWCF = self.CCTtoWCF(hit_pointCCF)

        return hit_pointWCF


    # The circogram is given by N rays that surround uniformly the vehicle, hitting the other objects
    # in the simulation and giving us a disnp.tance from the hitting points.
    # The function returns N pairs (d1, d2), where d1 is the disnp.tance between the center and the perimeter
    # of the ego (on the circogram ray) and d2 is the disnp.tance between the center of the ego
    # and the hitting point of the ray.
    def static_circogram(self, N: int, list_objects_simul, d_horizon: float):
        """
        PSEUDOCODE
        Create empty list "circogram_list"
        FOR every ray of the circogram
            Check the point of intersection of the line with the perimeter of the ego (P1)(METHOD 11)
            Find the line that describes the ray (having center point C and P1)(self.eval_line_point_point)
            FOR every vehicle close to the ego vehicle
                Find hitting point P2 (METHOD 10)
                Keep closest point P2
            Evaluate the disnp.tances d1 (segment CP1) and d2 (segment CP2) with METHOD 1
            Add the pair (d1,d2) to the list "circogram_list"
        return "circogram_list"
        """
        dist_center_P1 = []
        dist_center_P2 = []
        P1 = []
        P2 = []

        for n in range(N):
            list_hit_point = []
            dist = []

            angle = 2*np.pi/N * n

            # Find P1 and its disnp.tance from the center
            P1.append(self.find_intersection_line_ego(angle, self.length, self.width))
            dist_center_P1.append(self.dist_point_point(P1[n], self.center))

            ray_line = self.eval_line_point_point(self.center, P1[n])
            # Find P2 and its disnp.tance from the center
            for obj in list_objects_simul:
                hit_point = self.find_intersection_ray_object(ray_line, obj, angle)
                if hit_point is not None:
                    list_hit_point.append(hit_point[0])
                    dist.append(hit_point[1])

            if list_hit_point:  # if it not empty
                pos_min = dist.index(min(dist))  # find position of the minimum disnp.tance in the list
                P2.append(list_hit_point[pos_min])
                dist_center_P2.append(dist[pos_min])
            else:
                P2.append(None)
                dist_center_P2.append(d_horizon)
        # return the circogram function, which is a list of N tuple composed by [dist_center_P1, dist_center_P2, P1, P2]
        circogram = zip(dist_center_P1, dist_center_P2, P1, P2)
        return circogram


if __name__ == "__main__":
    #car2 = Vehicle(np.array([5, 4]), 2, 4, np.pi/2)
    #print("car2:\n", car2)

    # Spawn in 4 cars    
    car1 = Vehicle(np.array([1, 1]), 2, 4, np.pi/2)
    car2 = Vehicle(np.array([5, 4]), 2, 4, np.pi/2)
    car3 = Vehicle(np.array([-4, 6]), 2, 4, np.pi/2)
    car4 = Vehicle(np.array([-4, -4]), 2, 4, np.pi/2)

    list_cars = [car2, car3, car4]
    line1 = np.array([2, -1, 1.9])
    line2 = np.array([1, -1, 0])
    line3 = np.array([3/2, -1, -3])

    #car1.print_vehicle()
    print("car2:\n", car2)
    # Testing method 8
    if car1.is_line_intersect_entity(line2, car1.vertices):
        print("There is intersection!")
    else:
        print("There is NO intersection!")

    # Testing method 9
    print("The intersection is: ", car1.find_intersection_line_line(line1, line2))

    # Testing method 10
    print("The intersection ray object is: ", car1.find_intersection_ray_object(line3, car2, np.pi/3))

    # Testing method 10
    print("The intersection line ego vehicle is: ", car1.find_intersection_line_ego(np.pi/2, 4, 2))

    # CAR 1
    plt.xlim((-15, 15))
    plt.ylim((-15, 15))
    plt.plot(car1.center[0], car1.center[1], "r.")
    plt.plot(car1.vertices[:,0], car1.vertices[:,1], "b.")

    # CAR 2
    plt.plot(car2.center[0], car2.center[1], "rs")
    plt.plot(car2.vertices[:,0], car2.vertices[:,1], "b.")

    # CAR 3
    plt.plot(car3.center[0], car3.center[1], "ro")
    plt.plot(car3.vertices[:,0], car3.vertices[:,1], "b.")

    # CAR 4
    plt.plot(car4.center[0], car4.center[1], "r+")
    plt.plot(car4.vertices[:,0], car4.vertices[:,1], "b.")
    plt.show()
    
    
    
    
    # Testing Circogram
    N = 70
    horizon = 10
    x = np.linspace(0, 2*np.pi, N)
    #
    circog = list(car1.static_circogram(N, [car2, car3, car4], horizon))
    d1, d2, P1, P2 = zip(*circog)
    plt.title("Car_1")
    plt.scatter(x, d1, c='b', label='Ego perimeter')
    plt.scatter(x, d2, c='r', label='Objects surrounding')
    plt.legend(loc='upper left')
    #plt.savefig('figures/Circogram_graph.pdf')
    plt.savefig('figures/Circogram_Car_1.png', dpi=300)
    plt.show()
    
    #
    circog = list(car2.static_circogram(N, [car1, car3, car4], horizon))
    d1, d2, P1, P2 = zip(*circog)        
    plt.title("Car_2")
    plt.scatter(x, d1, c='b', label='Ego perimeter')
    plt.scatter(x, d2, c='r', label='Objects surrounding')
    plt.legend(loc='upper left')
    #plt.savefig('figures/Circogram_graph.pdf')
    plt.savefig('figures/Circogram_Car_2.png', dpi=300)
    plt.show()
    #
    circog = list(car3.static_circogram(N, [car1, car2, car4], horizon))
    d1, d2, P1, P2 = zip(*circog)    
    plt.title("Car_3")
    plt.scatter(x, d1, c='b', label='Ego perimeter')
    plt.scatter(x, d2, c='r', label='Objects surrounding')
    plt.legend(loc='upper left')
    #plt.savefig('figures/Circogram_graph.pdf')
    plt.savefig('figures/Circogram_Car_3.png', dpi=300)
    plt.show()
    #
    circog = list(car4.static_circogram(N, [car1, car2, car3], horizon))
    d1, d2, P1, P2 = zip(*circog)
    plt.title("Car_4")
    plt.scatter(x, d1, c='b', label='Ego perimeter')
    plt.scatter(x, d2, c='r', label='Objects surrounding')
    plt.legend(loc='upper left')
    #plt.savefig('figures/Circogram_graph.pdf')
    plt.savefig('figures/Circogram_Car_4.png', dpi=300)
    plt.show()
    
