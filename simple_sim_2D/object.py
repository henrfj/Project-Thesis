"""
The class for a general object: obstacle or vehicle, in the environment
Some basic ideas:
    - All coordinates for objects given in world coordinates
    - n-vertices for a general shape, not neccesarely a square (Initially we have the square assumption)
"""


import numpy as np
from numba import jit

class Object:
    def __init__(self, center: np.array, vertices: np.array,
                     #orientation: float #
                     ):
        #self.orientation = orientation
        self.center = center
        self.vertices = vertices
        self.radius = self.eval_radius()
        #self.sides = self.get_sides(self.vertices)
        self.sides = [[vertices[0], vertices[1]],
                      [vertices[1], vertices[2]],
                      [vertices[2], vertices[3]],
                      [vertices[3], vertices[0]]]
        self.lines = self.eval_lines(self.sides)

    @staticmethod
    def get_sides(vertices):
        sides = []
        for i in range(len((vertices))):
            start = i
            end   = i+1
            if end==len(vertices):
                end = 0
            sides.append([vertices[start], vertices[end]])
        return np.array(sides)

    # TODO: move to Vehicle class
    #"""Transformation from Center Coordinate Frame to World Coordinate Frame """
    #def CCTtoWCF(self, point: np.array) -> np.array:
    #    return point.rotate_rad(self.orientation-np.pi/2) + self.center
    #    
    #"""Transformation from Vehicle Coordinate Frame to World Coordinate Frame """
    #def VCTtoWCF(self, point: np.array) -> np.array:
    #    #print(point)
    #    trans_point = point.rotate_rad(self.orientation - np.pi / 2) + self.originVCF
    #    #print(trans_point)
    #    return trans_point
    #
    #"""Transformation from Vehicle Coordinate Frame to World Coordinate Frame """
    #def CCTtoVCF(self, point: np.array) -> np.array:
    #    trans_point = point
    #    return point


    def eval_radius(self) -> float:
        """Evaluate radius of circle surrounding vehicle"""
        # given the initial shape of the vehicle, that for now we consider as rectangular, we evaluate the radius.
        radius = self.dist_point_point(self.center, self.vertices[0])
        return radius


    def eval_lines(self, sides):
        """Evaluate lines passing throught vertices"""
        # evaluate the lines passing through the vertices and insert them in a list. every line is expressed as an
        # array with 3 values [a,b,c]
        lines = []
        for side in self.sides:
            lines.append(self.eval_line_point_point(side[0], side[1]))
        return lines

    def eval_line_point_point(self, point1: np.array, point2: np.array) -> np.array:
        # evaluate the line passing throught two points, expressing it as a vector with 3 values [a,b,c]
        delta_x = point1[0] - point2[0]
        if delta_x != 0: # Check if the line is vertical, otherwise there is a division by 0
            delta_y = point1[1] - point2[1]
            m = delta_y / delta_x
            c = (delta_x * point1[1] - delta_y * point1[0]) / delta_x
            line = np.array([m, -1, c])
        else:
            line = np.array([1, 0, - point1[0]])  # The equation of a vertical line
        return line

    ###############################################
    ### Methods useful for evaluation distances ###
    ###############################################
    def dist_point_point(self, point1: np.array, point2: np.array) :
        """ METHOD 1: The distance between two points """
        #distance = np.sqrt((point1[0]-point2[0])*(point1[0]-point2[0]) + (point1[1]-point2[1])*(point1[1]-point2[1]))
        return np.linalg.norm(point1 - point2)

    def dist_point_line(self, line: np.array, point: np.array):
        # METHOD 2: The distance between a line (defined in the normal form $a x+b y + c = 0$) and a point
        distance = abs(line[0]*point[0] + line[1]*point[1] + line[2]) / np.sqrt(line[0]*line[0] + line[1]*line[1])
        return distance

    
    def find_point_projection(self,line: np.array, point: np.array):
        # METHOD 3: find the closest point P on a given line $a x+b y + c = 0$ (which is the projection)
        # to a given point A ($x_0,y_0$)
        # One formula for doing it
        a = line[0]
        b = line[1]
        c = line[2]
        x = b / a * (a * a * point[1] + a * b * point[0] - b * c) / (b * b + a * a) - c / a
        y = (a * a * point[1] + a * b * point[0] - b * c) / (b * b + a * a)
        projection = np.array(x, y)
        return projection

    def is_point_in_segment(self, pointP: np.array, pointA: np.array, pointB: np.array) -> bool:
        # METHOD 4: find if a point P($x_0,y_0$) on a line stands between two other points A($x_1,y_1$)
        # and B($x_2,y_2$) on  same line
        return self.dist_point_point(pointA,pointP) + self.dist_point_point(pointB,pointP) == self.dist_point_point(pointA,pointB)

    
    def is_point_in_segment_shaodow(self, pointH: np.array, pointA: np.array, pointB: np.array, line: np.array) -> bool:
        # METHOD 5: find if a point H has its projection between two other points A and B
        pointP = self.find_point_projection(line, pointH)

        if self.is_point_in_segment(pointP, pointA, pointB):
            return True
        else:
            return False

    def dist_point_segment(self, pointH: np.array, pointA: np.array, pointB: np.array, line: np.array) -> float:
        # METHOD 6: distance of a point to a segment
        if self.is_point_in_segment_shaodow(pointH, pointA, pointB, line):
            distance = self.dist_point_line(line, pointH)
        else:
            distance = min(self.dist_point_point(pointH, pointA), self.dist_point_point(pointH, pointB))
        return distance

    def dist_point_object(self, pointH: np.array, obj) -> float:
        # METHOD 7: distance of a point to an Obj
        # TODO
        points_obj = obj.vertices
        lines_obj = obj.lines
        for lin in obj.lines:
            self.dist_point_segment(pointH,) # IT HAS TO BE COMPLETED
        distance = 5
        return distance

    
    def FSdist(self, radius1: float, radius2: float, center1: np.array, center2: np.array) -> float:
        # This is the far-sighted distance, which is approximation of the distance. Very fast to evaluate
        distance = self.dist_point_point(center1, center2) - radius1 - radius2
        return distance

    def NSdist(self, obj) -> float:
        # This is the near-sighted distance, which is the precise distance.
        # TODO
        distance = 5
        return distance


if __name__ == "__main__":
    """
    Some testing of the OBJECT class.
    """

    obj = Object(np.array([5, 5,]), vertices=np.array([[2, 2],
                                                       [1, 4], 
                                                       [2, 8],
                                                       [8, 8], 
                                                       [8, 2]]))
    print("Center", obj.center)

    print("Vertices:", obj.vertices.shape, "\n", str(obj.vertices))
    
    print("Sides:", obj.sides.shape, "\n", str(obj.sides))

    print("Radius:", obj.radius)

    print("Lines", obj.lines)