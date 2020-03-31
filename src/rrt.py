#!/usr/bin/env python3
"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np


def static_var(**kwargs):
    """
    This function creates decorator. Used for counting recursive calls.
    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

class RRT_utils:
    """
    Class for util. function for  informed RRTstar planning
    """
    def bresenham(self, x1, y1, x2, y2):
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end
        """
        # Setup initial conditions
        dx = (x2 - x1)
        dy = (y2 - y1)
    
        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)
    
        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
    
        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
    
        # Recalculate differentials
        dx = (x2 - x1)
        dy = (y2 - y1)
    
        # Calculate error
        error = int(dx / 2.0)
        ystep = self.map.info.resolution if y1 < y2 else -self.map.info.resolution
        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in np.arange(x1, x2, self.map.info.resolution):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
    
        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points

    def get_grid_cells(self, nodes):
        res = self.map.info.resolution
        grid_cells = []
        for node in nodes:
            x = node[0]/res
            y = node[1]/res
            grid_cells.append(self.grid[ int(round(x)), int(round(y)) ])
        return grid_cells

    def get_free_space(self, map, grid):
        resolution = map.info.resolution
        width = map.info.width
        height = map.info.height

        # add the index of element numbers that are free
        for i in range(0, width):
            for j in range(0, height):
                if grid[i,j] != 100 and grid[i,j] != -1:
                    self.index_free_space_list.append((i,j))

    def get_ned_position(self, coordinates):
        """
        Grid cell to ned position. index = (x,y)
        """
        resolution = self.map.info.resolution

        # Calculate distance
        x = coordinates[0] * resolution
        y = coordinates[1] * resolution

        return [x, y]

    def calc_dist_to_goal(self, x, y):
        dx = x - self.goal_node.x
        dy = y - self.goal_node.y
        return math.hypot(dx, dy)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacleList):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= size ** 2:
                return False  # collision

        return True  # safe
    
    def check_segment_collision(self, x1, y1, x2, y2):
        for (ox, oy, size) in self.obstacle_list:
            dd = self.distance_squared_point_to_segment(
                np.array([x1, y1]),
                np.array([x2, y2]),
                np.array([ox, oy]))
            if dd <= size**2:
                return False  # collision
        return True # safe

    def check_wall_collision(self, x1, y1, x2, y2):
        """
        Check collision using Bresenham directly.
        """
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end
        """
        res = self.map.info.resolution
        # Setup initial conditions
        dx = (x2 - x1)
        dy = (y2 - y1)
    
        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)
    
        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
    
        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
    
        # Recalculate differentials
        dx = (x2 - x1)
        dy = (y2 - y1)
    
        # Calculate error
        error = int(dx / 2.0)
        ystep = self.map.info.resolution if y1 < y2 else -self.map.info.resolution
        # Iterate over bounding box generating points between start and end
        y = y1
        for x in np.arange(x1, x2, self.map.info.resolution):
            coord = (y, x) if is_steep else (x, y)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
            
            # CHECK COLLISION
            x_grid = coord[0]/res
            y_grid = coord[1]/res
            cell_value = self.grid[ int(round(x_grid)), int(round(y_grid)) ]
            if cell_value == -1 or cell_value == 100:
                return False # collision
    
        return True # safe

    def check_wall_collision2(self, x1, y1, x2, y2):
        nodes = self.bresenham(x1, y1, x2, y2)
        res = self.map.info.resolution
        for node in nodes:
            x = node[0]/res
            y = node[1]/res
            cell_value = self.grid[ int(round(x)), int(round(y)) ]
            if cell_value == -1 or cell_value == 100:
                return False # collision 

        return True # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        alpha = math.atan2(dy, dx)
        return d, alpha

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

    @staticmethod
    def rotation_to_world_frame(a1):
        """
         Given two poses as the focal points of a hyperellipsoid, xfrom, xto ∈ X, the function RotationToWorldFrame (xfrom, xto) 
         returns the rotation matrix, R element in SO(2), from the hyperellipsoid-aligned frame to the NED-frame
        """
        # first column of idenity matrix transposed
        id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        M = a1 @ id1_t
        U, S, Vh = np.linalg.svd(M, True, True)
        R = np.dot(np.dot(U, np.diag(
            [1.0, 1.0, np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])), Vh)
        
        return R

    @staticmethod
    def compute_sampling_space(start_node, goal_node):
        """
        Computes values for the heuristic sampling domain, formed by an ellipse.
        Sample space is defined by cBest
        cMin is the minimum distance between the start point and the goal
        xCenter is the midpoint between the start and the goal
        cBest changes when a new path is found
        """
        cMin = math.sqrt(pow(start_node.x - goal_node.x, 2)
                         + pow(start_node.y - goal_node.y, 2))
        xCenter = np.array([[(start_node.x + goal_node.x) / 2.0],
                            [(start_node.y + goal_node.y) / 2.0], [0]])
        a1 = np.array([[(goal_node.x - start_node.x) / cMin],
                       [(goal_node.y - start_node.y) / cMin], [0]])

        etheta = math.atan2(a1[1], a1[0])

        return cMin, xCenter, a1, etheta

    @staticmethod
    def sample_unit_ball():
        """
        The function, sample_unit_ball returns a uniform sample from the volume of an circle of 
        unit radius centred at the origin.
        """
        a = random.random()
        b = random.random()

        if b < a:
            a, b = b, a

        sample = (b * math.cos(2 * math.pi * a / b),
                  b * math.sin(2 * math.pi * a / b))

        return np.array([[sample[0]], [sample[1]], [0]])

    def get_path_len(self, path):
        pathLen = 0
        for i in range(1, len(path)):
            node1 = path[i]
            node2 = path[i - 1]
            pathLen += self.euclidian_distance(node1, node2)

        return pathLen

    @staticmethod
    def get_path_cost(path):
        path_cost = 0
        for node in path:
            path_cost += node.cost
        
        return path_cost


    @staticmethod
    def euclidian_distance(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        # Return minimum distance between line segment vw and point p
        if (np.array_equal(v, w)):
            return (p-v).dot(p-v) # v == w case
        l2 = (w-v).dot(w-v) # i.e. |w-v|^2 -  avoid a sqrt
        # Consider the line extending the segment, parameterized as v + t (w - v).
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment vw.
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        projection = v + t * (w - v) # Projection falls on the segment
        return (p-projection).dot(p-projection)

    @staticmethod
    def ssa(angle):
        """
        Smallest signed angle. Maps angle into interval [-pi pi]
        """
        wrpd_angle = (angle + math.pi) % (2*math.pi) - math.pi
        return wrpd_angle

    @static_var(counter=0)
    def get_sum_c_c(self, from_node):
        """
        Finds sum of curvature cost, recursively. The static variable keeps track of depth/#parents
        """
        # stash counter in the function itself
        RRT_utils.get_sum_c_c.counter += 1
        if from_node.parent == None:
            return 0
        return from_node.cost + self.get_sum_c_c(from_node.parent)

    @staticmethod
    def get_max_kappa(node):
        """
        Finds maximum curvature from node to root, recursively.
        """
        if node.parent == None:
            return 0
        return max(node.cost, RRT_utils.get_max_kappa(node.parent))

    @staticmethod
    def get_min_obstacle_distance(node, obstacleList):
        """
        Finds minimum distance to obstacle from node.
        """
        dx_list = [ ox - node.x for (ox, oy, size) in obstacleList]
        dy_list = [ oy - node.y for (ox, oy, size) in obstacleList]
        d_list = [math.sqrt(dx * dx + dy * dy) for (dx, dy) in zip(dx_list, dy_list)]

        return min(d_list)

    def generate_final_path(self, target):
        path = [self.goal_node]

        node = target
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.append(node)

        path.reverse()

        return path

    def calc_heuristic(self, node, goal_node):
         d, _ = self.calc_distance_and_angle(node, goal_node)
         return d

    def get_heuristic(self, node, visisted_set):
        if node in visisted_set:
            return float('Inf')
        else:
            return node.heuristic_cost

    def print_path(self, path, name):
        print(name)
        print("  x,   y,  alpha,  kappa,  d,   h,  cost")
        for node in path:
            i = [node.x, node.y, math.degrees(node.alpha), node.kappa, node.d, node.heuristic_cost, node.cost]
            i = map(prettyfloat, i)
            print(list(i))

    def print_children(self, path):
        print("Children:")
        for node in path:
                string = ""
                for child in node.children:
                    i = [child.x, child.y]
                    i = map(prettyfloat, i)
                    string += str(list(i)) + ", "
                print(string)        

    def update_nearest_node(self, new_node):        
        d_new = new_node.heuristic_cost
        d_closest = self.neareast_node.heuristic_cost
        if d_new < d_closest:
            self.neareast_node = new_node

    @staticmethod
    def paths_are_equal(list1, list2):
        return all(elem1 == elem2 for elem1, elem2 in zip(list1, list2))

    def unblock_parents(self, node, visisted_set):

        while node.parent in visisted_set:
            visisted_set.remove(node.parent)
            node = node.parent

    def all_children_blocked(self, node, visited_set):
        """
        check if visited_set contains all elements in node.children
        """
        result =  all(elem in visited_set  for elem in node.children)
        return result

class prettyfloat(float):
    """
    Class for printing float with given precision
    """
    def __repr__(self):
        return "%0.2f" % self
