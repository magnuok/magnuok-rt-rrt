#!/usr/bin/env python3
"""

Path planning Sample Code with RRT*

author: Magnus Knædal

Forked from: Atsushi Sakai(@Atsushi_twi)

"""
import rospy
import math
import os
import sys
import numpy as np
import random
import cProfile
import pstats
import io
import copy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../RRTStar/")

# Import utility-functions
try:
    from _plot_utils import Plot_utils
    from rrt import RRT_utils, prettyfloat
    from grid import Grid
except ImportError:
    raise

class RRTStar(RRT_utils, Plot_utils, Grid):
    """
    Class for informed RRT Star planning
    """

    class Node:
        def __init__(self, x, y, alpha):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.alpha = alpha
            self.parent = None
            self.children = []

            self.cost = 0.0
            self.heuristic_cost = 0.0 # cost to reach
            self.kappa = 0.0
            self.d = float('Inf')

    def __init__(self, start, goal, obstacle_list, goal_region,
                 expand_dis, path_resolution, goal_sample_rate, beta,
                 max_alpha, max_kappa, k_max, K, r_s, grid_values):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        ### Parameters ###
        self.start = self.Node(start[0], start[1], start[2])
        self.root = self.start
        self.goal_node = self.Node(goal[0], goal[1], goal[2])
        
        self.start.heuristic_cost = self.calc_heuristic(self.start, self.goal_node)

        self.goal_region = goal_region
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.beta = beta
        self.obstacle_list = obstacle_list
        self.max_alpha = max_alpha
        self.max_kappa = max_kappa
        self.k_max = k_max
        self.node_list = [self.start] # All nodes =  The tree
        self.visited_set = [] # Keep track of "visited" nodesin path-planner to not get traped in local minima
        self.Q_r = []
        self.Q_s = []
        self.found_goal = False
        self.target = None
        self.K = K # Maximum steps for planning a path if goal not found
        self.r_s = r_s # Controls the closeness of the nodes

        self.near_goal_list = []
        self.neareast_node = self.start # Nodes nearest goal. Used for sampling.
        self.prev_root = None

        self.gridmap = Grid(grid_values[0], grid_values[1], grid_values[2], grid_values[3], grid_values[4])
        self.gridmap.add_index_to_cell(self.start.x, self.start.y, self.node_list.index(self.start))

        ### Ros node, pubs and subs ### 
        rospy.init_node('RRT', anonymous=True)
        self.pub = rospy.Publisher('nodes', Marker, queue_size = 10)
        self.pub_blocked_nodes = rospy.Publisher('blocked_nodes', Marker, queue_size = 10)
        self.pub_start_goal = rospy.Publisher('start_goal', Marker, queue_size = 10)
        self.pub_ellipse = rospy.Publisher('ellipse', Marker, queue_size = 1)
        self.pub_obst = rospy.Publisher('obstacles', MarkerArray, queue_size = 1)
        self.pub_tree = rospy.Publisher('tree', MarkerArray, queue_size = 10)
        self.pub_root_tree = rospy.Publisher('root_tree', MarkerArray, queue_size = 10)
        self.pub_final_path = rospy.Publisher('final_path', Marker, queue_size = 10)
        self.pub_test = rospy.Publisher('test', Marker, queue_size = 10)
        rospy.Subscriber("map", OccupancyGrid, self.callback_map)

        ## Initialize ##
        # Init map
        self.map = None
        while self.map is None:
            continue

        self.grid = np.reshape(np.array(self.map.data), (self.map.info.width, self.map.info.height) )
        self.index_free_space_list = []
        self.get_free_space(self.map, self.grid)
        self.search_space_area = self.map.info.resolution**2 * len(self.index_free_space_list)
        self.plot_start_and_goal(self.start, self.goal_node)
    
    def realtime_planner(self, animation=True):
        """
        Main function. Initialize, main loop.
        """
        current_path = []
        cBest = float('Inf')

        for iter in range(0, 20):
            print("Iteration %i", iter)

            for i in range(80): 
                self.expand_and_rewire(self.root, self.goal_node, cBest, i, animation = True)
            
            if iter == 2:
                self.obstacle_list.append((8.5, 10.5, 0.7))
                self.obstacle_list.append((11.5, 10.5, 0.7))
            self.block_obstacle_branches(self.obstacle_list)
            if iter == 5:
                self.obstacle_list.pop()

            rospy.sleep(1)

            # Add previous root to visisted, so that it never gets picked.
            if (self.prev_root != None) and (self.prev_root != self.root) and (self.prev_root not in self.visited_set):
                self.visited_set.append(self.prev_root)

            current_path, cBest, new_root = self.plan_path(self.root, self.goal_node, current_path)

            # Plotting
            self.draw_graph(self.root, self.goal_node, cBest)
            self.plot_final_path(current_path)
            
            if new_root:
                self.visited_set = []
                self.Q_s = []
                self.change_root(current_path[0])

            if current_path[0] == self.target or self.is_near_goal(current_path[0], self.goal_node):
                print("Found goal")
                break
        
        # Plotting
        self.draw_graph(self.root, self.goal_node, cBest)
        self.plot_final_path(current_path)
        
        return current_path

    def change_root(self, new_root):

        new_root.parent = None

        new_child = self.root
        new_child.parent = new_root
        if new_root in new_child.children: # TODO
            new_child.children.remove(new_root)
        new_child.cost = float('Inf')
        self.update_node_values(new_child)

        new_root.children.append(new_child)
        new_root.cost = 0
        new_root.d = float('Inf')

        self.prev_root = new_child
        self.root = new_root
        
    def expand_and_rewire(self, root, goal, cBest, anim_count, animation):
        """
        tree = self.node_list
        Q_r, Q_s: queues for rewiring
        k_max: Maximum number of neighbours  around a node. 
        r_s: Maximum allowed distance between the nodes in the tree. = expand_distance handles in find_nearby_nodes
            k_max and r_s controlls the dencity of the tree.
        cBest: best path length obtained to goal. Update in plan_path.
        """

        rnd_node = self.informed_sample(root, goal, cBest)
        index = self.gridmap.get_nearest_index_X_si(rnd_node, self.node_list)
        while index == -1: # no one close
            rnd_node = self.informed_sample(root, goal, cBest)
            index = self.gridmap.get_nearest_index_X_si(rnd_node, self.node_list)

        nearest_node = self.node_list[ self.gridmap.get_nearest_index_X_si(rnd_node, self.node_list) ]
        new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

        if self.check_segment_collision(new_node.x, new_node.y, nearest_node.x, nearest_node.y):
            near_inds = self.find_nodes_near(new_node)
            d = self.euclidian_distance(nearest_node, new_node)

            if len(near_inds) < self.k_max and len(near_inds) > 0 and d > self.r_s:
                # Returns none if no feasible. 
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.gridmap.add_index_to_cell(new_node.x, new_node.y, len(self.node_list)-1)
                    self.Q_r.append(new_node)
                    self.update_nearest_node(new_node)

                    if self.is_near_goal(new_node, goal) and self.check_segment_collision(new_node.x, new_node.y, self.goal_node.x , self.goal_node.y):
                        self.found_goal = True
                        self.near_goal_list.append(new_node)
                        self.target = self.get_best_target_node(goal)

            else:
                self.Q_r.append(nearest_node)
 
            self.rewire_random_nodes()

        self.rewire_from_root(root)
       
    def plan_path(self, root, goal, current_path):
        """
        Plan a path for k steps.
        """
        found_path = False
        if self.found_goal == True:
            goal_path = self.generate_final_path(self.target)
            if root in goal_path and goal_path[1] != self.prev_root:
                found_path = True

        if found_path == True:
            # If next cost is blocked
            if goal_path[1].cost == float('Inf'):
                self.found_goal = False
                cBest = float('Inf')
                goal_path = [root]
                return goal_path, cBest, False # not new root
            else:                
                cBest = self.get_path_len(goal_path)
                goal_path.pop(0)
                return goal_path, cBest, True

        else:
            current_node = root
            updated_path = []
            cBest = float('Inf')
            k_step_counter = 0
            
            # If only child is previous parent
            if len(current_node.children) == 1 and current_node.children[0] == self.prev_root:
                #stay in root.
                updated_path.append(root)
                return updated_path, cBest, False # not new root

            else:
                while len(current_node.children) > 0:
                    # Init with temp node different from prev root
                    temp_node = current_node.children[0]
                    if temp_node == self.prev_root:
                        temp_node = current_node.children[1]
                    min_cost = temp_node.cost + self.get_heuristic(temp_node, self.visited_set)
                    for child in current_node.children:
                        # never consider prev root.
                        if child != self.prev_root:
                            cost = child.cost + self.get_heuristic(child, self.visited_set)
                            if cost < min_cost:
                                min_cost = cost
                                temp_node = child
                    # Adds best child to updated path
                    updated_path.append(temp_node)
                    if len(temp_node.children) == 0 or self.all_children_blocked(temp_node, self.visited_set) or k_step_counter == self.K:
                        self.visited_set.append(temp_node)
                        break
                    current_node = temp_node
                    k_step_counter+=1

            if len(updated_path) == 0:
                updated_path.append(root)
            if len(current_path) > 0: # pop for previous round
                current_path.pop(0)
            if len(current_path) == 0:
                current_path.append(root)
            
            # If both only contains root
            if updated_path[0] == root and current_path[0] == root:
                return updated_path, cBest, False # Stay in root

            # If current_path is better than updated path
            if current_path[-1].heuristic_cost < updated_path[-1].heuristic_cost:
                # check cost is not inf
                if  current_path[0].cost == float('Inf'):
                    updated_path = [root]
                    return updated_path, cBest, False # not new root

                # If current path only containts root
                if current_path[-1] == root:
                    return current_path, cBest, False # not update root, stay on same path.
                else:
                    return current_path, cBest, True # update root, stay on same path.
            else:
                if updated_path[0].cost == float('Inf'):
                    updated_path = [root]
                    return updated_path, cBest, False # not new root                    
                else:
                    return updated_path, cBest, True

    def informed_sample(self, root, goal, cMax):
        """
        Performe a informed sample. Returns independent and identically distributed (i.i.d.) samples from the state space.
        """

        Pr = random.uniform(0, 1)

            # Sample line between nearest node to goal and goal
        if Pr > (1-self.goal_sample_rate):
            a = (goal.y - root.y) / (goal.x - root.x)
            b = goal.y - a * goal.x

            # Sample from a free grid cell.
            cell = -1
            while True:
                x_sample = random.uniform(root.x, goal.x)
                y_sample = a * x_sample + b
                cells = self.get_grid_cells( [(x_sample, y_sample)] )
                cell = cells[0]
                if cell != -1 and cell != 100:
                    rnd = self.Node(x_sample, y_sample, 0)
                    break
            
            return rnd
            #return self.Node(goal.x, goal.y, goal.alpha)

            # Sample unfiform
        if ( Pr <= (1-self.goal_sample_rate)/self.beta ) or (cMax == float('Inf')):
            pos = self.get_ned_position(random.choice(self.index_free_space_list))
            return self.Node(pos[0], pos[1], 0)
        
            # Sample ellipse
        else:                        
            cMin, xCenter, a1, _ = self.compute_sampling_space(root, goal)
            C = self.rotation_to_world_frame(a1)

            r = [cMax / 2.0,
                 math.sqrt(cMax ** 2 - cMin ** 2) / 2.0,
                 math.sqrt(cMax ** 2 - cMin ** 2) / 2.0]
            L = np.diag(r)

            # Sample from a free grid cell.
            cell = -1
            while True:
                xBall = self.sample_unit_ball()

                rnd = np.dot(np.dot(C, L), xBall) + xCenter
                cells = self.get_grid_cells( [(rnd[0][0], rnd[1][0])] )
                cell = cells[0]
                if cell != -1 and cell != 100:
                    rnd = self.Node(rnd[0][0], rnd[1][0], 0)
                    break
            return rnd

    def is_near_goal(self, node, goal):
        """
         Given a pose, the function is_near_goal returns True if and only if the state is in the goal region, as defined.
        """
        d = self.euclidian_distance(node, goal)
        if d < self.goal_region:
            return True
        return False

    def choose_parent(self, new_node, filtered_inds):
        """
        Set parent of new node to the one found with lowest cost and satisfying constraints.
        """
        costs = []
        for i in filtered_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node, self.expand_dis)

            no_col = self.check_segment_collision(new_node.x, new_node.y, t_node.x, t_node.y)
            #no_collision = self.check_collision(t_node, self.obstacle_list)
            no_wall_collision = self.check_wall_collision(near_node.x, near_node.y, t_node.x, t_node.y)

            if t_node and no_col and no_wall_collision:
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float('Inf'))  # the cost of collision node
        
        min_cost = min(costs)
        if min_cost == float('Inf'):
            return None

        # Set parent to the one found with lowest cost
        min_ind = filtered_inds[costs.index(min_cost)]
        parent_node = self.node_list[min_ind]
        new_node = self.steer(parent_node, new_node, self.expand_dis)
        new_node.parent = parent_node
        new_node.cost = min_cost
        self.update_node_values(new_node)

        self.unblock_parents(new_node, self.visited_set)
        
        # Set child of parent_node to new node
        parent_node.children.append(new_node)

        return new_node

    def update_node_values(self, node):
        """
        Updates values for new or rewired node.
        """
        d, alpha = self.calc_distance_and_angle(node.parent, node)
        node.d = d
        node.alpha = alpha
        node.heuristic_cost = self.calc_heuristic(node, self.goal_node) # calculate heuristic cost

        if d == 0 or node.parent.d == 0 or abs(self.ssa(node.parent.alpha - node.alpha)) > self.max_alpha:
            node.kappa = float('Inf')
        else:
            node.kappa = (2*math.tan(abs(self.ssa(node.parent.alpha - node.alpha)))) / min(node.parent.d, node.d)
        
    def get_best_target_node(self, goal):
        cost_list = []
        for node in self.near_goal_list:
            no_col = self.check_segment_collision(node.x, node.y, goal.x, goal.y)
            #no_collision = self.check_collision(node, self.obstacle_list)
            no_wall_collision = self.check_wall_collision(node.x, node.y, goal.x, goal.y)
            constraints_ok = self.check_constraints(node, goal)

            if no_col and no_wall_collision and constraints_ok:
                cost = self.calc_new_cost(node, goal)
            else:
                cost = float('Inf')

            cost_list.append(cost)
        
        best_target = self.near_goal_list[ cost_list.index( min(cost_list) ) ]

        return best_target

    def find_nodes_near_obstacle(self, obstacle):
        """
        obstacle = [x, ,y, radius]
        returns nearest nodes in X_si.
        """
        obst_x = obstacle[0]
        obst_y = obstacle[1]
        obst_r = obstacle[2]

        # First find nodes nearby
        X_si = self.gridmap.get_X_si_indices(obst_x, obst_y)
        dist_list = [(self.node_list[i].x - obst_x) ** 2 +
                     (self.node_list[i].y - obst_y) ** 2 for i in X_si]
        near_inds = [X_si[i] for dist, i in zip(dist_list, range(0, len(dist_list))) if dist <= obst_r ** 2]

        return near_inds
    
    def block_obstacle_branches(self, obstacles):
        
        for obstacle in obstacles:

            near_inds = self.find_nodes_near_obstacle(obstacle)

            for index in near_inds:
                node = self.node_list[index]

                node.cost = float('Inf')
                self.propagate_cost_to_leaves(node)

    def find_nodes_near(self, new_node):
        """
        returns nearest nodes in X_si. Filters nodes by angle and curvature constraints.
        """
        nnode = len(self.node_list) + 1
        epsilon = math.sqrt( (self.search_space_area * self.k_max) / (math.pi * nnode))

        if epsilon < self.r_s:
            r = self.r_s
        else:
            r = epsilon
        
        # First find nodes nearby
        X_si = self.gridmap.get_X_si_indices(new_node.x, new_node.y)
        dist_list = [(self.node_list[i].x - new_node.x) ** 2 +
                     (self.node_list[i].y - new_node.y) ** 2 for i in X_si]
        near_inds = [X_si[i] for dist, i in zip(dist_list, range(0, len(dist_list))) if dist <= r ** 2]

        # Filter by constraints
        filtered_inds = []
        for i in near_inds:
            near_node = self.node_list[i]
            if self.check_constraints(near_node, new_node):
                filtered_inds.append(i)

        return filtered_inds

    def find_nodes_near_root(self, new_node):
        """
        returns nearest nodes in X_si. Filters nodes by angle and curvature constraints.
        """
        nnode = len(self.node_list) + 1
        epsilon = math.sqrt( (self.search_space_area * self.k_max) / (math.pi * nnode))

        if epsilon < self.r_s:
            r = self.r_s
        else:
            r = epsilon
        
        r = 1 # TODO


        X_si = self.gridmap.get_X_si_indices(new_node.x, new_node.y)
        dist_list = [(self.node_list[i].x - new_node.x) ** 2 +
                     (self.node_list[i].y - new_node.y) ** 2 for i in X_si]
        near_inds = [X_si[i] for dist, i in zip(dist_list, range(0, len(dist_list))) if dist <= r ** 2]

        #l = [self.node_list[i] for i in near_inds]
        #self.print_path(l, "near nodes")

        return near_inds

    def check_constraints(self, from_node, to_node):
        """
        Checks constraints related to angle and cruvature.
        """
        to_node_d, to_node_alpha = self.calc_distance_and_angle(from_node, to_node)

        if to_node_d == 0 or from_node.d == 0:
            kappa_next = float('Inf')
        else:
            kappa_next = (2*math.tan(abs(self.ssa(from_node.alpha - to_node_alpha)))) / min(from_node.d, to_node_d)

        alpha_ok = abs( self.ssa(from_node.alpha - to_node_alpha)) < self.max_alpha        
        kappa_ok = kappa_next < self.max_kappa

        if kappa_ok and alpha_ok:
            return True # Constraints ok
        else:
            return False # Not ok

    def rewire_random_nodes(self):
        """
        This function checks if the cost to the nodes in near_inds is less through new_node as compared to their older costs, 
        then its parent is changed to new_node.
        """
        startTime = rospy.Time.now()
        duration = rospy.Duration(0.05)

        # Repeat until time is up or Qr is empty
        while (len(self.Q_r) > 0) and (rospy.Time.now() < startTime+duration):
            node = self.Q_r.pop()
            near_inds = self.find_nodes_near(node)
            
            for i in near_inds:
                near_node = self.node_list[i]
                edge_node = self.steer(node, near_node, self.expand_dis)
                edge_node.cost = self.calc_new_cost(node, near_node)

                no_col = self.check_segment_collision(node.x, node.y, edge_node.x, edge_node.y)
                #no_collision = self.check_collision(edge_node, self.obstacle_list)
                improved_cost = near_node.cost > edge_node.cost
                no_wall_collision = self.check_wall_collision(node.x, node.y, edge_node.x, edge_node.y)
                constraints_ok = self.check_constraints(node, near_node)

                if no_col and improved_cost and no_wall_collision and constraints_ok:                
                    near_node.parent.children.remove(near_node)
                    edge_node.parent = node
                    edge_node.children = near_node.children
                    self.update_node_values(edge_node)
                    self.node_list[i] = edge_node
                    node.children.append(edge_node)
                    
                    self.unblock_parents(edge_node, self.visited_set)

                    self.propagate_cost_to_leaves(node)

                    # Append edge_node to Q_r
                    self.Q_r.append(edge_node)

    def rewire_from_root(self, root):
        """
        This function checks if the cost to the nodes in near_inds is less through root as compared to their older costs, 
        then its parent is changed to new_node.
        """
        if len(self.Q_s) == 0:
            self.Q_s.append(root)
        
        # Keep track of nodes allready added to Q_s
        Q_s_old = []

        startTime = rospy.Time.now()
        duration = rospy.Duration(0.05)

        while len(self.Q_s) > 0 and (rospy.Time.now() < startTime+duration):
            node = self.Q_s.pop()
            Q_s_old.append(node)
            near_inds = self.find_nodes_near_root(node)

            for i in near_inds:
                near_node = self.node_list[i]
                edge_node = self.steer(node, near_node)
                edge_node.cost = self.calc_new_cost(node, near_node)

                no_col = self.check_segment_collision(node.x, node.y, edge_node.x, edge_node.y)
                #no_collision = self.check_collision(edge_node, self.obstacle_list)
                improved_cost = near_node.cost > edge_node.cost
                no_wall_collision = self.check_wall_collision(node.x, node.y, edge_node.x, edge_node.y)
                constraints_ok = self.check_constraints(node, near_node)
                #print("no_col: %r, imp_cost: %r, no wall: %r: constrok: %r" % (no_col, improved_cost, no_wall_collision, constraints_ok))

                if no_col and improved_cost and no_wall_collision and constraints_ok:
                    near_node.parent.children.remove(near_node)
                    edge_node.parent = node
                    edge_node.children = near_node.children
                    self.update_node_values(edge_node)
                    self.node_list[i] = edge_node
                    node.children.append(edge_node)
                    
                    self.unblock_parents(edge_node, self.visited_set)

                    self.propagate_cost_to_leaves(node)

                    # Append edge_node to Q_s if not been added before
                    if near_node not in Q_s_old:
                        self.Q_s.append(edge_node)

    def propagate_cost_to_leaves_old(self, parent_node):
        """
        When rewired, this function updates the costs of parents.
        """
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def propagate_cost_to_leaves(self, parent_node):
        for child in parent_node.children:
            if child.parent == parent_node:
                child.cost = self.calc_new_cost(parent_node, child)
                self.propagate_cost_to_leaves(child)

    def calc_new_cost(self, from_node, to_node):
        """
        Calculate cost functions. TODO: Now only one used at a time. Tuning and superpos. must be done.
        c_d - distance cost
        c_c - curvature cost
        c_o - obstacle cost
        """
        # Distance cost
        d, alpha_next = self.calc_distance_and_angle(from_node, to_node)
        c_d = from_node.cost + d

        # Obstacle cost
        c_o = from_node.cost + 1/self.get_min_obstacle_distance(to_node, self.obstacle_list)

        # Curvature cost
        if d == 0 or from_node.d == 0 or abs(self.ssa(from_node.alpha - alpha_next)) > self.max_alpha:
            c_c = float('Inf')
        else:
            RRTStar.get_sum_c_c.counter = 0
            kappa_next = (2*math.tan(abs(self.ssa(from_node.alpha - alpha_next)))) / min(from_node.d, d)

            c_c =( max(self.get_max_kappa(from_node), kappa_next) 
                + (self.get_sum_c_c(from_node) + kappa_next) / (RRTStar.get_sum_c_c.counter) )
        
        return c_d # c_d, c_o or c_c

    def steer(self, from_node, to_node, extend_length = float('Inf')):

        new_node = self.Node(from_node.x, from_node.y, 0)

        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node

        return new_node

    """ --- ROS callbacks --- """

    def callback_map(self, occupancy_grid_msg):
        self.map = occupancy_grid_msg

def main():
    show_live_animation = False

    # [x, y, radius]
    obstacleList = [
        (10, 10, 1)
        ]

    # Set Initial parameters
    rrt_star = RRTStar(start = [9.5, 8.7, -math.pi/2], # [x, y, theta]
                       goal = [10, 12.2, math.pi/2], # [x, y, theta]
                       obstacle_list = obstacleList,
                       goal_region = 0.3,
                       expand_dis = 1.5,
                       path_resolution = 0.1,
                       goal_sample_rate = 0.1, # percentage
                       beta = 2,
                       max_alpha = math.pi/2,
                       max_kappa = 20,
                       k_max = 3,
                       K = 5, # Maximum steps for planning a path if goal not found
                       r_s = 0.2, # stor innvirkning på hvor mange som blir samplet
                       grid_values = [7, 13, 7, 13, 5]) # [xmin, xmax, ymin, ymax, number of cells]
    path = rrt_star.realtime_planner(animation=show_live_animation)

    rrt_star.print_path(path, "Path:")
    rrt_star.print_children(path)

if __name__ == '__main__':
    # Profiling
    #cProfile.run('main()', 'rrt_profiling.txt')
    #p = pstats.Stats('rrt_profiling.txt')
    #p.sort_stats('cumulative').print_stats(20)
    main()

    rospy.spin()


