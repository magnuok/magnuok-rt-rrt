#!/usr/bin/env python3

import rospy
import math
import os
import sys
import numpy as np
import random
import cProfile
import pstats
import io

class Grid():

    class Cell():
        def __init__(self, i, j):
            self.indices = []
            self.i = i
            self.j = j
        
        def __str__(self):
            return " [(" + str(self.i) + ", " + str(self.j) + "), " + str(self.indices) + "] "


        def add_index(self, indice):
            self.indices.append(indice)
        
        def get_indices(self):
            return self.indices

    def __init__(self, x_min, x_max, y_min, y_max, quantity):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n = quantity
        self.x_org = x_min
        self.y_org = y_min
        self.cell_size = (x_max - x_min)/quantity

        self.grid = self.create_grid(self.n)

    def __str__(self):
        return "\n".join([" ".join([str(cell) for cell in row]) for row in self.grid])

    def __get_cell_indices(self, i, j):
        return self.grid[self.n-1 - i][j].indices

    def __get_cell(self, i, j):
        return self.grid[self.n-1 - i][j]

    def __get_closest_neighbors(self, radius, rowNumber, columnNumber):
        return [[self.__get_cell(i,j) if  i >= 0 and i < len(self.grid) and j >= 0 and j < len(self.grid[0]) else 0
                    for j in range(columnNumber-radius, columnNumber+radius+1)]
                        for i in range(rowNumber-radius, rowNumber+radius+1)]

    def __get_nonempty_neighbours(self, rowNumber, columnNumber):
        neighbors = self.__get_closest_neighbors(1, rowNumber, columnNumber)
        non_emptyneigh = []

        # Down
        lower_cells = neighbors[0]
        for cell in lower_cells:
            if cell is not 0:                    
                i = cell.i
                current_cell = cell
                while 0 < i and len(current_cell.indices) is 0:
                    i-= 1 # go down
                    current_cell = self.__get_cell(i,cell.j)
                non_emptyneigh.append(current_cell)

        # Up
        upper_cells = neighbors[2]
        for cell in upper_cells:
            if cell is not 0:
                i = cell.i
                current_cell = cell
                while i < self.n-1 and len(current_cell.indices) is 0:
                    i+= 1 # go left
                    current_cell = self.__get_cell(i, cell.j)
                non_emptyneigh.append(current_cell)

        # left
        left_cells = [row[0] for row in neighbors]
        for cell in left_cells:
            if cell is not 0:
                j = cell.j                    
                current_cell = cell
                while 0 < j and len(current_cell.indices) is 0:
                    j-= 1 # go up
                    current_cell = self.__get_cell(cell.i, j)
                non_emptyneigh.append(current_cell)

        # right
        right_cells = [row[2] for row in neighbors] 
        for cell in right_cells:
            if cell is not 0:
                j = cell.j
                current_cell = cell
                while j < self.n-1 and len(current_cell.indices) is 0:
                    j+= 1 # go right
                    current_cell = self.__get_cell(cell.i, j)
                non_emptyneigh.append(current_cell)

        return non_emptyneigh

    def add_index_to_cell(self, x, y, index):
        """
        Input: x- and y position, and corresponding index. Adds the index to the corresponding cell.
        """
        cell_i = math.floor( ( y - self.y_org ) / self.cell_size) # row direction
        cell_j = math.floor( ( x - self.x_org ) / self.cell_size) # column direction
        # print("row = %f, col= %f" % (cell_i, cell_j))
        self.__get_cell(cell_i, cell_j).add_index(index)

    def create_grid(self, n):
        grid = [ [ self.Cell(i,j) for j in range(n)] for i in range(n-1, -1, -1)]
        return grid
    
    def get_X_si_indices(self, x, y):
        """
        Returns the set of nodes X_si with the closest nodes in the adjecent grid squares.
        """
        cell_i = math.floor( ( y - self.y_org ) / self.cell_size) # row direction
        cell_j = math.floor( ( x - self.x_org ) / self.cell_size) # column direction

        nonempty_neighbours = self.__get_nonempty_neighbours(cell_i, cell_j)

        X_si = []
        # Add nodes inside same grid.
        X_si += self.__get_cell_indices(cell_i, cell_j)
        # Add neighbours
        for neighbour in nonempty_neighbours:
            X_si += neighbour.indices
        
        return X_si
    
    def get_nearest_index_X_si(self, node, node_list):
        """
        Retunrs index of nearest node. Returns False in no near. 
        """
        X_si = self.get_X_si_indices(node.x, node.y)

        if len(X_si) == 0:
            return -1

        dist_list = [(node_list[i].x - node.x) ** 2 +
                     (node_list[i].y - node.y) ** 2 for i in X_si]        
        nearest_index = X_si[dist_list.index(min(dist_list))]

        return nearest_index

def main():
    gridmap = Grid(7, 13, 7, 13, 5)
    
    # i = rad
    # j = kolonne
    gridmap.add_index_to_cell(9.5, 8.8, 1) # x, y, indice
    gridmap.add_index_to_cell(9.5, 12.2, 2)

    print(str(gridmap))
    
    X_si = gridmap.get_X_si_indices(9.5, 8.8) # x, y

    print(list(X_si))

if __name__ == "__main__":
    main()


        #     string =""
        # for cell in right_cells:
        #     string+= str(cell)        
        # print(string)

        # for row in neighbors:
        #     string = ""
        #     for cell in row:
        #         string+= str(cell)
        #     print(string)
                
