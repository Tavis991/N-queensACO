# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:42:08 2018

@author: ofersh@telhai.ac.il
Based on code by <github/Akavall>
"""
import numpy as np

"""
A class for defining an Ant Colony Optimizer for TSP-solving.
The c'tor receives the following arguments:
    Graph: TSP graph 
    Nant: Colony size
    Niter: maximal number of iterations to be run
    rho: evaporation constant
    alpha: pheromones' exponential weight in the nextMove calculation
    beta: heuristic information's (\eta) exponential weight in the nextMove calculation
    seed: random number generator's seed
"""
class AntforTSP(object) :
    def __init__(self, Graph, Nant, Niter, rho, alpha=1, beta=1, seed=None):
        self.Graph  = Graph
        self.Nant = Nant
        self.Niter = Niter
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones(self.Graph.shape) / len(Graph)
        self.local_state = np.random.RandomState(seed)
        """
        This method invokes the ACO search over the TSP graph.
        It returns the best tour located during the search.
        Importantly, 'all_paths' is a list of pairs, each contains a path and its associated length.
        Notably, every individual 'path' is a list of edges, each represented by a pair of nodes.
        """
    def run(self) :
        #Book-keeping: best tour ever
        shortest_path = None
        best_path = ("TBD", np.inf)
        for i in range(self.Niter):
            all_paths = self.constructColonyPaths()
            self.depositPheronomes(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print(i+1, ": ", shortest_path[1])
            if shortest_path[1] < best_path[1]:
                best_path = shortest_path            
            self.pheromone * self.rho  #evaporation
        return best_path

        """
        This method deposits pheromones on the edges.
        Importantly, unlike the lecture's version, this ACO selects only 1/4 of the top tours - and updates only their edges, 
        in a slightly different manner than presented in the lecture.
        """
    def depositPheronomes(self, all_paths) :
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        Nsel = int(self.Nant/4) # Proportion of updated paths
        for path, dist in sorted_paths[:Nsel]:
            for move in path:
                self.pheromone[move] += 1.0 / self.Graph[move] #dist

        """
        This method generates paths for the entire colony for a concrete iteration.
        The input, 'path', is a list of edges, each represented by a pair of nodes.
        Therefore, each 'arc' is a pair of nodes, and thus Graph[arc] is well-defined as the edges' length.
        """
    def evalTour(self, path) :
        res = 0
        for arc in path:
            res += self.Graph[arc]
        return res
#
        """
        This method generates a single Hamiltonian tour per an ant, starting from node 'start'
        The output, 'path', is a list of edges, each represented by a pair of nodes.
        """
    def constructSolution(self, start) :
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.Graph) - 1) :
            move = self.nextMove(self.pheromone[prev], self.Graph[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start)) # completing the tour   
        return path
        """
        This method generates 'Nant' paths, for the entire colony, representing a single iteration.
        """
    def constructColonyPaths(self) :
        all_paths = []
        for i in range(self.Nant):
            path = self.constructSolution(0)
            #constructing pairs: first is the tour, second is its length
            all_paths.append((path, self.evalTour(path))) 
        return all_paths
    
        """
        This method probabilistically calculates the next move (node) given a neighboring 
        information per a single ant at a specified node.
        Importantly, 'pheromone' is a specific row out of the original matrix, representing the neighbors of the current node.
        Similarly, 'dist' is the row out of the original graph, associated with the neighbors of the current node.
        'visited' is a set of nodes - whose probability weights are constructed as zeros, to eliminate revisits.
        The random generation relies on norm_row, as a vector of probabilities, using the numpy function 'choice'
        """
    def nextMove(self, pheromone, dist, visited) :
        pheromone = np.copy(pheromone) #Careful: python passes arguments "by-object"; pheromone is mutable
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = self.local_state.choice(range(len(self.Graph)), 1, p=norm_row)[0]
        return move