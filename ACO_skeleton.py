# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:42:08 2018
Updated on Tue Jan 7 20:30:01 2020
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


class AntforTSP(object):
    def __init__(self, n, Nant, Niter, rho, alpha=1, beta=1, seed=None):
        self.n = n #husein graph N**2 col N rows
        self.Graph = np.array([(np.arange(1,n**2 + 1)) for i in range(n)])
        self.Nant = Nant
        self.Niter = Niter
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones(self.Graph.shape) / len(self.Graph)
        self.local_state = np.random.RandomState(seed)
        self.threat_cnt = np.empty(n ** 2 )
        #print(self.threat_cnt)
        """
        This method invokes the ACO search over the N queen graph.
        It returns the best tour located during the search.
        'all_paths' is a list of pairs, each contains a path and its associated length.
         every individual 'path' is a list of positions, each represented as node.
        """

    def run(self):
        # Book-keeping: best tour ever
        shortest_path = None
        best_path = ("TBD", np.inf)
        for i in range(self.Niter):
            all_paths = self.constructColonyPaths()
            self.depositPheronomes(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print(i + 1, ": ", shortest_path[1])
            if shortest_path[1] < best_path[1]:
                best_path = shortest_path
            self.pheromone *= self.rho  # evaporation
        return best_path

        """
        This method deposits pheromones on the edges.
        Importantly, unlike the lecture's version, this ACO selects only 1/4 of the top tours - and updates only their edges, 
        in a slightly different manner than presented in the lecture.
        """

    def depositPheronomes(self, all_paths):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        Nsel = int(self.Nant / 4)  # Proportion of updated paths
        for path, dist in sorted_paths[:Nsel]:
            for i in range (len(path)-1):
                self.pheromone[path[i],path[i+1]] += 1.0 / self.Graph[path[i],path[i+1]]  # dist
        #ERRRROR
        """
        This method generates paths for the entire colony for a concrete iteration.
        The input, 'path', is a list of edges, each represented by a pair of nodes.
        Therefore, each 'arc' is a pair of nodes, and thus Graph[arc] is well-defined as the edges' length.
        """

    def evalTour(self, path):

        return np.sum(self.threat_cnt[path])



    def constructSolution(self, start):
        path = []
        visited = set()
        prev = start
        path.append(prev)
        self.threat_cnt_update(prev)
        visited.add(start)
        for i in range(self.n):
            next_v = self.nextMove(self.pheromone[i][:], self.Graph[i][:], visited)
            visited.add(next_v)
            self.threat_cnt_update(next_v)
            path.append(next_v)

        return path
        """
        This method generates 'Nant' paths, for the entire colony, representing a single iteration.
        """


    def threat_cnt_update(self, num):
        n = self.n
        row = num % n
        col = num // n
        print(row)
        print(col)
        threats = set(np.arange(num, -1, -n))  #rows back
        threats = threats.union(set(np.arange(num, n ** 2, n))) #rows front
        threats = threats.union(set([num + j for j in range(n - row)]))
        threats = threats.union(set([num - j for j in range(row)]))

        threats = threats.union(set([num + (n+1) * i for i in range(min(n-row, n-col))])) #diag for down
        threats = threats.union(set([num - ((n+1) * i) for i in range(min(n-row, col))])) #diag back top
        threats = threats.union(set([num + ((n - 1) * i) for i in range(min(row, n - col))])) #diag for top
        threats = threats.union(set([num - ((n + 1) * i) for i in range(min(row, col))])) # diag back down
        threats.remove(num)
        threats = list(threats)
        print("queen at", num)
        print("threats",threats)
        self.threat_cnt[threats] += 1


    def constructColonyPaths(self):
        # TODO 3
        paths = []  # all Nant paths of graph size
        for i in range(self.Nant):
            sol = self.constructSolution(0)
            paths.append((sol, self.evalTour(sol) ))
        return paths
        """"""

    def nextMove(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)  # Careful
        """This method probabilistically calculates the next move (node) given a neighboring 
        information per a single ant at a specified node.
        Importantly, 'pheromone' is a specific row out of the original matrix, representing the neighbors of the current node.
        Similarly, 'dist' is the row out of the original graph, associated with the neighbors of the current node.
        'visited' is a set of nodes - whose probability weights are constructed as zeros, to eliminate revisits.
        The random generation relies on norm_row, as a vector of probabilities, using the numpy function 'choice'
        : python passes arguments "by-object"; pheromone is mutable"""
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        node = self.local_state.choice(range(self.n ** 2), 1, p=norm_row)[0]
        return node

if __name__ == "__main__" :
    Niter = 1000
    Nant = 200
    n = 8
    ant_colony = AntforTSP(n, Nant, Niter, rho=0.95, alpha=1.5, beta=1.5)
    shortest_path = ant_colony.run()