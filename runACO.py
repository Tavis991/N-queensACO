# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""
from ACO import AntforTSP as ACO
import numpy as np
import os


if __name__ == "__main__" :
    dirname = "" #"~/sandbox/ici/ex2 
    fname = os.path.join(dirname,"SF96.dat")
    data = []
    with open(fname) as f :
        for line in f:
            data.append(line.split())
    n = len(data)
    G = np.empty([n,n])
    for i in range(n) :
        G[i,i] = np.inf
        for j in range(i+1,n) :
            G[i,j] = np.linalg.norm(np.array([float(data[i][1]),float(data[i][2])]) - np.array([float(data[j][1]),float(data[j][2])]))
            G[j,i] = G[i,j]
#        
    Niter = 1000
    Nant = 200
    ant_colony = ACO(G,Nant, Niter, rho=0.95, alpha=1, beta=3)
    shortest_path = ant_colony.run()
    print("shotest_path: {}".format(shortest_path))