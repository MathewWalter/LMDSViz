# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:21:00 2021

@author: mjwalter
"""
# Example using MDS/LMDS (using Playpus to optimise).


from platypus import NSGAIII, Problem, Real
from LMDSViz import TauCalculate, landmark_MDS, Visualise, XYCreate, embeddingFunction
import platypus as pl
import numpy as np 
import scipy.spatial.distance as spd
from sklearn.manifold import MDS, Isomap
from sklearn.decomposition import PCA
import umap  
import random


# Create a callback function to retrive the decision and objective popualitons from Platypus
TotalPopulation = []

def callback_function(algorithm):
	print('number of function evaluations complete:', algorithm.nfe)
	TotalPopulation.append(algorithm.population)
	return TotalPopulation

# Set up problem and perform optimisation using Platypus
def generate_data(problem, runtime):
	algorithm = NSGAIII(problem, divisions_outer=32)
	algorithm.run(runtime, callback=callback_function)
	return algorithm, TotalPopulation




if __name__ == "__main__":
	
	# For MDS:
	#problem = pl.DTLZ2(3)
	#algorithm, TotalPopulation = generate_data(problem, 10000)
	#population_size = algorithm.population_size
	#Xs, Ys = XYCreate(TotalPopulation, population_size) #Splits the decision and objective space population into Xs and Ys respectivley.
	#colouring = TauCalculate(Ys, population_size)  # Create exploration/expolitation colouring
	
	#distanceMatrix = spd.cdist(Ys, Ys)  # Create distance matrix
	#embedding = PCA(n_components=2)  # Choose dimentionality reduction technique 
	
	#DataReduced_coords = embeddingFunction(embedding, distanceMatrix) # Embedding of coordinates
	
	#Visualise(DataReduced_coords,colouring,population_size, label2 = problem)  # Visualise

	#For LMDS 
	problem = pl.DTLZ2(3)
	algorithm, TotalPopulation = generate_data(problem, 10000)
	#global population_size
	population_size = algorithm.population_size
	Xs, Ys = XYCreate(TotalPopulation, population_size) #Splits the decision and objective space population into Xs and Ys respectivley.
	colouring = TauCalculate(Ys, population_size)  # Create exploration/expolitation colouring
	
	#decide on the landmarks
	lands = random.sample(range(0,Ys.shape[0],1), 100)
	lands = np.array(lands,dtype=int)
	Dl2 = spd.cdist(Ys[lands,:], Ys, 'euclidean')
	
	DataReduced_coords = landmark_MDS(Dl2, lands, dim=2) # Embedding of coordinates
	#Axis labels may require readjusment, can be done in the visualise function
	Visualise(DataReduced_coords,colouring,population_size, label2 = problem)  # Visualise