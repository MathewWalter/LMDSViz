# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:44:05 2020

@author: mjwalter
"""

from platypus import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.stats as st
import platypus as pl
import random
import scipy.spatial.distance as spd
from sklearn.manifold import MDS, Isomap
import re   
from sklearn.decomposition import PCA
import umap  
from mpl_toolkits.mplot3d import Axes3D
import platypus.problems as pprob
from sklearn import manifold
from scipy.spatial import distance as dist
import time
import statistics
import scipy as sp
from sklearn.metrics import mean_squared_error

# =============================================================================
# 
#  Functions
# 
# =============================================================================


def Visualise(DataReduced_coords,colouring, label2 = 'problem' ):
		'''
		plots the 2D reduced data over time (generations). MDS rotation is fixed.
		X_transformed (output) is a 3D array (generation, solution in population, (x,y) coords).
		'''
		
		#Plots the data
		fig = plt.figure("MDS Space - Objective Space, (%s)" % (label2), figsize=(8,6))
		ax = fig.add_subplot(111, projection='3d')
		x = DataReduced_coords[:,0]
		y = DataReduced_coords[:,1]
		z = np.array([np.repeat(i, population_size) for i in range(int(len(x)/population_size))]).flatten()
		
		#Append first value of tau array to the front of array 
		tau_xxDummy = colouring[0]
		colouring.insert(0, tau_xxDummy)
		tau_xxx = np.array([np.repeat(colouring, population_size)]).flatten()
		
		sc = ax.scatter(x, y, z, c=tau_xxx, cmap = 'viridis' ,edgecolors='black', linewidth=0.2)
		
		#Colours approximation set white
		#Plots the ND solutions, finds the index
		#changed population_size to len(Indx) as final population may not be all non-dominated for small runs
		#NDSet = np.array([s.objectives[0] for s in pl.nondominated(self.result)])
		#xPopulation = Ys[:,0]
		#NDSet, xPopulation = NDSet.tolist(), xPopulation.tolist()
		#Indx = [xPopulation.index(i) for i in NDSet] 
		#ax.scatter(x[Indx], y[Indx], np.array(np.repeat((len(x)/population_size), len(Indx))), marker= 'P', c= 'white') 

		plt.title('Reduced Objective Space')
		ax.set_xlabel('$y_1$')
		ax.set_ylabel('$y_2$')
		ax.set_zlabel('Generation')
		# Set the background color
		ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
		cbar = plt.colorbar(sc, shrink=0.5, aspect=10)
		cbar.ax.text(-1.2, 1.07, "Exploration", fontsize=9.8, rotation=0, va='center') 
		cbar.set_label('Exploitation', labelpad=-25, y=-0.05, rotation=0)
		ax.view_init(elev=45, azim=-45)
		cbar.set_ticks([])
		
		return x,y,z


def embeddingFunction(embedding, distanceMatrix):
		DataReduced_coords = embedding.fit_transform(distanceMatrix)
		return DataReduced_coords


def MDS2(D,dim=[]):
	'''
	Classical MDS
	'''
	# Number of points
	n = len(D)
	
	# Centering matrix
	H = - np.ones((n, n))/n
	np.fill_diagonal(H,1-1/n)
	# YY^T
	H = -H.dot(D**2).dot(H)/2
	
	# Diagonalize
	evals, evecs = np.linalg.eigh(H)

	# Sort by eigenvalue in descending order
	idx   = np.argsort(evals)[::-1]
	evals = evals[idx]
	evecs = evecs[:,idx]

	# Compute the coordinates using positive-eigenvalued components only
	w, = np.where(evals > 0)
	if dim!=[]:
		arr = evals
		w = arr.argsort()[-dim:][::-1]
		if np.any(evals[w]<0):
			print('Error: Not enough positive eigenvalues for the selected dim.')
			return []
	L = np.diag(np.sqrt(evals[w]))
	V = evecs[:,w]
	Y = V.dot(L)
	return Y

def landmark_MDS(D, lands, dim):
	Dl = D[:,lands]
	n = len(Dl)

	# Centering matrix
	H = - np.ones((n, n))/n
	np.fill_diagonal(H,1-1/n)
	# YY^T
	H = -H.dot(Dl**2).dot(H)/2

	# Diagonalize
	evals, evecs = np.linalg.eigh(H)

	# Sort by eigenvalue in descending order
	idx   = np.argsort(evals)[::-1]
	evals = evals[idx]
	evecs = evecs[:,idx]

	# Compute the coordinates using positive-eigenvalued components only
	w, = np.where(evals > 0)
	if dim:
		arr = evals
		w = arr.argsort()[-dim:][::-1]
		if np.any(evals[w]<0):
			print('Error: Not enough positive eigenvalues for the selected dim.')
			return []
	if w.size==0:
		print('Error: matrix is negative definite.')
		return []

	V = evecs[:,w]
	L = V.dot(np.diag(np.sqrt(evals[w]))).T
	N = D.shape[1]
	Lh = V.dot(np.diag(1./np.sqrt(evals[w]))).T
	Dm = D - np.tile(np.mean(Dl,axis=1),(N, 1)).T
	dim = w.size
	X = -Lh.dot(Dm)/2.
	X -= np.tile(np.mean(X,axis=1),(N, 1)).T

	_, evecs = sp.linalg.eigh(X.dot(X.T))

	return (evecs[:,::-1].T.dot(X)).T


def TauCalculate(Ys, population_size):
	'''Calculates Tau - the exploration/exploitation metric
	   used to colour the plots
	   '''
	distanceArrays = []
	N = population_size
	#creates min distance for each solution with prev gens
	for g in range(1, (int(len(Ys)/population_size))):
		Xcurrent = Ys[g*N:g*N+N]
		Xprev = Ys[:g*N]
		D = dist.cdist(Xcurrent, Xprev)
		#Min distance for each solution
		distanceArrays.append(D.min(axis=1))
		assert distanceArrays[-1].shape[0] == N
		
	scn_med = statistics.median(np.concatenate(distanceArrays, axis=0))
	
	tau_x = []	  
	tau = 0
		
	for k in range(len(distanceArrays)): 
		for i in range(N):   
			if distanceArrays[k][i] > scn_med:
				tau = tau + 1
			else:
				tau = tau
		tau_x.append(tau)
		tau = 0
	
#	pickle.dump(tau_x, open('Data.pkl', 'wb')) 
	return tau_x


TotalPopulation = []

def callback_function(self):
	'''returns the population after an EA run with Platypus library
	'''
	
	#print(self.nfe)
	TotalPopulation.append(self.population)
	return TotalPopulation


def XYCreate(TotalPopulation, population_size):
	'''Records Xs as total population in varaible space 
	and Ys as total popualtion in objective space when using Playpus library'''
	
	Xs = np.array([TotalPopulation[i][j].variables for i in range(len(TotalPopulation)) for j in range(population_size)])
	Ys = np.array([TotalPopulation[i][j].objectives for i in range(len(TotalPopulation)) for j in range(population_size)])
	return Xs, Ys


# =============================================================================
# Example with Platypus library 
# =============================================================================


def generate_data(problem, runtime):
	
	sbxProbability =  0.8
	pmProbability =  0.1
	xo = pl.operators.SBX(probability=sbxProbability, distribution_index=15)
	mut = pl.operators.PM(probability=pmProbability, distribution_index=7)
	variator = pl.operators.GAOperator(xo, mut)
	algorithm = NSGAII(problem,  population_size=100)
	TotalPopulation = algorithm.run(runtime, callback=callback_function)
	return algorithm


if __name__ == "__main__":
	#Choose problem from Playpus library e.g., DTLZ2 in 3 objectives and evaluate for 10000 function evaluations
	for problem, runtime in zip([DTLZ2(3)], [10000]):

			start_time = time.time()

			#Evaluate problem with NSGAII
			algorithm = generate_data(problem, runtime)

			population_size = algorithm.population_size
			#Record the population
			Xs, Ys = XYCreate(TotalPopulation, population_size)
			colouring = TauCalculate(Ys, population_size)
			distanceMatrix = spd.cdist(Ys, Ys)

			#Choose dimentionality reduction technique 
			#embedding = PCA(n_components=2)
			#embedding = MDS(n_components=2, n_jobs=-1, dissimilarity="precomputed", random_state=1) #stress function type
			#embedding = Isomap(n_components=2)
			#embedding = PCA(n_components=2)
			#embedding = umap.UMAP(n_components=2)

			#Embedding of coordinates
			#DataReduced_coords = embeddingFunction(embedding, distanceMatrix)
			# MDS (SVD)
			#DataReduced_coords = MDS2(distanceMatrix, dim=2)

			#or use LMDS (e.g., with 100 landmarks)
			lands = random.sample(range(0,Ys.shape[0],1), 100)
			lands = np.array(lands,dtype=int)
			Dl2 = spd.cdist(Ys[lands,:], Ys, 'euclidean')
			DataReduced_coords = landmark_MDS(Dl2, lands, dim=2)

			Visualise(DataReduced_coords,colouring,label2 = problem)

			print("--- %s seconds ---" % (time.time() - start_time))
			#pickle.dump(Xs, open('XsDataFile.pkl', 'wb')) 






