#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:26:20 2017

@author: ishanshrivastava
"""
import numpy as np
import scipy.io as spio
import random
from sklearn.metrics import pairwise_distances
from scipy import sparse
import matplotlib.pyplot as plt
import os

path=os.getcwd()
#'/' symbol works in Mac and linux. Change it for Windows. Automation Required here.
mat = spio.loadmat(path+'/'+'kmeans_data.mat', squeeze_me=True)
mdata = mat['data']
num_features = len(mat['data'].T)
mdataSparseMatrix = sparse.csr_matrix(mdata)

def getInitialClusters_KmeansPlusPlus(num_clusters):
    listOfPoints= list(mdata)
    cluster_centers = random.sample(listOfPoints, 1)
    while len(cluster_centers) < num_clusters:
        centersFoundTillNow = cluster_centers
        distances = np.array([min([np.linalg.norm(p-c)**2 for c in centersFoundTillNow]) for p in listOfPoints])
        probabilities = distances/distances.sum()
        cummilative_probabilities = probabilities.cumsum()
        nextCenter_index = np.where(cummilative_probabilities >= random.random())[0][0]
        cluster_centers.append(listOfPoints[nextCenter_index])
    return cluster_centers

def M_Step_calculateNewCenters(assignCluster,num_clusters,mdataSparseMatrix,num_features):
    newCenters = np.ndarray(shape=(num_clusters,num_features))
    for k in range(0,num_clusters):
        newCenters[k]=np.mean(mdataSparseMatrix[np.where(assignCluster == k)[0]].T,axis=1).T
    return newCenters

def calculateCost(newCenters,assignCluster,num_clusters):
    cost=0
    for k in range(0,num_clusters):
        kth_ClusterPoints=mdataSparseMatrix[np.where(assignCluster == k)[0]]
        cost+=sum(pairwise_distances(newCenters[k],kth_ClusterPoints)[0])
    return cost

#distanceToAllClusterCenters will contain distances of each point to each cluster in a column
#assignCluster is a list of cluster associations for each data point. Index of this list gives
#index of the data point in the orginal dataset.
def E_Step_AssignClusters(cluster_centers, mdataSparseMatrix):
    distanceToAllClusterCenters = pairwise_distances(cluster_centers, mdataSparseMatrix)
    assignCluster=np.argmin(distanceToAllClusterCenters, axis=0)
    return assignCluster

costList_kmeansPlusPlus = [0]*len(range(2,11))
for k in range(2,11):
    num_clusters = k #= 3
    cluster_centers = getInitialClusters_KmeansPlusPlus(num_clusters)
    assignCluster = E_Step_AssignClusters(cluster_centers, mdataSparseMatrix)
    newCenters = sparse.csr_matrix(M_Step_calculateNewCenters(assignCluster,num_clusters,mdataSparseMatrix,num_features))
    sumChangeInCenters = pairwise_distances(cluster_centers, newCenters).trace()

    while sumChangeInCenters !=0:
        oldCenters = newCenters
        assignCluster = E_Step_AssignClusters(newCenters, mdataSparseMatrix)
        newCenters = sparse.csr_matrix(M_Step_calculateNewCenters(assignCluster,num_clusters,mdataSparseMatrix,num_features))
        sumChangeInCenters = pairwise_distances(oldCenters, newCenters).trace()
    costList_kmeansPlusPlus[k-2]=calculateCost(newCenters,assignCluster,num_clusters)
    print('Cost for K: '+str(k)+' is '+str(costList_kmeansPlusPlus[k-2]))

plt.ylim(2000, 4000)
plt.title("Cost vs. K")
plt.ylabel("Cost")
plt.xlabel("K")
plt.scatter(range(2,11),costList_kmeansPlusPlus)
plt.plot(range(2,11),costList_kmeansPlusPlus)
plt.show()
#plt.savefig('KMeans++ Analysis Cost vs. K.jpg')
