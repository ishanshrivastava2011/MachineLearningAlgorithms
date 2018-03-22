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

def getInitialRandomClusterCenters(num_clusters,mdata):
    cluster_centers = [len(mdata)*10]*num_clusters
    secure_random = random.SystemRandom(1)
    dataPoints = range(0,len(mdata))
    for i in range(0,num_clusters):
        cluster_center = secure_random.choice(dataPoints)
        if cluster_center not in cluster_centers:
            cluster_centers[i]=cluster_center
        else:
            i-=1
    return mdataSparseMatrix[cluster_centers]

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
    
costList = [0]*len(range(2,11))
for k in range(2,11):
    num_clusters = k
    mdataSparseMatrix = sparse.csr_matrix(mdata)
    cluster_centers = getInitialRandomClusterCenters(num_clusters,mdata)
    assignCluster = E_Step_AssignClusters(cluster_centers, mdataSparseMatrix)
    newCenters = sparse.csr_matrix(M_Step_calculateNewCenters(assignCluster,num_clusters,mdataSparseMatrix,num_features))
    sumChangeInCenters = pairwise_distances(cluster_centers, newCenters).trace()

    while sumChangeInCenters !=0:
        oldCenters = newCenters
        assignCluster = E_Step_AssignClusters(newCenters, mdataSparseMatrix)
        newCenters = sparse.csr_matrix(M_Step_calculateNewCenters(assignCluster,num_clusters,mdataSparseMatrix,num_features))
        sumChangeInCenters = pairwise_distances(oldCenters, newCenters).trace()
    costList[k-2]=calculateCost(newCenters,assignCluster,num_clusters)
    print('Cost for K: '+str(k)+' is '+str(costList[k-2]))
    
plt.ylim(2000, 4000)
plt.title("Cost vs. K")
plt.ylabel("Cost")
plt.xlabel("K")
plt.scatter(range(2,11),costList)
plt.plot(range(2,11),costList)
plt.show()