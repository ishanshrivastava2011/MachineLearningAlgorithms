#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:36:31 2017

@author: ishanshrivastava
"""

import numpy as np
import pandas as pd
from scipy import sparse   


def sigmoid(WsXs):
    return 1 / (1 + np.exp(-WsXs))

def addIntercept(Xs):
    W0 = np.ones((Xs.shape[0], 1))
    Xs = np.hstack((W0, Xs))
    return Xs

def myLogisticRegression(Xs_train,Ys_train,learning_rate,num_steps):
    Xs_train = addIntercept(Xs_train) 
    Xs_train = sparse.csr_matrix(Xs_train)
    Ws = np.zeros(Xs_train.shape[1])
    for step in range(num_steps):
        WsXs = Xs_train.dot(Ws)
        probOf1 = sigmoid(WsXs)
        error = Ys_train - probOf1
        Ws += learning_rate * Xs_train.T.dot(error)
    return Ws

def getPredictions(Xs_test,Ws):
    Xs_test = addIntercept(Xs_test)
    Xs_test = sparse.csr_matrix(Xs_test)
    predictions = np.round(sigmoid(Xs_test.dot(Ws)))
    return predictions