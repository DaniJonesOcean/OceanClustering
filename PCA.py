# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:47:02 2017

@author: harryholt

PCA.py

Purpose:
    - Use trainingdataset to reduce the number of dimensions in the problem
    - Transform the full dataset onto these reduced dimensions
    - Store the PCA object and the information about the eigenvectors it has created
    
"""

import pickle
from sklearn import decomposition
from scipy.interpolate import PPoly
import numpy as np
import scipy as sp
import time

import Print

start_time = time.clock()

def create(address, runIndex, n_dimen, use_fPCA):
    print("Entering function PCA.create")
    """ This function takes the training dataset and creates the PCA object,
    whilst returning the transfromed dataset """
    
    # load depth
    depth = None
    depth = Print.readDepth(address, runIndex)
    
    # load training data
    lon_train, lat_train, dynHeight_train, Tint_train_array, X_train_array, \
            Sint_train_array, varTime_train = \
            None, None, None, None, None, None, None
    lon_train, lat_train, dynHeight_train, Tint_train_array, X_train_array, \
            Sint_train_array, varTime_train = \
            Print.readLoadFromFile_Train(address, runIndex, depth)
    
    # start the PCA process
    pca, pca_store, X_pca_train, variance_sum = \
            None, None, None, None
    pca, pca_store, X_pca_train, variance_sum  = \
            PrincipalComponentAnalysis( address, runIndex, \
                                        X_train_array, n_dimen, \
                                        use_fPCA)

    # variable col_reduced retains number of reduced dimensions
    col_reduced = np.size(X_pca_train,1)   
    
    """ Now we can print the reduced training dataset to a file """
#    print("Starting Print PCA")
# NOTE: I'm excluding Tint and Sint at this point in the code
    Print.printPCAToFile_Train(address, runIndex, lon_train, lat_train, dynHeight_train, \
                                X_pca_train, varTime_train, col_reduced)
    Print.printColreduced(address, runIndex, col_reduced)
    
def apply(address, runIndex):
    print("PCA.apply")
    # Load depth
    depth = None
    depth = Print.readDepth(address, runIndex)
    
    # Load col_reduced value
    col_reduced = None
    col_reduced = Print.readColreduced(address, runIndex)
    
    # Load full data array - X
    lon, lat, dynHeight, Tint_array, X_array, \
            Sint_array, varTime = None, None, None, None, None, None, None
    lon, lat, dynHeight, Tint_array, X_array, Sint_array, varTime  = \
            Print.readLoadFromFile(address, runIndex, depth)
            
    # Load PCA object
    pca = None
    with open(address+'Objects/PCA_object.pkl', 'rb') as input:
        pca = pickle.load(input)
    
    # tansform X to X_pca
    X_pca = None
    X_pca = pca.transform(X_array)    
    
    # Print X_pca to file
    Print.printPCAToFile(address, runIndex, lon, lat, dynHeight, X_pca, varTime, col_reduced)  
    del pca
    
###############################################################################    
    
def PrincipalComponentAnalysis(address, runIndex, X_train, \
                               n_dimen, use_fPCA):
    print("PCA.PrincipalComponentAnalysis")
    """ Initialises the PCA object and is called in create() """
    # initialise PCA object
    pca = decomposition.PCA(n_components = n_dimen) 
    # if use_fPCA=True, interpolate into BSpline basis   
    if use_fPCA==True:
        X_train = convert2Bspline(depth,X_train)

    # fit the PCA to the training data 
    pca.fit(X_train)   
    # transform the training data to reduced PCA/EOF space
    X_pca_train = pca.transform(X_train)    
    # get the variance explained by each principal component
    variance_sum = np.cumsum(pca.explained_variance_ratio_)
   
    # store the results as a PKL object 
    pca_store = address+"Objects/PCA_object.pkl"    
    with open(pca_store, 'wb') as output:
        pcaObject = pca
        pickle.dump(pcaObject, output, pickle.HIGHEST_PROTOCOL)
    del pcaObject    
    
    return pca, pca_store, X_pca_train, variance_sum

print('PCA runtime = ', time.clock() - start_time,' s')

###############################################################################  

def convert2Bspline(depth, X_train_in):
    print('Converting to Bspline basis')

    # fPCA applies PCA to the coefficients of the Bspline model

    # initialise empty array
    X_train_out = np.empty([X_train_in.shape[0],X_train_in.shape[1]])

    # create spl object, evaluate on pressure levels
    for nprof in range(0,X_train_in.shape[0]):
        x = depth
        y = X_train_in[nprof,:]
        tck = sp.interpolate.splrep(x, y)
        F = sp.interpolate.PPoly.from_spline(tck)
        Bcoeff = F.c
        Bknot = F.x 
        #_train_out[nprof,:] = y2

    # return B-spline basis points
    return X_train_out
     



