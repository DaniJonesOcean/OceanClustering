#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:48:50 2017

@author: harryholt

Reconstruct.py

Purpose:
    - Reverse transform the PCA
    - This makes it easier to interpret the results
    - Load the Scaling object to uncentre the results
    - Print the GMM means, weights and covariances in real space
    - Print the Reconstructed train and full datasets

"""
import pickle
from sklearn import mixture
import numpy as np
import time

import Print

start_time = time.clock()

def gmm_reconstruct(address, run, n_comp):
    print("Reconstruct.gmm_reconstruct")
    # Load the pca object for the inverse transform
    pca = None
    with open(address+'Objects/PCA_object.pkl', 'rb') as input:
        pca = pickle.load(input)
    
    # Load the scaled object for the uncentering 
    stand = None
    with open(address+"Objects/Scale_object.pkl", 'rb') as input:
        stand = pickle.load(input)
    
    # Load col_reduced value
    col_reduced = None
    col_reduced = Print.readColreduced(address, run)
    col_reduced_array = np.arange(col_reduced)
    
    # Load depth
    depth = None
    depth = Print.readDepth(address, run)
    
    # Load the gmm properties
    gmm_weights, gmm_means, gmm_covariances = None, None, None
    gmm_weights, gmm_means, gmm_covariances = \
            Print.readGMMclasses(address, run, col_reduced_array, 'reduced')
    
    """ Finished loading, now inverse transform and print """
    
    # Inverse transform gmm properties
    weights, means, covariances = None, None, None
    weights = gmm_weights
    means = pca.inverse_transform(gmm_means)
    covariances = pca.inverse_transform(gmm_covariances)
    
    # Print the results to a file
    class_number_array = np.arange(0,n_comp)
    Print.printGMMclasses(address, run, class_number_array, weights, means, covariances, depth, 'depth')
    
    # Un-centre the GMM class information
    weights_UC, means_UC, covariances_UC = None, None, None
    weights_UC = weights
    means_UC = stand.inverse_transform(means)
    covariances_UC = stand.inverse_transform(covariances)
    Print.printGMMclasses(address, run, class_number_array, weights_UC, means_UC, covariances_UC, depth, 'uncentred')

    
    del pca, stand
###############################################################################
    
def train_reconstruct(address, run):
    print("Reconstruct.train_reconstruct")
    # Load the pca object for the inverse transform
    pca = None
    with open(address+'Objects/PCA_object.pkl', 'rb') as input:
        pca = pickle.load(input)
    # Load the scaled object for the uncentering 
    stand = None
    with open(address+"Objects/Scale_object.pkl", 'rb') as input:
        stand = pickle.load(input)
        
    # Load col_reduced value
    col_reduced, col_reduced_array = None, None
    col_reduced = Print.readColreduced(address, run)
    col_reduced_array = np.arange(col_reduced)
    
    # Load depth
    depth = None
    depth = Print.readDepth(address, run)
    
    # Load the training varaible
    lon_train, lat_train, dynHeight_train, X_train_array, varTime_train = None, None, None, None, None
    lon_train, lat_train, dynHeight_train, X_train_array, varTime_train = Print.readPCAFromFile_Train(address, run, col_reduced)
    
    # Reconstruct
    XRC_train = None     # R = reconstructed, C = centred
    XRC_train = pca.inverse_transform(X_train_array)
    
    # Uncentre
    XR_train = None          # R = reconstructed
    XR_train = stand.inverse_transform(XRC_train)
    
    # Print the results to a file
    Print.printReconstruction(address, run, lon_train, lat_train, dynHeight_train, XR_train, XRC_train, varTime_train, depth, True)
    
def full_reconstruct(address, run):
    print("Reconstruct.full_reconstruct")
    # Load the pca object for the inverse transform
    pca = None
    with open(address+'Objects/PCA_object.pkl', 'rb') as input:
        pca = pickle.load(input)
    
    # Load the scaled object for the uncentering 
    stand = None
    with open(address+"Objects/Scale_object.pkl", 'rb') as input:
        stand = pickle.load(input)
    
    # Load col_reduced value
    col_reduced, col_reduced_array = None, None
    col_reduced = Print.readColreduced(address, run)
    col_reduced_array = np.arange(col_reduced)
    
    # Load depth
    depth = None
    depth = Print.readDepth(address, run)
    
    # Load the training varaible
    lon, lat, dynHeight, X_array, varTime = None, None, None, None, None
    lon, lat, dynHeight, X_array, varTime = Print.readPCAFromFile(address, run, col_reduced)
    
    # Reconstruct
    XRC = None     # R = reconstructed, C = centred
    XRC = pca.inverse_transform(X_array)
    
    # Uncentre
    XR = None          # R = reconstructed
    XR = stand.inverse_transform(XRC)
    
    # Print the results to a file
    Print.printReconstruction(address, run, lon, lat, dynHeight, XR, XRC, varTime, depth, False)
    
    

print('Reconstruct runtime = ', time.clock() - start_time,' s')
