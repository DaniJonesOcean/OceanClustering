# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:14:47 2017

@author: harryholt

Bic.py

Purpose:
    - Calculate the most appropriate number of components for the GMM process.
    - Do this by testing different training datasets selecting
    - loop over these 50 times and calculate a mean and standard deviation for the 

"""
# Import the various modules necessary
import time
import numpy as np
from sklearn import preprocessing
from sklearn import mixture
from sklearn import decomposition
from math import log10, floor

import Load
import Print

start_time = time.clock()

def main(address, filename_raw_data, subsample_bic, repeat_bic, max_groups, grid_bic,\
         conc_bic, size_bic, n_dimen, fraction_nan_samples, fraction_nan_depths):
    
    bic_many = np.ones((repeat_bic,max_groups-1)) # Need to use (max_groups-1) as the bic runs from 1 to max_groups 
    n_lowest_array = np.zeros(repeat_bic)
    n_comp_array = None
    for i in range(0,repeat_bic):
        print("Starting ", i)
        bic = bic_oneRun(address, filename_raw_data, subsample_bic, repeat_bic, max_groups, grid_bic,\
                   conc_bic, size_bic, n_dimen, fraction_nan_samples, fraction_nan_depths)
        bic_many[i,:] = bic[0]
        n_lowest_array[i] = bic[1]
        if i == 0 :
            n_comp_array = bic[2]
        del bic
        print("finished ", i)
        
    # bic_many shape (repeat, n_comp_array)
        
    bic_stand = preprocessing.StandardScaler()
    bic_stand.fit(bic_many)
    bic_mean = bic_stand.mean_
    bic_stdev = np.sqrt(bic_stand.var_)

    # Calculate the most appropriate number of components from the bic_scores
    def round_sig(x, x_2, sig=2):
        return round(x, sig-int(floor(log10(abs(x_2))))-1)
        
    n_stand = preprocessing.StandardScaler()
    n_stand.fit(n_lowest_array.reshape(-1,1))
    n_mean = n_stand.mean_[0]
    n_stdev = np.sqrt(n_stand.var_[0])
    
    n_mean = round_sig(n_mean, n_stdev)
    n_stdev = round_sig(n_stdev, n_stdev)
    
    # Alternative way of calculating the minimum number of components
    min_index = np.where(bic_mean==bic_mean.min())[0]
    n_min = (n_comp_array[min_index])[0]
             
    # Print to file
    Print.printBIC(address, repeat_bic, bic_many, bic_mean, bic_stdev, n_mean, n_stdev, n_min)
  
###############################################################################
def bic_oneRun(address, filename_raw_data, subsample_bic, repeat_bic, max_groups, grid_bic,\
         conc_bic, size_bic, n_dimen, fraction_nan_samples, fraction_nan_depths):

    # Load the training data
    lon_train, lat_train, Tint_train, varTrain_centre, Sint_train, varTime_train \
            = None, None, None, None, None, None
    lon_train, lat_train, Tint_train, varTrain_centre, Sint_train, varTime_train \
        = Load.main(address, filename_raw_data, None, subsample_bic, False,\
         False, grid_bic, conc_bic, None, None, None,\
         fraction_nan_samples, fraction_nan_depths, run_Bic=True)
    
    # Calculate PCA
    pca, X_pca_train = None, None
    pca = decomposition.PCA(n_components = n_dimen)     # Initialise PCA object
    pca.fit(varTrain_centre)                         # Fit the PCA to the training data
    X_pca_train = pca.transform(varTrain_centre) 
    del pca
    
    # Run BIC for GMM with different number of components
    # bic_values contains the array of scores for the different n_comp
    # n_lowest is the lowest n for each repeat
    # n_comp_array is from 0 to max_groups in integers
    bic_values, n_lowest, n_comp_array = None, None, None
    bic_values, n_lowest, n_comp_array = bic_calculate(X_pca_train, max_groups)
    
    return bic_values, n_lowest, n_comp_array

###############################################################################
def bic_calculate(X, max_groups):
#    print("BIC X shape = ",X.shape)
    X = X.reshape(-1,1)
#    print("BIC X.reshape shape = ", X.shape)
    lowest_bic, bic_score = np.infty, []
    n_components_range = np.arange(1, max_groups)
    for n_components in n_components_range:
#        print("BIC n_comp = ",n_components)
        gmm = mixture.GaussianMixture(n_components = n_components, covariance_type = 'diag')
        gmm.fit(X)
        bic_score.append(gmm.bic(X))
        if bic_score[-1] < lowest_bic:
            lowest_bic = bic_score[-1]
            lowest_n = n_components
    bic_score = np.asarray(bic_score).reshape(1,-1)
    return bic_score, lowest_n, n_components_range

print('Bic runtime = ', time.clock() - start_time,' s')