# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:36:46 2017

@authors: harryholt, DanJonesOcean

Main.py

This script is the central script for my GMM program. It coordinates:
    Load.py
    Print.py
    Plot.py
    GMM.py
    PCA.py
    Bic.py
    Reconstruct.py

The intention is to have a program, comprised of a few modules, which can take
a data set, select a training dataset, a test data set and return:
    GMM model (Classes)
    Labels and probabilities

The should be flexibility in seclecting the Data set, the locations of the
input and output files, the ML algorithm we want to use and the option to loop
over the files/modules themselves
"""

""" Initial setup """

# import modules
import time
import numpy as np
import scipy as sp
import Load, Print, PCA, GMM, Reconstruct, Bic
import Plot

# start the clock (performance timing)
start_time = time.clock()

# set run mode (BIC, GMM, or Plot)
# -- BIC = calculates BIC scores for a range of classes
# -- GMM = performs GMM procedure (no BIC, no plots)
# -- Plot = plots the results (located in address+ploc)
run_mode = "Plot"

# plot ACC fronts 
plotFronts = False

# set parameters
n_comp = 8           # number of classes in GMM object
n_dimen = 0.999      # amount of variance retained in PCA
cov_type = 'full'    # covariance type (full, tied, diag, or spherical)

# put here for a quick fix 
# we can get rid of this variable in a later version of the code
runIndex = None

# root location of program 
address = "/data/expose/OceanClustering/"     
# location of data to plot (make sure n_comp is correct)
ploc = address + "n08_classes/"
# location of raw data file
filename_raw_data = "/data/expose/OceanClustering/Data_in/SO_Argo_all.mat"  
# root location of ACC front files
address_fronts = "/data/expose/OceanClustering/Data_in/Fronts/"

# define some flags relevant for BIC calculations
# if run_bic = True, program will calculate BIC scores
run_bic = (run_mode=="BIC")       
if run_bic:
    subsample_bic = "uniform"
    repeat_bic = 50
    max_groups = 20     # highest number of GMM classes tested
    grid_bic = 4        # size of cell in lat/lon degrees
    conc_bic = 1        # number of samples from each grid
    size_bic = 1100     # ideal size of the BIC training set

subsample_uniform = True  # indicates how the training dataset is selected
subsample_random = False  # indicates how the training dataset is selected
subsample_inTime = False

# declare some empty variables
grid, conc, fraction_train, inTime_start, inTime_finish = \
None, None, None, None, None

if subsample_uniform:
    grid = 1        # size of cell in lat/lon degrees
    conc = 1        # number of samples from each grid
if subsample_random:
    # size of training dataset as a fraction of whoel dataset
    fraction_train = 0.1  
if subsample_inTime:
    inTime_start = 0.0      # WRONG AT THE MOMENT
    inTime_finish = 200.0   # WRONG AT THE MOMENT

# cutoff values 
# - profiles/depths with > NaN percentages will be removed
fraction_nan_samples = 16.0 
fraction_nan_depths = 32.0 
""" end of initialisation conditions """

###############################################################################
###############################################################################

""" Program """
def main(runIndex=None):
    print("Starting Main.main()")  
    
    # if it doesn't exist, create directory structure
    makeDirectoryStructure(address)

    # now start the GMM process
    Load.main(address, filename_raw_data, runIndex, subsample_uniform,\
              subsample_random, subsample_inTime, grid, conc, \
              fraction_train, inTime_start, inTime_finish,\
              fraction_nan_samples, fraction_nan_depths, cov_type,\
              run_bic=False)

    # loads data, selects Train, cleans, centres/standardises, prints
    PCA.create(address, runIndex, n_dimen)     
    GMM.create(address, runIndex, n_comp, cov_type)   
    PCA.apply(address, runIndex)                   
    GMM.apply(address, runIndex, n_comp)     
    
    # reconstruction (back into depth space)
    Reconstruct.gmm_reconstruct(address, runIndex, n_comp)  
    Reconstruct.full_reconstruct(address, runIndex)
    Reconstruct.train_reconstruct(address, runIndex)
    
# function that only carries out plotting 
def mainPlot(address, address_fronts, runIndex, n_comp, plotFronts):
    Plot.plotMapCircular(address, address_fronts, runIndex, n_comp)
#   Plot.plotByDynHeight(address, address_fronts, runIndex, n_comp)
    Plot.plotPosterior(address, address_fronts, runIndex, n_comp, plotFronts=False)
    Plot.plotProfileClass(address, runIndex, n_comp, 'uncentred')
    Plot.plotProfileClass(address, runIndex, n_comp, 'depth')
    Plot.plotGaussiansIndividual(address, runIndex, n_comp, 'reduced')
    Plot.plotProfile(address, runIndex, 'original')
    Plot.plotProfile(address, runIndex, 'uncentred')
    Plot.plotWeights(address, runIndex)

""" If directory structure does not exist, create it """
def makeDirectoryStructure(address):
    import os
    # make Data_store
    mydir = address+"Data_store/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make Objects directory
    mydir = address+"Objects/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make Plots directory
    mydir = address+"Plots/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make Results directory
    mydir = address+"Results/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"CentredAndUncentred/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"CentredAndUncentred_Train/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"CentredAndUncentred_Test/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"GMM_classes_depth/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"GMM_classes_reduced/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"GMM_classes_uncentred/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"Info/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"Labels/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"PCA/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"PCA_Train/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"Probabilities/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"Reconstruction/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    # make various Data_store subdirectories
    mydir = address+"Data_store/"+"Reconstruction_Train/"
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    
""" Program starts here """
""" Process
    - Choose between BIC and GMM
    - Load, clean, select train
    - Centre train and then USE result to centre overall profiles
    - PCA train and then use result on test/overall array
    - Calculate GMM and store the object
    - Calculate the Labels for the training and test data
    - Print the results to a file for later plotting
    - Score the accuracy of the model using Scotts function
    - Plot the result
    """

# main program loop 
if (run_mode=="BIC"):
	Bic.main(address, filename_raw_data,subsample_bic, repeat_bic, max_groups, \
        	grid_bic, conc_bic, size_bic, n_dimen, fraction_nan_samples, \
        	fraction_nan_depths, cov_type)
	Plot.plotBIC(address, repeat_bic, max_groups)
elif (run_mode=="GMM"):
	main()
elif (run_mode=="Plot"):
	mainPlot(ploc, address_fronts, runIndex, n_comp, plotFronts) 
else:
	print('Parameter run_mode not set properly. Check Main.py')

# print runtime for performance
print('Main runtime = ', time.clock() - start_time,' s')
    
