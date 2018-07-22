"""
ClassProperties.py

Purpose:
    Get mean and stdev properties for classes, and 
    give new label indices by SST

Input: 
    address = root location
    n_comp = number of classes/components in GMM
Output:
    - data frame with everything, including SST sorted labels
       (saved as pickle file)
    - old2new: new class indices (sorted by SST)

"""

# import relevant modules
import glob
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import Print
import pickle

def main(address, runIndex, n_comp):
    print("ClassProperties.main()")

    # set paths
    floc = address + 'Data_store/CentredAndUncentred/' 
    labloc_unsorted = address + 'Data_store/Labels/Labels_unsorted.csv'
    frame_store = address + 'Objects/AllProfiles.pkl'

    # find all csv files
    allFiles = glob.glob(floc + "*.csv") 
    frame = pd.DataFrame()

    # read in label data - now passed as an argument
    df0 = pd.read_csv(labloc_unsorted, index_col=None, header=0)
    df1 = df0['label']
    labels = df1.values

    # read depth levels
    depths_retained = Print.readDepth(address, runIndex)

    # load posterior probabilities for each class
    class_number_array = None
    class_number_array = np.arange(0,n_comp).reshape(-1,1)
    lon_pp,lat_pp,dynHeight_pp,varTime_pp,post_prob = \
    Print.readPosteriorProb(address, runIndex, class_number_array)

    # read in T,S data. stack together with label data
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0)
        c = np.column_stack((df.values,labels))
        list_.append(c)

    # stack depths as new dimension / shape is (profile number, variable, depth)
    # where variables are lon lat dynHeight Tint Tint_centred Sint Time Label
    allOne = np.dstack(list_)
    numberOfProfiles, numberOfVars, numberOfDepths = allOne.shape

    # make a pandas dataframe that can be easily split
    print('ClassProperties.main() : creating data frame (this may take a while)')
    d = []
    for i in range(0,numberOfProfiles):
        for k in range(0,numberOfDepths):
            d.append({'profile_index': i, 
                      'depth_index': k,
                      'longitude': allOne[i,0,k],
                      'latitude': allOne[i,1,k],
                      'pressure': depths_retained[k],
                      'dynamic_height': allOne[i,2,k],
                      'temperature': allOne[i,3,k],
                      'temperature_standardized': allOne[i,4,k],
                      'salinity': allOne[i,5,k],
                      'time': allOne[i,6,k],
                      'class': int(allOne[i,7,k]),
                      'posterior_probability': np.max(post_prob[i,:])})
    allDF = pd.DataFrame(d)        

    # clear some memory by getting rid of variables
    del allOne
    del post_prob

    # read the pickle file data frame (for testing only)
#   allDF = pd.read_pickle(frame_store, compression='infer')

    # surface only
    surfaceDF = allDF[allDF.pressure == 15]
    surfaceDFg = surfaceDF.groupby(['class'])

    # sea surface properties
    SST_medians = surfaceDFg['temperature'].median()

    # sort by temperature (these are the new class numbers)
    old2new = np.argsort(SST_medians.values) 

    # construct dictionary to replace old class numbers with new ones
    di = dict(zip(range(0,n_comp),old2new))

    # add sorted class numbers as a new column
    allDF['class_sorted']=allDF['class'].map(di)

    # save allDF pickle object for later use
    print('ClassProperties.main(): pickling data frame')
    allDF.to_pickle(frame_store, compression='infer')

    # write some summaries to csv
    print('ClassProperties.main(): writing summaries')
    allDFgrouped = allDF.groupby('class_sorted')
    for column in allDF:
        fname = address + 'Results/' + column + '_stats.csv'
        tmp = allDFgrouped[column].describe()
        tmp.to_csv(fname)

#######################################################################

def sortLabels(oldLabels,old2new):

    # this vector will store the sorted labels 
    sortedLabels = np.zeros_like(oldLabels)
    
    # update the labels based on old2new
    for nprofile in range(0,len(oldLabels)):
        sortedLabels[nprofile] = old2new[int(oldLabels[nprofile])]

    # let's keep python indexing right up until plotting
    # shift indices by one (from python to conventional)
    # sortedLabels = sortedLabels + 1
 
    return sortedLabels    

#######################################################################

