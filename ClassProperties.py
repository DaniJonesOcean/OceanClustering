"""
ClassProperties.py

Purpose:
    Get mean and stdev properties for classes, and 
    give new label indices by SST

Input: 
    address = root location
    n_comp = number of classes/components in GMM
Output:
    - surface mean and standard deviation class properties
       (saved as pickle file)
    - new class indices (sorted by SST)

"""

# import relevant modules
import glob
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import Print
import pickle

def main(address,n_comp):

    # set paths
    floc = address + 'Data_store/CentredAndUncentred/' 
    labloc = address + 'Data_store/Labels/Labels.csv'
    propery_store = address + 'Objects/ClassProperties.pkl'

    # find all csv files
    allFiles = glob.glob(floc + "*.csv") 
    frame = pd.DataFrame()

    # read in label data
    df0 = pd.read_csv(labloc, index_col=None, header=0)
    df1 = df0['label']
    labels = df1.values

    # read in T,S data. stack together with label data
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0)
        c = np.column_stack((df.values,labels))
        list_.append(c)

    # stack depths as new dimension
    # shape is (profile number, variable, depth)
    # where variables are lon lat dynHeight Tint Tint_centred Sint Time Label
    allOne = np.dstack(list_)
    numberOfProfiles, numberOfVars, numberOfDepths = allOne.shape

    # make a list of dictionaries, with array entries
    profiles_list = []
    for nprof in range(0,numberOfProfiles-1):
        A = {'lon':allOne[nprof,0,0], \
             'lat' : allOne[nprof,1,0], \
             'dynHeight' : allOne[nprof,2,0], \
             'T' : allOne[nprof,3,:], \
             'T_cent' : allOne[nprof,4,:], \
             'S' : allOne[nprof,5,:], \
             'Time' : allOne[nprof,6,0], \
             'class' : allOne[nprof,7,0] }
        # append dictionary to list
        profiles_list.append(A)

    # declare empty dictionary
    lon_by_class = {}
    lat_by_class = {}
    dynHeight_by_class = {}
    T_by_class = {}
    T_cent_by_class = {}
    S_by_class = {}
    Time_by_class = {}
    # add new keys to dictionary 
    for kclass in range(0,n_comp):
        lon_by_class[kclass] = []
        lat_by_class[kclass] = []
        dynHeight_by_class[kclass] = []
        T_by_class[kclass] = []
        T_cent_by_class[kclass] = []
        S_by_class[kclass] = []
        Time_by_class[kclass] = []

    for kclass in range(0,n_comp):
        for item in [i for i in range(0,numberOfProfiles-1) \
                     if profiles_list[i]['class']==kclass]:
            # each list entry is a vector
            lon_by_class[kclass].append(profiles_list[item]['lon'])
            lat_by_class[kclass].append(profiles_list[item]['lat'])
            dynHeight_by_class[kclass].append(profiles_list[item]['dynHeight'])
            T_by_class[kclass].append(profiles_list[item]['T'])
            T_cent_by_class[kclass].append(profiles_list[item]['T_cent'])
            S_by_class[kclass].append(profiles_list[item]['S'])
            Time_by_class[kclass].append(profiles_list[item]['Time'])
	
    lon_mean = np.zeros(n_comp)
    lat_mean = np.zeros(n_comp)
    dynHeight_mean = np.zeros(n_comp)
    T_mean = np.zeros(n_comp)
    T_cent_mean = np.zeros(n_comp)
    S_mean = np.zeros(n_comp)
    Time_mean = np.zeros(n_comp)
    lon_std = np.zeros(n_comp)
    lat_std = np.zeros(n_comp)
    dynHeight_std = np.zeros(n_comp)
    T_std = np.zeros(n_comp)
    T_cent_std = np.zeros(n_comp)
    S_std = np.zeros(n_comp)
    Time_std = np.zeros(n_comp)

    # just some properties
    for kclass in range(0,n_comp):
        # format as ndarrays
        lon_arr = np.asarray(lon_by_class[kclass])
        lat_arr = np.asarray(lat_by_class[kclass])
        dynHeight_arr = np.asarray(dynHeight_by_class[kclass])
        T_arr = np.asarray(T_by_class[kclass])
        T_cent_arr = np.asarray(T_cent_by_class[kclass])
        S_arr = np.asarray(S_by_class[kclass])
        Time_arr = np.asarray(Time_by_class[kclass])
        # nanmeans and standard deviations
        lon_mean[kclass] = np.nanmean(lon_arr,axis=0)
        lon_std[kclass] = np.std(lon_arr,axis=0)
        lat_mean[kclass] = np.nanmean(lat_arr,axis=0)
        lat_std[kclass] = np.std(lat_arr,axis=0)
        dynHeight_mean[kclass] = np.nanmean(dynHeight_arr,axis=0)
        dynHeight_std[kclass] = np.std(dynHeight_arr,axis=0)
        T_mean[kclass] = np.nanmean(T_arr[:,0],axis=0)
        T_std[kclass] = np.std(T_arr[:,0],axis=0)
        T_cent_mean[kclass] = np.nanmean(T_cent_arr[:,0],axis=0)
        T_cent_std[kclass] = np.std(T_cent_arr[:,0],axis=0)
        S_mean[kclass] = np.nanmean(S_arr[:,0],axis=0)
        S_std[kclass] = np.std(S_arr[:,0],axis=0)
        Time_mean[kclass] = np.nanmean(Time_arr,axis=0)
        Time_std[kclass] = np.std(Time_arr,axis=0)

    # sort by temperature (these are the new class numbers)
    newClassIndices = np.argsort(T_mean) 
    #print(T_mean[newClassIndices])

    # save properties as pkl object
    with open(property_store, 'wb') as fileout:
        pickle.dump([lon_mean, lon_std, \
                     lat_mean, lat_std, \
                     dynHeight_mean, dynHeight_std, \
                     T_mean, T_std \
                     T_cent_mean, T_cent_std \
                     S_mean, S_std, \
                     Time_mean, Time_std], fileout)             

    # return the new class numbers
    return newClassIndices
