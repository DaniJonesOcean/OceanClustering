# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:03:46 2017

@author: harryholt

Print.py

Purpose:
    - Print the data to files is csv format
    - Reload data for later use
    
"""
import time
import numpy as np
import csv

start_time = time.clock()

separator = ','

###############################################################################

# depth Printing
def printDepth(address, runIndex, depth):
    print("Print.printDepth")
    filename = address+"Data_store/Info/Depth_retained.csv"
    file = open(filename,'w')
    data = depth.reshape(len(depth),1)
    writer = csv.DictWriter(file, fieldnames = ['Depth'], \
                            delimiter = separator)
    writer.writeheader()
    writer = csv.writer(file, delimiter=separator)
    for line in data:
        writer.writerow(line)
    file.close() 
    
def readDepth(address, runIndex):
    print("Print.readDepth")
    filename = address+"Data_store/Info/Depth_retained.csv"
    depth = None
    head_number = 1
    csvfile = np.genfromtxt(filename, delimiter=",",\
                            skip_header=head_number)
    depth = csvfile[:]
    return depth

# Col_reduced Printing
def printColreduced(address, runIndex, col_reduced):
    print("Print.printColreduced")
    filename = address+"Data_store/Info/Col_reduced.csv"
    file = open(filename,'w')
    data = [col_reduced]
    writer = csv.DictWriter(file, fieldnames = ['Col_reduced'], \
                            delimiter = separator)
    writer.writeheader()
    writer = csv.writer(file, delimiter=separator)
    writer.writerow(data)
    file.close()
    
def readColreduced(address, runIndex):
    print("Print.readColreduced")
    filename = address+"Data_store/Info/Col_reduced.csv"
    col_reduced = None
    head_number = 1
    csvfile = np.genfromtxt(filename, delimiter=",",\
                            skip_header=head_number)
    col_reduced = int(csvfile)
    print("col_reduced = ", col_reduced)
    return col_reduced

    
###############################################################################

# BIC printing and loading
def printBIC(address, repeat_bic, bic_many, bic_mean, bic_stdev, \
             n_mean, n_stdev, n_min):
    print("Print.printBIC")
    print("SHAPE OF BIC MANY = ", bic_many.shape)
    data, columns = None, None
    
    # This section prints the information on components, minimum scores and means
    filename = address + "Data_store/Info/BIC_Info.csv"
    file = open(filename, 'w')
    columns = np.column_stack((n_mean, n_stdev, n_min))
    data = columns
    writer = csv.DictWriter(file, fieldnames = \
        ['n_mean_from_ave_nmin_individual','n_stdev','n_min_from_bic_mean'], \
        delimiter = separator)
    writer.writeheader()
    writer = csv.writer(file, delimiter=separator)
    for line in data:
        writer.writerow(line)    
    file.close()
    del filename, file
    
    # This section prints the bic scores
    for r in range(repeat_bic):
        filename = address + "Data_store/Info/BIC_r"+str(int(r)).zfill(3)+".csv"
        file = open(filename,'w')
        data, columns = None, None
        columns = np.column_stack((bic_many[r,:], bic_mean, bic_stdev))
        data = columns
        writer = csv.DictWriter(file, fieldnames = \
                 ['bic_many'+str(int(r)).zfill(3),'bic_mean','bic_stdev'], \
                 delimiter = separator)
        writer.writeheader()
        writer = csv.writer(file, delimiter=separator)
        for line in data:
            writer.writerow(line)    
        file.close()
        del filename, file

#######################################################################

def readBIC(address, repeat_bic):
    # reads the information on components, minimum scores and means
    n_mean, n_stdev, n_min = None, None, None
    head_number = 1
    filename = address+"Data_store/Info/BIC_Info.csv"
    csvfile = np.genfromtxt(filename, delimiter=",",\
                            skip_header=head_number)
    n_mean  = csvfile[0]
    n_stdev = csvfile[1]
    n_min   = csvfile[2]

    # Read the bic_scores
    bic_r, bic_many, bic_mean, bic_stdev = None, None, None, None
    head_number = 1

    for r in range(repeat_bic):
        filename = address+"Data_store/Info/BIC_r"+str(int(r)).zfill(3)+".csv"
        
        csvfile = np.genfromtxt(filename, delimiter=",",\
                                skip_header=head_number)
        
        if r == 0:
            bic_mean     = csvfile[:,1]
            bic_stdev     = csvfile[:,2]

        bic_r    = csvfile[:,0]

        if r == 0:
            bic_many        = bic_r.reshape(1, bic_r.size)
        else:
            bic_r = bic_r.reshape(1,bic_r.size)
            bic_many = np.vstack([bic_many, bic_r])
            
    print("SHAPE OF BIC MANY AFTER READING = ", bic_many.shape)

    return bic_many, bic_mean, bic_stdev, n_mean, n_stdev, n_min

###############################################################################

# load Printing
def printLoadToFile(address, runIndex, lon, lat, dynHeight, Tint, \
                    var_centre, Sint, varTime, depth ):
    print("Print.printLoadToFile")
    i = 0 
    for d in depth:
        filename = address+\
                   "Data_store/CentredAndUncentred/CentredAndUncentred_depth"+\
                   str(int(d)).zfill(3)+".csv"

        file = open(filename,'w')
        columns= np.column_stack((lon, lat, dynHeight, \
                                 Tint[:,i], var_centre[:,i], \
                                 Sint[:,i], varTime))
        data = columns
        writer = csv.DictWriter(file, fieldnames = \
                               ['lon','lat','dynHeight',\
                               'Tint_'+str(int(d)).zfill(3),'Tint_centred',\
                               'Sint','Time'], delimiter = separator)
        writer.writeheader()
        writer = csv.writer(file, delimiter=separator)
        for line in data:
            writer.writerow(line)    
        file.close()
        del filename, file
        
        i = i + 1

#######################################################################
        
def printLoadToFile_Train(address, runIndex, lon_train, lat_train, \
                          dynHeight_train, Tint_train, \
                          varTrain_centre, Sint_train, varTime_train,\
                          depth ):
    print("Print.printLoadToFile_Train")
    i = 0 
    for d in depth:
        filename_train = address+\
                         "Data_store/CentredAndUncentred_Train/CentredAndUncentred_Train_depth"\
                         +str(int(d)).zfill(3)+".csv"        

        file_train = open(filename_train,'w')
        columns_train = np.column_stack((lon_train, lat_train, \
                                        dynHeight_train, Tint_train[:,i], \
                                        varTrain_centre[:,i], Sint_train[:,i], \
                                        varTime_train))
        data_train = columns_train
        writer = csv.DictWriter(file_train, \
                        fieldnames = ['lon_train','lat_train','dynHeight_train',\
                                      'VAR_train_'+str(int(d)).zfill(3),\
                                      'Var_train_centred',\
                                      'Sint_train','Time_train'], \
                                       delimiter = separator)
        writer.writeheader()
        writer = csv.writer(file_train, delimiter=separator)    
        for line in data_train:
            writer.writerow(line)
        file_train.close() 
        del filename_train, file_train
        
        i = i + 1

#######################################################################
        
def printLoadToFile_Test(address, runIndex, lon_test, lat_test, dynHeight_test, Tint_test, \
                                varTest_centre, Sint_test, varTime_test, depth):
    print("Print.printLoadToFile_Test")
    i = 0 
    for d in depth:
        filename_test = address+\
           "Data_store/CentredAndUncentred_Test/CentredAndUncentred_Test_depth"+\
           str(int(d)).zfill(3)+".csv"        

        file_test = open(filename_test,'w')
        columns_test = np.column_stack((lon_test, lat_test, \
                       dynHeight_test, Tint_test[:,i], varTest_centre[:,i], \
                       Sint_test[:,i], varTime_test))
        data_test = columns_test
        writer = csv.DictWriter(file_test, \
                      fieldnames = ['lon_test','lat_test',\
                      'dynHeight_test','VAR_test_'+str(int(d)).zfill(3),\
                      'Var_test_centred','Sint_test','Time_test'], \
                      delimiter = separator)
        writer.writeheader()
        writer = csv.writer(file_test, delimiter=separator)    
        for line in data_test:
            writer.writerow(line)
        file_test.close() 
        del filename_test, file_test
        
        i = i + 1
    
#######################################################################

def readLoadFromFile(address, runIndex, depth):
    print("Print.readLoadFromFile")
    lon, lat, dynHeight, Tint, var, Sint, varTime = \
        None, None, None, None, None, None, None
    head_number = 1
    i = 0
    for d in depth:
#        print(i)
        filename = address+\
           "Data_store/CentredAndUncentred/CentredAndUncentred_depth"+\
           str(int(d)).zfill(3)+".csv"
        
        csvfile = np.genfromtxt(filename, delimiter=",",\
                      skip_header=head_number)
        if i == 0:
            lon     = csvfile[:,0]
            lat     = csvfile[:,1]
            dynHeight = csvfile[:,2]
            varTime = csvfile[:,6]

        Tint    = csvfile[:,3]
        var     = csvfile[:,4]
        Sint    = csvfile[:,5]

        if i == 0:
            Tint_array      = Tint.reshape(Tint.size,1)
            Sint_array      = Sint.reshape(Sint.size,1)
            X_array         = var.reshape(var.size,1)
        else:
            Tint = Tint.reshape(Tint.size,1)
            Sint = Sint.reshape(Sint.size,1)
            Tint_array = np.hstack([Tint_array, Tint])
            Sint_array = np.hstack([Sint_array, Sint])
            
            X = var.reshape(var.size,1)
            X_array = np.hstack([X_array, X])
        i = i + 1   

    return lon, lat, dynHeight, Tint_array, X_array, Sint_array, varTime

#######################################################################

def readLoadFromFile_Train(address, runIndex, depth):
    print("Print.readLoadFromFile_Train")
    lon_train, lat_train, dynHeight_train, Tint_train, varTrain, \
        Sint_train, varTime_train = None, None, None, None, None, None, None
    head_number = 1
    i = 0
    for d in depth:
#        print(i)
        filename_train = address+\
            "Data_store/CentredAndUncentred_Train/CentredAndUncentred_Train_depth"+\
            str(int(d)).zfill(3)+".csv"

        csvfile_train = np.genfromtxt(filename_train, delimiter=",",\
                        skip_header=head_number)
        if i == 0:
            lon_train     = csvfile_train[:,0]
            lat_train     = csvfile_train[:,1]
            dynHeight_train = csvfile_train[:,2]
            varTime_train = csvfile_train[:,6]

        Tint_train    = csvfile_train[:,3]
        varTrain     = csvfile_train[:,4]
        Sint_train    = csvfile_train[:,5]
    
        if i == 0:
            Tint_train_array = Tint_train.reshape(Tint_train.size,1)
            Sint_train_array = Sint_train.reshape(Sint_train.size,1)
            X_train_array   = varTrain.reshape(varTrain.size,1)
        else:
            Tint_train = Tint_train.reshape(Tint_train.size,1)
            Sint_train = Sint_train.reshape(Sint_train.size,1)
            Tint_train_array = np.hstack([Tint_train_array, Tint_train])
            Sint_train_array = np.hstack([Sint_train_array, Sint_train]) 
            
            X_train = varTrain.reshape(varTrain.size,1)
            X_train_array = np.hstack([X_train_array, X_train])
        
        i = i + 1

    return lon_train, lat_train, dynHeight_train, Tint_train_array, X_train_array, \
            Sint_train_array, varTime_train
            
#######################################################################

def readLoadFromFile_Test(address, runIndex, depth):
    print("Print.readLoadFromFile_Test")
    lon_test, lat_test, dynHeight_test, Tint_test, varTest, \
        Sint_test, varTime_test = None, None, None, None, None, None
    head_number = 1
    i = 0
    for d in depth:
#        print(i)
        filename_test = address+\
           "Data_store/CentredAndUncentred_Test/CentredAndUncentred_Test_depth"+\
           str(int(d)).zfill(3)+".csv"

        csvfile_test = np.genfromtxt(filename_test, delimiter=",",\
                       skip_header=head_number)
        if i == 0:
            lon_test     = csvfile_test[:,0]
            lat_test     = csvfile_test[:,1]
            dynHeight_test = csvfile_test[:,2]
            varTime_test = csvfile_test[:,6]

        Tint_test    = csvfile_test[:,3]
        varTest     = csvfile_test[:,4]
        Sint_test    = csvfile_test[:,5]
    
        if i == 0:
            Tint_test_array = Tint_test.reshape(Tint_test.size,1)
            Sint_test_array = Sint_test.reshape(Sint_test.size,1)
            X_test_array   = varTest.reshape(varTest.size,1)
        else:
            Tint_test = Tint_test.reshape(Tint_test.size,1)
            Sint_test = Sint_test.reshape(Sint_test.size,1)
            Tint_test_array = np.hstack([Tint_test_array, Tint_test])
            Sint_test_array = np.hstack([Sint_test_array, Sint_test]) 
            
            X_test = varTest.reshape(varTest.size,1)
            X_test_array = np.hstack([X_test_array, X_test])
        
        i = i + 1

    return lon_test, lat_test, dynHeight_test, Tint_test_array, X_test_array, \
            Sint_test_array, varTime_test

###############################################################################

# PCA Printing
def printPCAToFile(address, runIndex, lon, lat, dynHeight, \
                   X_pca, varTime, col_reduced):
    print("Print.printPCAToFile")
    for d in range(col_reduced):
        filename = address+"Data_store/PCA/PCA_reddepth"+str(d)+".csv"        

        file = open(filename, 'w')
        columns = np.column_stack((lon, lat, dynHeight, X_pca[:,d], varTime))
        data = columns
        writer = csv.DictWriter(file, \
            fieldnames = ['lon','lat','dynHeight','VAR_'+\
            str(int(d)).zfill(3),'Time'], delimiter = separator)
        writer.writeheader()
        writer = csv.writer(file, delimiter=separator)    
        for line in data:
            writer.writerow(line)
        file.close() 
        del filename, file

#######################################################################

def printPCAToFile_Train(address, runIndex, lon_train, lat_train, dynHeight_train, \
                                X_pca_train, varTime_train, col_reduced):
    print("Print.printPCAToFile_Train")
    for d in range(col_reduced):
        filename_train = address+\
           "Data_store/PCA_Train/PCA_Train_reddepth"+\
            str(d)+".csv"        

        file_train = open(filename_train,'w')
        columns_train = np.column_stack((lon_train, lat_train, \
            dynHeight_train, X_pca_train[:,d], varTime_train))
        data_train = columns_train
        writer = csv.DictWriter(file_train, fieldnames = \
            ['lon_train','lat_train','dynHeight_train','VAR_train_'+\
            str(int(d)).zfill(3),'Time_train'], delimiter = separator)
        writer.writeheader()
        writer = csv.writer(file_train, delimiter=separator)    
        for line in data_train:
            writer.writerow(line)
        file_train.close() 
        del filename_train, file_train

#######################################################################

def readPCAFromFile(address, runIndex, col_reduced):
    print("Print.readPCAFromFile")
    lon, lat, dynHeight, var, varTime= None, None, None, None, None
    head_number = 1
    for d in range(col_reduced):
        filename = address+"Data_store/PCA/PCA_reddepth"+str(int(d)).zfill(3)+".csv"

        csvfile = np.genfromtxt(filename, delimiter=",",skip_header=head_number)
        
        if d == 0:
            lon     = csvfile[:,0]
            lat     = csvfile[:,1]
            dynHeight = csvfile[:,2]
            varTime = csvfile[:,4]

        var     = csvfile[:,3]
    
        if d == 0:
            X_array   = var.reshape(var.size,1)
        else:
            X = var.reshape(var.size,1)
            X_array = np.hstack([X_array, X])
#    print("X_array.shape = ", X_array.shape)
#    print(lon.shape,lat.shape,varTime.shape)
    
    return lon, lat, dynHeight, X_array, varTime

#######################################################################
        
def readPCAFromFile_Train(address, runIndex, col_reduced):
    print("Print.readPCAFromFile_Train")
    lon_train, lat_train, dynHeight_train, varTrain, varTime_train = \
      None, None, None, None, None

    head_number = 1
    for d in range(col_reduced):
        filename_train = address+\
                         "Data_store/PCA_Train/PCA_Train_reddepth"+\
                         str(int(d)).zfill(3)+".csv"

        csvfile_train = np.genfromtxt(filename_train, \
                        delimiter=",",skip_header=head_number)
        
        if d == 0:
            lon_train     = csvfile_train[:,0]
            lat_train     = csvfile_train[:,1]
            dynHeight_train = csvfile_train[:,2]
            varTime_train = csvfile_train[:,4]

        varTrain     = csvfile_train[:,3]
    
        if d == 0:
            X_train_array   = varTrain.reshape(varTrain.size,1)
        else:
            X_train = varTrain.reshape(varTrain.size,1)
            X_train_array = np.hstack([X_train_array, X_train])
#    print("X_train_array.shape = ", X_train_array.shape)
#    print(lon_train.shape,lat_train.shape,varTime_train.shape)
    
    return lon_train, lat_train, dynHeight_train, X_train_array, varTime_train

###############################################################################

def printGMMclasses(address, runIndex, class_number_array, gmm_weights, gmm_means,\
                    gmm_covariances, depth_array, space):
    print("Print.printGMMclasses "+space)
    # space is either 'depth', 'reduced' or 'uncentred'
    # depth_number is either col_reduced or len(depth)
    i = 0
    for d in depth_array:
        filename_train = address+\
            "Data_store/GMM_classes_"+\
             space+"/GMM_classes_"\
             +space+str(int(d).zfill(3))+".csv"        

        file_train = open(filename_train,'w')
        columns_train = np.column_stack((class_number_array, \
                            gmm_weights, gmm_means[:,i], gmm_covariances[:,i]))
        data_train = columns_train
        writer = csv.DictWriter(file_train, \
            fieldnames = ['Class','Weights','Means_'+\
                          str(int(d)).zfill(3),'Covariances_'+\
                          str(int(d)).zfill(3)], delimiter = separator)
        writer.writeheader()
        writer = csv.writer(file_train, delimiter=separator)    
        for line in data_train:
            writer.writerow(line)
        file_train.close() 
        del filename_train, file_train
        i = i + 1

#######################################################################
        
def readGMMclasses(address, runIndex, depth_array, space):
    print("Print.readGMMclasses "+space)
    # space is either 'depth', 'reduced' or 'uncentred'
    # depth_number is either col_reduced or len(depth)
    gmm_weights, gmm_means_i, gmm_covariances_i = None, None, None
    head_number = 1
    i = 0
    for d in depth_array:
        filename_train = address+"Data_store/GMM_classes_"+\
                         space+"/GMM_classes_"+space+str(int(d)).zfill(3)+".csv"        

        csvfile_train = np.genfromtxt(filename_train, \
                        delimiter=",",skip_header=head_number)
        
        if i == 0:
            gmm_weights     = csvfile_train[:,1]
        # Note [:,0] would return the class number
        gmm_means_i     = csvfile_train[:,2]
        gmm_covariances_i = csvfile_train[:,3]
        
        if i == 0:
            gmm_means   = gmm_means_i.reshape(gmm_means_i.size,1)
            gmm_covariances   = gmm_covariances_i.reshape(gmm_covariances_i.size,1)
        else:
            gmm_means_i = gmm_means_i.reshape(gmm_means_i.size,1)
            gmm_covariances_i = gmm_covariances_i.reshape(gmm_covariances_i.size,1)
            gmm_means = np.hstack([gmm_means, gmm_means_i])
            gmm_covariances = np.hstack([gmm_covariances, gmm_covariances_i])
        i = i + 1
#    print("gmm_means.shape = ", gmm_means.shape)
#    print("gmm_covariances.shape = ", gmm_covariances.shape)
#    print(gmm_weights.shape)
    
    return gmm_weights, gmm_means, gmm_covariances

########################################################################

def printLabels(address, runIndex, lon, lat, dynHeight, varTime, labels):
    print("Print.printLabels")
    filename = address+"Data_store/Labels/Labels.csv"

    file = open(filename,'w')
    columns = np.column_stack(( lon, lat, dynHeight, varTime, labels ))
    data = columns
    writer = csv.DictWriter(file, fieldnames = \
             ['lon','lat','dynHeight','varTime','label'], delimiter = separator)
    writer.writeheader()
    writer = csv.writer(file, delimiter=separator)    
    for line in data:
        writer.writerow(line)
    file.close() 
    del filename, file

#######################################################################
    
def readLabels(address,runIndex):
    print("Print.readLabels")
    head_number = 1
    filename = address+"Data_store/Labels/Labels.csv"
        
    csvfile = np.genfromtxt(filename, delimiter=",",skip_header=head_number)
    lon, lat, dynHeight, varTime, labels = None, None, None, None, None
    
    lon = csvfile[:,0]
    lat = csvfile[:,1]
    dynHeight = csvfile[:,2]
    varTime = csvfile[:,3]
    labels = csvfile[:,4]
    
    return lon, lat, dynHeight, varTime, labels

###############################################################################
    
def printPosteriorProb(address, runIndex, lon, lat, dynHeight, \
                       varTime, post_prob, class_number_array):
    print("Print.printPosteriorProb")
    i = 0
    for class_number in class_number_array:
        filename = address+"Data_store/Probabilities/Post_prob_class"\
                          +str(class_number)+".csv"
        file = open(filename,'w')
        columns = np.column_stack(( lon, lat, dynHeight, varTime, post_prob[:,i] ))
        data = columns
        writer = csv.DictWriter(file, fieldnames = \
                 ['lon','lat','dynHeight','varTime','post_prob_class_'+\
                 str(class_number)], delimiter = separator)
        writer.writeheader()
        writer = csv.writer(file, delimiter=separator)    
        for line in data:
            writer.writerow(line)
        file.close() 
        del filename, file
        i = i + 1

#######################################################################
        
def readPosteriorProb(address,runIndex,class_number_array):
    print("Print.readPosteriorProb")
    head_number = 1
    post_prob = None
    for class_number in class_number_array:
        filename = address+\
                   "Data_store/Probabilities/Post_prob_class"+\
                   str(class_number)+".csv"

        csvfile = np.genfromtxt(filename, delimiter=",",\
                                skip_header=head_number)
        
        #lon, lat, dynHeight, varTime, post_prob_i = \
        # None, None, None, None, None
       
        # grab values from csvfile 
        lon = csvfile[:,0]
        lat = csvfile[:,1]
        dynHeight = csvfile[:,2]
        varTime = csvfile[:,3]
        post_prob_i = csvfile[:,4]

        if class_number == 0:
            post_prob  = post_prob_i.reshape(post_prob_i.size,1)
        else:            
            post_prob_i = post_prob_i.reshape(post_prob_i.size,1)
            post_prob = np.hstack([post_prob, post_prob_i])
    
    return lon, lat, dynHeight, varTime, post_prob
        
###############################################################################

def printReconstruction(address, runIndex, lon, lat, dynHeight, X, XC, \
                        varTime, depth, isTrain):
    print("Print.printReconstruction isTrain = "+str(isTrain))
    # isTrain is True or False
    i = 0
    for d in depth:
        filename = address+"Data_store/Reconstruction/Recon_depth"+\
                   str(int(d)).zfill(3)+".csv"
        if isTrain:
            filename = address+"Data_store/Reconstruction_Train/Recon_Train_depth"+\
                   str(int(d)).zfill(3)+".csv"
        file = open(filename,'w')
        columns = np.column_stack(( lon, lat, dynHeight, X[:,i], XC[:,i], varTime ))
        data = columns
        writer = csv.DictWriter(file, \
                 fieldnames = ['lon','lat','dynHeight',\
                               'X_'+str(int(d)).zfill(3), 'X_centred', \
                               'varTime'], delimiter = separator)
        writer.writeheader()
        writer = csv.writer(file, delimiter=separator)    
        for line in data:
            writer.writerow(line)
        file.close() 
        del filename, file
        i = i + 1
        
#######################################################################

def readReconstruction(address, runIndex, depth, isTrain):
    print("Print.readReconstruction isTrain = "+str(isTrain))
    # Function reads the Reconstructed XR, XRC, XR_Train, XRC_Train
    head_number = 1
    i = 0
    for d in depth:
        
        filename = address+\
                   "Data_store/Reconstruction/Recon_depth"+\
                   str(int(d)).zfill(3)+".csv"
        if isTrain:
            filename = address+\
                       "Data_store/Reconstruction_Train/Recon_Train_depth"+\
                       str(int(d)).zfill(3)+".csv"

        csvfile = np.genfromtxt(filename, delimiter=",",skip_header=head_number)
        if i == 0:
            lon     = csvfile[:,0]
            lat     = csvfile[:,1]
            dynHeight = csvfile[:,2]
            varTime = csvfile[:,5]

        var         = csvfile[:,3]
        var_centred = csvfile[:,4]

        if i == 0:
            X_array_centred = var_centred.reshape(var_centred.size,1)
            X_array         = var.reshape(var.size,1)
        else:            
            X = var.reshape(var.size,1)
            X_array = np.hstack([X_array, X])
            X_centred = var_centred.reshape(var_centred.size,1)
            X_array_centred = np.hstack([X_array_centred, X_centred])
            
        i = i + 1   
#        del Tint, Tint_train, Sint, Sint_train, var, X, varTrain, X_train

#    print("X_array.shape = ", X_array.shape)
#    print("Tint_array.shape = ", Tint_array.shape)
#    print("Sint_array.shape = ", Sint_array.shape)
    
#    print(lon.shape,lat.shape,dynHeight.shape,varTime.shape)
    
    return lon, lat, dynHeight, X_array, X_array_centred, varTime
        
    
print('Printing runtime = ', time.clock() - start_time,' s')
