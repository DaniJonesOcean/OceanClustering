#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:57:50 2017

@author: harryholt

Plot.py

Purpose:
    - Almost stand alone module which plots the results to the rest of the program
    - Loads the data form the stored files
    

"""
import numpy as np
import time
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import Print

start_time = time.clock()

def plotMapCircular(address, address_fronts, runIndex, n_comp, plotFronts=True):
    print("Plot.plotMapCircular")
    # Load lat, lon and labels
    lon0, lat0, dynHeight, varTime, labels = None, None, None, None, None
    lon0, lat0, dynHeight, varTime, labels = Print.readLabels(address, runIndex)
    
    # subselect (no more than 10 profiles per 1x1 degree box)
    lon = lon0
    lat = lat0

    # Plot the data in map form - individual
    colorname = 'RdYlBu'
    colormap = plt.get_cmap(colorname,n_comp)
    
#    proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90.0, globe=None)
    proj = ccrs.SouthPolarStereo()
#   proj = ccrs.Mollweide()
    proj_trans = ccrs.PlateCarree()
    
    ax1 = plt.axes(projection=proj)
    CS = ax1.scatter(lon, lat, s = 0.5, lw = 0, c = labels, cmap=colormap, \
                     vmin = -0.5, vmax = n_comp-0.5, transform = proj_trans)
    
    if plotFronts:
        SAF, SACCF, SBDY, PF = None, None, None, None
        SAF, SACCF, SBDY, PF = loadFronts(address_fronts)     # Format is Lon col = 0 and Lat col = 1
        
        ax1.plot(SAF[:,0], SAF[:,1], lw = 1, ls='-', label='SAF', color='black', transform=proj_trans)
        ax1.plot(SACCF[:,0], SACCF[:,1], lw = 1,ls='-', label='SACCF', color='green', transform=proj_trans)
        ax1.plot(SBDY[:,0], SBDY[:,1], lw = 1,ls='-', label='SBDY', color='blue', transform=proj_trans)
        ax1.plot(PF[:,0], PF[:,1], lw = 1,ls='-', label='PF', color='grey', transform=proj_trans)
        
        #ax1.legend(loc='upper left')
        ax1.legend(bbox_to_anchor=( 1.25,1.2), ncol=4, columnspacing = 0.8)

    # Compute a circle in axes coordinates, which we can use as a boundary for the map.
    theta = np.linspace(0, 2*np.pi, 100)
    center = [0.5, 0.5]
    radius = 0.46   # 0.46 corresponds to roughly 30S Latitude
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax1.set_boundary(circle, transform=ax1.transAxes)
 
    # Add features
    ax1.gridlines()
#    ax1.add_feature(cfeature.LAND)
    ax1.coastlines()
    
    colorbar = plt.colorbar(CS)
    colorbar.set_label('Class', rotation=270, labelpad=10)
    plt.savefig(address+"Plots/Labels_Map_n"+str(n_comp)+".pdf",bbox_inches="tight",transparent=True)
    plt.show()
    
def loadFronts(address_fronts):
    SAF, SACCF, SBDY, PF = None, None, None, None
    SAF =   np.loadtxt(address_fronts+'saf.txt')
    SACCF = np.loadtxt(address_fronts+'saccf.txt')
    SBDY =  np.loadtxt(address_fronts+'sbdy.txt')
    PF =    np.loadtxt(address_fronts+'pf.txt')
    
    return SAF, SACCF, SBDY, PF

###############################################################################
###############################################################################
def plotByDynHeight(address, address_fronts, runIndex, n_comp):

    # print function name 
    print("Plot.plotByDynHeight")

    # load lat, lon and labels
    lon, lat, dynHeight, varTime, labels = None, None, None, None, None
    lon, lat, dynHeight, varTime, labels = Print.readLabels(address, runIndex)

    # Load the posterior probabilities for each class
    class_number_array = None
    class_number_array = np.arange(0,n_comp).reshape(-1,1)
    lon_pp, lat_pp, dynHeight_pp, varTime_pp, post_prob = Print.readPosteriorProb(address, runIndex, class_number_array)
 
    # plot the data in map form - individual
    colorname = 'RdYlBu'
    colormap = plt.get_cmap(colorname,n_comp)

    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
    # loop through each class, process and cut
    for numclass in range(0,8):
        xplotNow = lon[labels==numclass]
        yplotNow = dynHeight[labels==numclass]
        postNow = post_prob[:,numclass][labels==numclass]
        xplotNow = xplotNow[postNow>=0.95]
        yplotNow = yplotNow[postNow>=0.95]
        xplotNow = xplotNow[np.logical_not(np.isnan(yplotNow))]
        yplotNow = yplotNow[np.logical_not(np.isnan(yplotNow))]
        axes[int(np.floor(numclass/4)), numclass%4].scatter(xplotNow[::20], yplotNow[::20], 
          s = 1.0, vmin = -0.5, vmax = n_comp-0.5, lw = 0)
        axes[int(np.floor(numclass/4)), numclass%4].set_ylim((0.02, 0.18)) 
        axes[int(np.floor(numclass/4)), numclass%4].set_xlim((-180, 180)) 

    plt.setp(axes, xticks=[-90, 0, 90])
    
    plt.savefig(address+"Plots/classes_dynHeight_heatmap_95pct.pdf",bbox_inches="tight",transparent=True) 

    # next, plot all classes on single plot
    plt.figure(figsize=(5,5))
    xplotNow = lon
    yplotNow = dynHeight
    xplotNow = xplotNow[np.logical_not(np.isnan(dynHeight))]
    yplotNow = yplotNow[np.logical_not(np.isnan(dynHeight))]
    labelsNow = labels[np.logical_not(np.isnan(dynHeight))]
    CS = plt.scatter(xplotNow[::5], yplotNow[::5], s = 1.0, c = labelsNow[::5], cmap = colormap, 
                     vmin = -0.5, vmax = n_comp-0.5, lw = 0)
    plt.xlim((-180, 180)) 
    plt.ylim((0.02, 0.18)) 
    plt.xlabel('Longitude')
    plt.ylabel('Dynamic height (m)')
    plt.grid(color = '0.9')
    colorbar = plt.colorbar(CS)
    colorbar.set_label('Class', rotation=270, labelpad=10)
    plt.savefig(address+"Plots/classes_dynHeight_single.pdf",bbox_inches="tight",transparent=True) 

###############################################################################
###############################################################################
def plotPosterior(address, address_fronts, runIndex, n_comp, plotFronts=True):
    print("Plot.plotPosterior")
    # Load lat, lon and labels
    lon, lat, dynHeight, varTime, labels = None, None, None, None, None
    lon, lat, dynHeight, varTime, labels = Print.readLabels(address, runIndex)

    # Load the posterior probabilities for each class
    class_number_array = None
    class_number_array = np.arange(0,n_comp).reshape(-1,1)
    lon_pp, lat_pp, dynHeight_pp, varTime_pp, post_prob = Print.readPosteriorProb(address, runIndex, class_number_array)

    for k in class_number_array:
        lon_k, lat_k, dynHeight_k, post_k, indices_k = None, None, None, None, None
        indices_k = (np.where(labels == (1.0*k)))
        lon_k, lat_k = lon[indices_k], lat[indices_k]
        post_k = post_prob[:,k][indices_k]  
        # Idea is to take one class, n, and select all indices_k that are actaully assigned to that class.
        likelihood = np.zeros(len(post_k))
        for i in range(len(post_k)):
            if post_k[i] >= 0.99:
                likelihood[i] = 0.99
            elif post_k[i] >= 0.9 and post_k[i] < 0.99 :
                likelihood[i] = 0.9
            elif post_k[i] >= 0.66 and post_k[i] < 0.9:
                likelihood[i] = 0.66
            elif post_k[i] >= 1/(n_comp) and post_k[i] < 0.66:
                likelihood[i] = 1/(n_comp)
            else:
                print("WARNING : Posterior Value less than 1/k")
        # Plot the posterior probabilites
        #ax = plt.subplot(111, polar=True)
        ax = plt.axes(projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90.0, globe=None))
        colorname = 'RdYlBu'
        colormap = plt.get_cmap(colorname, 4)
        
        theta = np.pi*(lon_k)/180.0
        rho = 90 - abs(lat_k)
   
        CS = ax.scatter(theta[::1000] ,rho[::1000], 0.5, lw = 0, c = likelihood[::50], cmap=colormap, vmin = 0, vmax = 1)

        # plot fronts 
        if plotFronts:
            SAF, SACCF, SBDY, PF = None, None, None, None
            SAF, SACCF, SBDY, PF = loadFronts(address_fronts)     # Format is Lon col = 0 and Lat col = 1
            
            theta_saf = np.pi*(SAF[:,0])/180.0
            rho_saf = 90 - abs(SAF[:,1])
            ax.plot(theta_saf, rho_saf, lw = 1, ls='-', label='SAF', color='black')
            
            theta_saccf = np.pi*(SACCF[:,0])/180.0
            rho_saccf = 90 - abs(SACCF[:,1])
            ax.plot(theta_saccf, rho_saccf, lw = 1,ls='-', label='SACCF', color='green')
            
            theta_sbdy = np.pi*(SBDY[:,0])/180.0
            rho_sbdy = 90 - abs(SBDY[:,1])
            ax.plot(theta_sbdy, rho_sbdy, lw = 1,ls='-', label='SBDY', color='blue')
            
            theta_pf = np.pi*(PF[:,0])/180.0
            rho_pf = 90 - abs(PF[:,1])
            ax.plot(theta_pf, rho_pf, lw = 1,ls='-', label='PF', color='grey')
            
            #ax1.legend(loc='upper left')
            #ax.legend(bbox_to_anchor=( 1.25,1.25), ncol=4, columnspacing = 0.8)
        
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0,60)
        plt.text(0, 1, "Class "+str(k[0]), transform = ax.transAxes)
        ax.set_yticklabels([])
        #colorbar = plt.colorbar(CS)
        #colorbar.set_label('Probability of belonging to Class', rotation=270, labelpad=10)
        plt.savefig(address+"Plots/v2_PostProb_Class"+str(k[0])+"_n"+str(n_comp)+".pdf",bbox_inches="tight",transparent=True)
        plt.show()

###############################################################################
###############################################################################
def plotProfileClass(address, runIndex, n_comp, space):
    # space will be 'depth', 'reduced' or 'uncentred'
    print("Plot.plotProfileClass "+str(space))
    # Load depth
    depth = None
    depth = Print.readDepth(address, runIndex)
    
    # Load reduced depth
    col_reduced = None
    col_reduced = Print.readColreduced(address, runIndex)
    col_reduced_array = np.arange(col_reduced)
    
    #
    depth_array = None
    depth_array = depth
    if space == 'reduced':
        depth_array = col_reduced_array
    
    # Load class properties
    gmm_weights, gmm_means, gmm_covariances = None, None, None
    gmm_weights, gmm_means, gmm_covariances = Print.readGMMclasses(address,\
                                                        runIndex, depth_array, space)
    
    fig, ax1 = plt.subplots()
    for d in range(n_comp):
        ax1.plot(gmm_means[d,:], depth_array, lw = 1, label = "Class "+str(d))
        
    if space == 'depth':
        ax1.set_xlabel("Normalized Temperature Anomaly /degree")
        ax1.set_ylabel("Depth")
        ax1.set_xlim(-3,3)
    elif space == 'uncentred':
        ax1.set_xlabel("Temperature /degrees")
        ax1.set_ylabel("Depth")
    elif space == 'reduced':
        ax1.set_xlabel("Normalized Anomaly")
        ax1.set_ylabel("Reduced Depth")
    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.legend(loc='best')
    #ax1.set_title("Class Profiles with Depth in SO - "+space)
    filename = address+"Plots/Class_Profiles_"+space+"_n"+str(n_comp)+".pdf"  
    if runIndex != None:
        filename = address+"Plots/Class_Profiles_"+space+"_runIndex"+str(int(runIndex))+"_n"+str(n_comp)+".pdf"  
    plt.savefig(filename,bbox_inches="tight",transparent=True)
    plt.show()

###############################################################################
###############################################################################
def plotProfile(address, runIndex, space): # Uses traing profiles at the moment
        # space will be 'depth', 'original' or 'uncentred'
    print("Plot.plotProfileClass "+str(space))
    # Load depth
    depth = None
    depth = Print.readDepth(address, runIndex)
    #
    depth_array = None
    depth_array = depth
    X_profiles = None
    if space == 'uncentred' or space == 'depth':
        # Load profiles
        lon_train, lat_train, dynHeight_train, X_train, X_train_centred, varTime_train = None, None, None, None, None, None
        lon_train, lat_train, dynHeight_train, X_train, X_train_centred, varTime_train = \
                        Print.readReconstruction(address, runIndex, depth, True)
        """
        lon_train, lat_train, dynHeight_train, Tint_train_array, X_train_array, \
            Sint_train_array, varTime_train = None, None, None, None, None, None, None
        lon_train, lat_train, dynHeight_train, Tint_train_array, X_train_array, \
            Sint_train_array, varTime_train = Print.readLoadFromFile_Train(address, runIndex, depth)    
        X_train_centred = X_train_array
        """
        if space == 'uncentred':
            X_profiles = X_train
        if space == 'depth':
            X_profiles = X_train_centred
    elif space == 'original':
        lon_train, lat_train, dynHeight_train, Tint_train_array, X_train_array, \
            Sint_train_array, varTime_train = None, None, None, None, None, None, None
        lon_train, lat_train, dynHeight_train, Tint_train_array, X_train_array, \
            Sint_train_array, varTime_train = Print.readLoadFromFile_Train(address, runIndex, depth)
        
        X_profiles = Tint_train_array
    
    fig, ax1 = plt.subplots()
    for d in range(np.ma.size(X_profiles, axis=0)):
        ax1.plot(X_profiles[d,:], depth_array, lw = 1, alpha = 0.01, color = 'grey')
        
    if space == 'depth':
        ax1.set_xlabel("Normalized Temperature Anomaly /degree")
        ax1.set_ylabel("Depth")
    elif space == 'uncentred':
        ax1.set_xlabel("Temperature /degrees")
        ax1.set_ylabel("Depth")
    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.legend(loc='best')
    #ax1.set_title("Profiles with Depth in SO - "+space)
    ax1.set_xlabel("Temperature /degrees")
    ax1.set_ylabel("Depth /dbar")
    filename = address+"Plots/Profiles_"+space+".pdf"  
    if runIndex != None:
        filename = address+"Plots/Profiles_"+space+"_runIndex"+str(int(runIndex))+".pdf"  
    plt.savefig(filename,bbox_inches="tight",transparent=True)
    plt.show()
    
###############################################################################
###############################################################################
def plotGaussiansIndividual(address, runIndex, n_comp, space, Nbins=1000):
    # space will be 'depth', 'reduced' or 'uncentred'
    print("Plot.plotGaussiansIndividual "+str(space))
    if space == 'depth' or space == 'uncentred':
        # Load depth
        depth = None
        depth = Print.readDepth(address, runIndex)
        depth_array = depth
        print("depth.shape = ", depth.shape)
        depth_array_mod = np.array([0,50,100,150,-1])
        print("depth_array_mod.shape = ", depth_array_mod.shape)
        
        # Load X_train array and X_train_centred array
        lon_train, lat_train, dynHeight_train, X_train, X_train_centred, varTime_train = None, None, None, None, None, None
        lon_train, lat_train, dynHeight_train, X_train, X_train_centred, varTime_train = \
                        Print.readReconstruction(address, runIndex, depth, True)
        """
        lon_train, lat_train, dynHeight_train, Tint_train_array, X_train_array, \
            Sint_train_array, varTime_train = None, None, None, None, None, None, None
        lon_train,lat_train, dynHeight_train, Tint_train_array, X_train_array, \
            Sint_train_array, varTime_train = Print.readLoadFromFile_Train(address, runIndex, depth)    
        X_train_centred = X_train_array
        """
        print("VALUE = ", X_train_centred[10,0])
        
    if space == 'reduced':
        # Load reduced depth
        col_reduced = None
        col_reduced = Print.readColreduced(address, runIndex)
        depth_array = np.arange(col_reduced)
        depth_array_mod = depth_array
        
        lon_train, lat_train, dynHeight_train, X_train_centred, varTime_train = None, None, None, None, None
        lon_train, lat_train, dynHeight_train, X_train_centred, varTime_train = \
                        Print.readPCAFromFile_Train(address, runIndex, col_reduced)
        print("VALUE = ", X_train_centred[10,0])
    
    # Load class properties
    gmm_weights, gmm_means, gmm_covariances = None, None, None
    gmm_weights, gmm_means, gmm_covariances = Print.readGMMclasses(address,\
                                                        runIndex, depth_array, space)
    if space == 'uncentred':
        stand = None
        with open(address+"Objects/Scale_object.pkl", 'rb') as input:
            stand = pickle.load(input)
        gmm_means = stand.inverse_transform(gmm_means)
        gmm_covariances = stand.inverse_transform(gmm_covariances)
    
    print("Shapes: ", gmm_weights.shape, gmm_means.shape, gmm_covariances.shape)
    print("depth_array_mod.shape = ", depth_array_mod.shape)
    
    # Define the gaussian function
    def gaussianFunc(x, mu, cov):
        return (np.exp(-np.power(x - mu, 2.) / (2 * cov)))/(np.sqrt(cov*np.pi*2))
    
    for i in range(len(depth_array_mod)):
        print("About to plot")
        X_row = None
        X_row = X_train_centred[:,int(depth_array_mod[i])]
        if space == 'uncentred':
            X_row = None
            X_row = X_train[:,int(depth_array_mod[i])]
        means_row, cov_row = None, None
        means_row = gmm_means[:,int(depth_array_mod[i])]
        cov_row = abs(gmm_covariances[:,int(depth_array_mod[i])])
        print("Covariance = ", cov_row)
        
        xmax, xmin = None, None
        xmax = np.max(X_row)*1.1
        xmin = np.min(X_row)*1.1
        
        print("Xmin = ", xmin, "Xmax = ", xmax)
    
        fig, ax1 = plt.subplots()
        x_values = None
        x_values = np.linspace(xmin, xmax, 120)
        print(x_values.shape, min(x_values), max(x_values))
        
        y_total = np.zeros(n_comp*120).reshape(n_comp,120)
        
        for n in range(n_comp):
            y_gaussian = None
#            y_gaussian = gmm_weights[n]*gaussianFunc(x_values, gmm_means[n,int(depth_array_mod[i])] , gmm_covariances[n,int(depth_array_mod[i])]) # Use if diag
            y_gaussian = gmm_weights[n]*gaussianFunc(x_values, means_row[n] , cov_row[n]) # Use if diag
            y_total[n,:] = y_gaussian
            ax1.plot(x_values, y_gaussian, label=str(n))
        
        
        ax1.plot(x_values, np.sum(y_total,axis=0), lw = 2, color = 'black', label="Overall")     # Use if diag
        ax1.hist(X_row, bins=Nbins, normed=True, facecolor='grey', lw = 0)
        ax1.set_ylabel("Probability density")
        ax1.set_xlabel("Normalized Temperature Anomaly")
        if space == 'reduced':
            ax1.set_xlabel("Normalized Anomaly")
        if space == 'uncentred':
            ax1.set_xlabel("Temperature /degrees")
        ax1.set_title("GMM n = "+str(n_comp)+", "+space+" = "+str(int(depth_array[depth_array_mod[i]])))
        ax1.grid(True)
        ax1.set_xlim(xmin,xmax)
        ax1.legend(loc='best')
        plt.savefig(address+"Plots/TrainHisto_Gaussians_n"+str(n_comp)+"_"+space+str(int((depth_array[depth_array_mod[i]])))+".pdf",bbox_inches="tight",transparent=True)
        plt.show()
    
###############################################################################    
def plotBIC(address, repeat_bic, max_groups, trend=True): 
    # Load the data and define variables first
    bic_many, bic_mean, bic_stdev, n_mean, n_stdev, n_min = None, None, None, None, None, None
    bic_many, bic_mean, bic_stdev, n_mean, n_stdev, n_min = Print.readBIC(address, repeat_bic)
    n_comp_array = None
    n_comp_array = np.arange(1, max_groups)
    
    print("Calculating n and then averaging across runIndexs, n = ", n_mean, "+-", n_stdev)
    print("Averaging BIC scores and then calculating, n = ", n_min)
    
    # Plot the results
    fig, ax1 = plt.subplots()
    ax1.errorbar(n_comp_array, bic_mean, yerr = bic_stdev, lw = 2, ecolor = 'black', label = 'Mean BIC Score')
    
    if trend:
        # Plot the trendline
        #initial_guess = [20000, 1, 20000, 0.001]
        initial_guess = [47030, 1.553, 23080, 0.0004652]
        def expfunc(x, a, b, c, d):
            return (a * np.exp(-b * x)) + (c * np.exp(d * x))
        
        popt, pcov, x, y = None, None, None, None
        popt, pcov = curve_fit(expfunc, n_comp_array, bic_mean, p0 = initial_guess, maxfev=10000)
        print("Exponential Parameters = ", *popt)
        x = np.linspace(1, max_groups, 100)
        y = expfunc(x, *popt)
        ax1.plot(x, y, 'r-', label="Exponential Fit")
        
        y_min_index = np.where(y==y.min())[0]
        x_min = (x[y_min_index])[0]
        ax1.axvline(x=x_min, linestyle=':', color='black', label = 'Exponential Fit min = '+str(np.round_(x_min, decimals=1)))


    # Plot the individual and minimum values
    ax1.axvline(x=n_mean, linestyle='--', color='black', label = 'n_mean_min = '+str(n_mean))
    ax1.axvline(x=n_min, linestyle='-.', color='black', label = 'n_bic_min = '+str(n_min))
    for r in range(repeat_bic):
        ax1.plot(n_comp_array, bic_many[r,:], alpha = 0.3, color = 'grey')
        
    ax1.set_ylabel("BIC value")
    ax1.set_xlabel("Number of classes in GMM")
    ax1.grid(True)
    ax1.set_title("BIC values for GMM with different number of components")
    ax1.set_ylim(min(bic_mean)*0.97, min(bic_mean)*1.07)
    ax1.legend(loc='best')
    if trend:
        plt.savefig(address+"Plots/BIC_trend.pdf",bbox_inches="tight",transparent=True)
    else:
        plt.savefig(address+"Plots/BIC.pdf",bbox_inches="tight",transparent=True)
    plt.show()
    
###############################################################################
# Use the VBGMM to determine how many classes we should use in the GMM
def plotWeights(address, runIndex):
    # Load depth
    depth = None
    depth = Print.readDepth(address, runIndex)

    # Load Weights    
    gmm_weights, gmm_means, gmm_covariances = None, None, None
    gmm_weights, gmm_means, gmm_covariances = Print.readGMMclasses(address, runIndex, depth, 'depth')
    
    n_comp = len(gmm_weights)
    class_array = np.arange(0,n_comp,1)
    
    # Plot weights against class number
    fig, ax1 = plt.subplots()
    ax1.scatter(class_array, np.sort(gmm_weights)[::-1], s = 20, marker = '+', color = 'blue', label = 'Class Weights')
    ax1.axhline(y=1/(n_comp+1), linestyle='-.', color='black', label = str(np.round_(1/(n_comp+1), decimals=3))+' threshold')
    ax1.axhline(y=1/(n_comp+5), linestyle='--', color='black', label = str(np.round_(1/(n_comp+5), decimals=3))+' threshold')
    ax1.set_xlabel("Class")
    ax1.set_xlim(-1,n_comp)
    ax1.set_ylabel("Weight")
    ax1.grid(True)
    ax1.set_title("VBGMM Class Weights")
    ax1.legend(loc='best')
    plt.savefig(address+"Plots/Weights_VBGMM.pdf", bbox_inches="tight",transparent=True)
    
    
print('Plot runtime = ', time.clock() - start_time,' s')
