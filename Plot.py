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
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.optimize import curve_fit
import pickle
import pdb

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import Print
import time

start_time = time.clock()

def plotMapCircular(address, address_fronts, plotFronts, n_comp, allDF, colormap):
    print("Plot.plotMapCircular")
    runIndex = None

    # set plotting thresholds
    threshold = 0.90
    pskip = 10

    # select colormap (qualitative)
    # now passed in as an external argument
#   colorname = 'RdYlBu_r'
#   colormap = plt.get_cmap(colorname,n_comp)

    # select x, y, and color data    
    surfaceDF = allDF[allDF.pressure == 15]
    surfaceDF = surfaceDF[surfaceDF.posterior_probability >= threshold]
    xplot = surfaceDF['longitude'].values
    yplot = surfaceDF['latitude'].values
    cplot = surfaceDF['class_sorted'].values

    # subselect
    xplot = xplot[::pskip]
    yplot = yplot[::pskip]
    cplot = cplot[::pskip]

    # shift indices for plotting
    cplot = cplot + 1

    # select map and projection
    proj = ccrs.SouthPolarStereo()
    proj_trans = ccrs.PlateCarree()
    
    # create plot axes
    ax1 = plt.axes(projection=proj)

    # create scatter plot 
    CS = ax1.scatter(xplot, yplot, s = 2.0, lw = 0, c = cplot, \
                     cmap=colormap, vmin = 0.5, vmax = n_comp + 0.5, transform = proj_trans)
    
    # add fronts
    if plotFronts:
        SAF, SACCF, SBDY, PF = None, None, None, None
        SAF, SACCF, SBDY, PF = loadFronts(address_fronts)   
        
        ax1.plot(SAF[:,0], SAF[:,1], lw = 1, ls='-', label='SAF', \
                 color='black', transform=proj_trans)
        ax1.plot(PF[:,0], PF[:,1], lw = 1,ls='-', label='PF', \
                 color='grey', transform=proj_trans)
        ax1.plot(SACCF[:,0], SACCF[:,1], lw = 1,ls='-', label='SACCF', \
                 color='green', transform=proj_trans)
        ax1.plot(SBDY[:,0], SBDY[:,1], lw = 1,ls='-', label='SBDY', \
                 color='blue', transform=proj_trans)
        
        #ax1.legend(loc='upper left')
        ax1.legend(bbox_to_anchor=( 1.25,1.2), ncol=4, \
                   columnspacing = 0.8)

    # compute a circle in axes coordinates, 
    # which we can use as a boundary for the map.
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
    cblabels = np.arange(1, int(n_comp)+1, 1)
#   cblabels = np.arange(0, int(n_comp), 1)
    cbloc = cblabels
    colorbar.set_ticks(cbloc)
    colorbar.set_ticklabels(cblabels)
    colorbar.set_label('Class', rotation=270, labelpad=10)
    plt.savefig(address+"Plots/Labels_Map_n"+str(n_comp)+\
                ".pdf",bbox_inches="tight",transparent=True)
#   plt.show()
    
#######################################################################
    
def loadFronts(address_fronts):
    SAF, SACCF, SBDY, PF = None, None, None, None
    SAF =   np.loadtxt(address_fronts+'saf_kim.txt')
    SACCF = np.loadtxt(address_fronts+'saccf_kim.txt')
    SBDY =  np.loadtxt(address_fronts+'sbdy_kim.txt')
    PF =    np.loadtxt(address_fronts+'pf_kim.txt')
    
    return SAF, SACCF, SBDY, PF

#######################################################################

def plotByDynHeight(address, address_fronts, runIndex, n_comp, allDF, colormap):

    # print function name 
    print("Plot.plotByDynHeight")

    # plotting parameters
    threshold = 0.9
    pskip = 10

    # plot the data in map form - individual
    # now passed in as an argument
#   colorname = 'RdYlBu_r'
#   colormap = plt.get_cmap(colorname,n_comp)

    # select points for plotting 
    surfaceDF = allDF[allDF['depth_index']==0].dropna()
    surfaceDF = surfaceDF[surfaceDF.posterior_probability >= threshold]
    xplot = surfaceDF['longitude'].values
    yplot = surfaceDF['dynamic_height'].values
    cplot = surfaceDF['class_sorted'].values

    # skip selected points
    xplot = xplot[::pskip]
    yplot = yplot[::pskip]
    cplot = cplot[::pskip]

    # shift index for plotting
    cplot = cplot + 1

    # next, plot all classes on single plot
    plt.figure(figsize=(5,5))

    # scatter plot
    CS = plt.scatter(xplot, yplot, s = 2.0, c = cplot, cmap = colormap, \
                     vmin = 0.5, vmax = n_comp+0.5, lw = 0)
    plt.xlim((-180, 180)) 
    plt.ylim((0.02, 0.18)) 
    plt.xlabel('Longitude')
    plt.ylabel('Dynamic height (m)')
    plt.grid(color = '0.9')

    # fix colorbar
    colorbar = plt.colorbar(CS)
    cblabels = np.arange(1, int(n_comp)+1, 1)
#   cblabels = np.arange(0, int(n_comp), 1)
    cbloc = cblabels
    colorbar.set_ticks(cbloc)
    colorbar.set_ticklabels(cblabels)
    colorbar.set_label('Class', rotation=270, labelpad=10)

    # save figure
    plt.savefig(address+"Plots/classes_dynHeight_single.pdf",bbox_inches="tight",transparent=True) 
    #lt.show()

###############################################################################

def plotPosterior(address, address_fronts, runIndex, n_comp, plotFronts, allDF):

    print("Plot.plotPosterior")

    # threshold
    nsub = 5

    # get colormap
#   colorname = 'RdYlBu'
#   colormap = plt.get_cmap(colorname, 4)

    # cutom colors from colorbrewer2.org
    col4 = [254/256, 240/256, 217/256]
    col3 = [253/256, 204/256, 138/256]
    col2 = [252/256, 141/256, 89/256]
    col1 = [215/256, 48/256, 31/256]
    cmap = mpl.colors.ListedColormap([col1, col2, col4])
#   cmap.set_under(col1)
#   cmap.set_over(col4)
    bounds = [0.00, 0.75, 0.90, 1.00]
#   norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # just select the surface
    surfaceDF = allDF[allDF['depth_index']==0]

    # loop over index, make some plots
    for k in range(0,n_comp):
        print(k)
        lon_k = surfaceDF[surfaceDF.class_sorted == k]['longitude']
        lat_k = surfaceDF[surfaceDF.class_sorted == k]['latitude']
        prob_k = surfaceDF[surfaceDF.class_sorted ==k]['posterior_probability']
        likelihood = prob_k

        # calculate numbers of pirofiles in each
        ncount = np.zeros(4,)
        ntotal = 0
        pp = likelihood.values
        for nprof in range(0,pp.size):
            ntotal = ntotal + 1
            if pp[nprof]<=0.50:
                ncount[0] = ncount[0] + 1
            elif pp[nprof]<=0.75:
                ncount[1] = ncount[1] + 1
            elif pp[nprof]<=0.90:
                ncount[2] = ncount[2] + 1
            elif pp[nprof]<=1.00:
                ncount[3] = ncount[3] + 1

        ncount_pct = 100*ncount/ntotal
  
        # print
        print('number in each pp range (0.0 <= 0.50 <= 0.75 <= 0.90 <= 1.0)')
        print(ncount)
        print('scaled by total in this class')
        print(ncount_pct)
     
        # subselect
        xplot = lon_k[::nsub]
        yplot = lat_k[::nsub]
        cplot = likelihood[::nsub]

        # projection
        proj = ccrs.SouthPolarStereo()
        proj_trans = ccrs.PlateCarree()
        ax1 = plt.axes(projection=proj)
        ax1.set_extent((-180,180,-90,-30),crs=proj_trans)
        CS = ax1.scatter(xplot , yplot, s = 3.0, lw = 0, c = cplot, \
                         cmap = cmap, transform = proj_trans)
        cb3 = mpl.colorbar.ColorbarBase(plt.gca(), \
                                        cmap=cmap, \
                                        norm=norm, \
                                        ticks=bounds, \
                                        spacing='uniform', \
                                        orientation='vertical')
        cb3.ax.tick_params(labelsize=22)
        print(cb3.ax.viewLim)
        # plot fronts 
        if plotFronts:
            SAF, SACCF, SBDY, PF = None, None, None, None
            SAF, SACCF, SBDY, PF = loadFronts(address_fronts)  
            ax1.plot(PF[:,0], PF[:,1], lw = 1,ls='-', label='PF', \
                     color='grey', transform=proj_trans) 

        # compute a circle in axes coordinates, which we can use as a boundary for the map.
        theta = np.linspace(0, 2*np.pi, 100)
        center = [0.5, 0.5]
        radius = 0.52   # 0.46 corresponds to roughly 30S Latitude
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax1.set_boundary(circle, transform=ax1.transAxes)

        # add features
        ax1.gridlines()
        ax1.coastlines()
        ax1.set_extent((-180,180,-90,-30),crs=proj_trans)

#       plt.text(0, 1, "Class "+str(k+1), transform = ax1.transAxes)

        # show plot
        plt.savefig(address+"Plots/v3_PostProb_Class"+str(k)+\
                    "_n"+str(n_comp)+".pdf",bbox_inches="tight",transparent=True)
#       plt.show()
        ax1.clear()

###############################################################################

def plotProfilesByClass(address, runIndex, n_comp, allDF, colormap):

    # print
    print('Plot.plotProfilesByClass')

    # set dimensions of plot
    w = 6
    h = 12 

    # plot the data in map form - individual
    # now passed in as an external argument
#   colorname = 'RdYlBu_r'
#   colormap = plt.get_cmap(colorname,n_comp)

    # get colors for plots
    cNorm = colors.Normalize(vmin=0, vmax=n_comp)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)

    # loop through all classes, get mean/std temperature of profiles
    for k in range(0,n_comp):

        # create figure
        fig = plt.figure(figsize=(7,7))

        # select color for plot
        colorVal = scalarMap.to_rgba(k)
 
        # select all profiles from class k
        class_k_DF = allDF[ allDF['class_sorted'] == k ]

        # calculate statistics of those profiles at each pressure level
        Tmean = class_k_DF.groupby(['depth_index'])['temperature'].mean().values
        Tmedian = class_k_DF.groupby(['depth_index'])['temperature'].median().values
        Tsig = class_k_DF.groupby(['depth_index'])['temperature'].std().values
        P = class_k_DF.groupby(['depth_index'])['pressure'].mean().values 

        # create plot
        plt.plot(Tmean, P, color=colorVal, linestyle='solid', linewidth=5.0)
        plt.plot(Tmean-Tsig, P, color=colorVal, linestyle='dashed', linewidth=5.0)
        plt.plot(Tmean+Tsig, P, color=colorVal, linestyle='dashed', linewidth=5.0)

        # invert axes
        ax = plt.gca()
        ax.set_ylim(0,1000)
        ax.set_xlim(-2,25)
        ax.grid(alpha=0.2)
        ax.invert_yaxis()
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Pressure (dbar)')

        # save and show the plot 
        plt.savefig(address + 'Plots/Tprof_by_pressure_class' + \
                    str(k).zfill(3) + '.pdf', bbox_inches="tight", \
                    transparent=True)
#       plt.show()

###############################################################################

def plotProfileClass(address, runIndex, n_comp, space, allDF, colormap):

    # space will be 'depth', 'reduced' or 'uncentred'
    print("Plot.plotProfileClass "+str(space))

    # plot the data in map form - individual
    # now passed in as an external argument
#   colorname = 'RdYlBu_r'
#   colormap = plt.get_cmap(colorname,n_comp)
    
    # Load reduced depth
    col_reduced = None
    col_reduced = Print.readColreduced(address, runIndex)
    col_reduced_array = np.arange(col_reduced)
    
    #
    depth_array = None
    depth_array = depth
    if space == 'reduced':
        depth_array = col_reduced_array
    
    # load class properties
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
    plt.savefig(filename,bbox_inches="tight",transparent=True)
#   plt.show()

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
    plt.savefig(filename,bbox_inches="tight",transparent=True)
#   plt.show()
    
###############################################################################

def plotPCAcomponents(address, runIndex, n_comp):

    # space will be 'depth', 'reduced' or 'uncentred'
    print("Plot.plotPCAcomponents")

    # get depths
    pressures = Print.readDepth(address, runIndex) 

    # read PCA object
    with open('../Objects/PCA_object.pkl','rb') as input:
        pca_object = pickle.load(input)

    # get number of components
    n_pca_comp = pca_object.n_components_
    pca_comp = pca_object.components_

    # define colormap
    colorname = 'Dark2'
    colormap = plt.get_cmap(colorname, n_pca_comp)
    cNorm = colors.Normalize(vmin = 0, vmax = n_pca_comp)
    scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = colormap) 
    colorVal = scalarMap.to_rgba

    # line plot
    for k in range(0, n_pca_comp):
        fig = plt.figure(figsize=(7,7))
        x = np.transpose(pca_comp[k])
        y = pressures
        plt.plot(x, y, color=colorVal(k), linestyle='solid', linewidth=5.0)
        plt.show()

###############################################################################

def plotGaussiansIndividual(address, runIndex, n_comp, space, allDF, Nbins, colormap):

    # space will be 'depth', 'reduced' or 'uncentred'
    print("Plot.plotGaussiansIndividual "+str(space))

    # read old2new for mapping between unsorted and sorted classes
    pkl_file = open(address + 'Results/old2new.pkl', 'rb')
    old2new = pickle.load(pkl_file)
    pkl_file.close()
    sortEm = np.asarray(list(old2new.values()))

    # read PCA object
    with open('../Objects/PCA_object.pkl','rb') as input:
        pca_object = pickle.load(input)

    # fraction of variance explained
    pcavar = pca_object.explained_variance_ratio_ 
 
    # get colors for plots
    cNorm = colors.Normalize(vmin=0, vmax=n_comp)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)
 
    # if in depth space or uncentred
    if space == 'depth' or space == 'uncentred':

        # load depth
        depth = None
        depth = Print.readDepth(address, runIndex)
        depth_array = depth
        print("depth.shape = ", depth.shape)
        depth_array_mod = np.array([0,50,100,150,-1])
        print("depth_array_mod.shape = ", depth_array_mod.shape)
        
        # load X_train array and X_train_centred array
        lon_train, lat_train, dynHeight_train, X_train, \
          X_train_centred, varTime_train = None, None, None, None, None, None
        lon_train, lat_train, dynHeight_train, X_train, X_train_centred, varTime_train = \
                        Print.readReconstruction(address, runIndex, depth, True)
        print("VALUE = ", X_train_centred[10,0])
        
    if space == 'reduced':

        # load reduced depth
        col_reduced = None
        col_reduced = Print.readColreduced(address, runIndex)
        depth_array = np.arange(col_reduced)
        depth_array_mod = depth_array
        
        lon_train, lat_train, dynHeight_train, X_train_centred, \
          varTime_train = None, None, None, None, None
        lon_train, lat_train, dynHeight_train, X_train_centred, varTime_train = \
                        Print.readPCAFromFile_Train(address, runIndex, col_reduced)
        print("VALUE = ", X_train_centred[10,0])
    
    # load class properties
    gmm_weights, gmm_means, gmm_covariances = None, None, None
    gmm_weights, gmm_means, gmm_covariances = Print.readGMMclasses(address,\
                                                        runIndex, depth_array, space)

#   if space == 'uncentred':
#       stand = None
#       with open(address+"Objects/Scale_object.pkl", 'rb') as input:
#           stand = pickle.load(input)
#       gmm_means = stand.inverse_transform(gmm_means)
#       gmm_covariances = stand.inverse_transform(gmm_covariances)
    
    print("shapes: ", gmm_weights.shape, gmm_means.shape, gmm_covariances.shape)
    print("depth_array_mod.shape = ", depth_array_mod.shape)
    
    # define the gaussian function
    def gaussianFunc(x, mu, cov):
        return (np.exp(-np.power(x - mu, 2.) / (2 * cov)))/(np.sqrt(cov*np.pi*2))
    
    for i in range(len(depth_array_mod)):
#       print("About to plot")
        X_row = None
        X_row = X_train_centred[:,int(depth_array_mod[i])]
        if space == 'uncentred':
            X_row = None
            X_row = X_train[:,int(depth_array_mod[i])]
        means_row, cov_row = None, None
        means_row = gmm_means[:,int(depth_array_mod[i])]
        cov_row = abs(gmm_covariances[:,int(depth_array_mod[i])])
#       print("Covariance = ", cov_row)
        
        xmax, xmin = None, None
        xmax = np.max(X_row)*1.1
        xmin = np.min(X_row)*1.1
        
        print("Xmin = ", xmin, "Xmax = ", xmax)
    
        fig, ax1 = plt.subplots()
        x_values = None
        x_values = np.linspace(xmin, xmax, 120)
#       print(x_values.shape, min(x_values), max(x_values))
        y_total = np.zeros(n_comp*120).reshape(n_comp,120)

        # switch order of classes using old2new
        gmm_weights_s = gmm_weights[sortEm]
        means_row_s = means_row[sortEm]
        cov_row_s = cov_row[sortEm]

        # loop through classes       
        for n in range(n_comp):
            colorVal = scalarMap.to_rgba(n)
            y_gaussian = None
            # use if diag
            y_gaussian = gmm_weights_s[n]*gaussianFunc(x_values, \
                         means_row_s[n] , cov_row_s[n]) 
            y_total[n,:] = y_gaussian
            ax1.plot(x_values, y_gaussian, label=str(n+1), color=colorVal, lw=2)
        
        ax1.plot(x_values, np.sum(y_total,axis=0), lw = 2, color = 'black', \
                 label="Overall", linestyle='dashed')
        ax1.hist(X_row, bins=Nbins, normed=True, facecolor='grey', lw = 0)
        ax1.set_ylabel("Probability density")
        ax1.set_xlabel("Normalized temperature anomaly")
        if space == 'reduced':
            ax1.set_xlabel("Normalized anomaly")
        if space == 'uncentred':
            ax1.set_xlabel("Temperature /degrees")
        ts0 = "PC" + str( int(depth_array[depth_array_mod[i]]) + 1 )
        ts1 = "{0:.2f}".format(round(100*pcavar[i],2))
        ts2 = '(' + ts1 + '% variance explained)' 
        ax1.set_title( ts0 + ' ' + ts2 )
        ax1.grid(True)
        ax1.set_xlim(xmin,xmax)

        # only add legend to first depth
        if i==0:
            ax1.legend(loc='best')

        plt.savefig(address+\
                    "Plots/TrainHisto_Gaussians_n"+\
                    str(n_comp)+"_"+\
                    space+str(int((depth_array[depth_array_mod[i]])))+\
                    ".pdf",bbox_inches="tight",transparent=True)
#       plt.show()
   
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
    
        # Commented out exponential fit (it looks terrible)    
        #popt, pcov, x, y = None, None, None, None
        #popt, pcov = curve_fit(expfunc, n_comp_array, bic_mean, p0 = initial_guess, maxfev=10000)
        #print("Exponential Parameters = ", *popt)
        #x = np.linspace(1, max_groups, 100)
        #y = expfunc(x, *popt)
        #ax1.plot(x, y, 'r-', label="Exponential Fit")
        
        #y_min_index = np.where(y==y.min())[0]
        #x_min = (x[y_min_index])[0]
        #ax1.axvline(x=x_min, linestyle=':', color='black', label = 'Exponential Fit min = '+str(np.round_(x_min, decimals=1)))

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
#   plt.show()
    
###############################################################################
# Use the VBGMM to determine how many classes we should use in the GMM

def plotWeights(address, runIndex):

    # load depth
    depth = None
    depth = Print.readDepth(address, runIndex)

    # load Weights    
    gmm_weights, gmm_means, gmm_covariances = None, None, None
    gmm_weights, gmm_means, gmm_covariances = Print.readGMMclasses(address, runIndex, depth, 'depth')
    
    n_comp = len(gmm_weights)
    class_array = np.arange(0,n_comp,1)
    
    # plot weights against class number
    fig, ax1 = plt.subplots()
    ax1.scatter(class_array+1, np.sort(gmm_weights)[::-1], s = 30, marker = '+', color = 'blue', label = 'class weights')
    ax1.axhline(y=1/(n_comp+1), linestyle='-.', color='black', label = str(np.round_(1/(n_comp+1), decimals=3))+' threshold')
    ax1.axhline(y=1/(n_comp+5), linestyle='--', color='black', label = str(np.round_(1/(n_comp+5), decimals=3))+' threshold')
    ax1.set_xlabel("Class")
    ax1.set_xlim(0,n_comp)
    ax1.set_ylabel("Weight")
    ax1.grid(True)
    ax1.set_title("VBGMM class weights")
    ax1.legend(loc='best')
    plt.savefig(address+"Plots/Weights_VBGMM.pdf", bbox_inches="tight",transparent=True)
    
###############################################################################
# New function to plot the eigenvectors

def plotEigenvectors(address, runIndex, allDF):
    
    # load reduced depth
    depth = None
    depth = Print.readDepth(address, runIndex)
   
    # define the mean profile
    Tmean = allDF.groupby(['depth_index'])['temperature'].mean().values

    # load the PCA object
    pca = None
    with open(address+'Objects/PCA_object.pkl', 'rb') as input:
        pca = pickle.load(input)
    pca_comp = pca.components_
    
    # load the Standardise and Centre object
    stand = None
    with open(address+"Objects/Scale_object.pkl", 'rb') as input:
        stand = pickle.load(input)
    pca_comp_unstand = stand.inverse_transform(pca_comp)

    # plus/minus relative to mean profile
#   Tmean_plus_PC = pca_comp_unstand
#   Tmean_minus_PC = pca_comp_unstand
#   for m in range(np.ma.size(pca_comp, axis=0)):
#       Tmean_plus_PC[m] = Tmean + pca_comp_unstand[m]
#       Tmean_minus_PC[m] = Tmean - pca_comp_unstand[m]    

    #depths = (5 * (np.arange(np.ma.size(pca_components, axis=1)) )) + 15
    for i in range(np.ma.size(pca_comp, axis=0)):

        fig = plt.figure(figsize=(7,7))

        # create plot
        plt.plot(pca_comp[i,:], depth, linewidth=2.0, color='black')

        # axis
        ax = plt.gca()
        ax.set_ylabel("Pressure (dbar)")
#       ax.set_xlabel("Centred Eigenvector Coefficient")
        ax.invert_yaxis()
        ax.grid(True)
#       ax.legend(loc='best')

        # save figure
        plt.savefig(address+"Plots/PCA_Eigenvectors_centred" + str(i).zfill(2)  + \
                    ".pdf",bbox_inches="tight",transparent=True)
        plt.show()
    
    # plot uncentred
    #depths = (5 * (np.arange(np.ma.size(pca_components, axis=1)) )) + 15
    for i in range(np.ma.size(pca_comp_unstand, axis=0)):

        fig = plt.figure(figsize=(7,7))

        # create plot
        plt.plot(pca_comp_unstand[i,:], depth, linewidth=2.0, color='black')

        # fix axes
        ax = plt.gca()
        ax.set_xlim(3,10)
        ax.set_ylim(0,1000)
        ax.set_ylabel("Pressure (dbar)")
#       ax.set_xlabel("Eigenvector Coefficient")
        ax.invert_yaxis()
        ax.grid(True)
#       ax.legend(loc='best')

        # save figures
        plt.savefig(address+"Plots/PCA_Eigenvectors_uncentred" + str(i).zfill(2) + \
                    ".pdf",bbox_inches="tight",transparent=True)
        plt.show()

    # plot uncentred plus/minus the profiles
    #depths = (5 * (np.arange(np.ma.size(pca_components, axis=1)) )) + 15
    for i in range(np.ma.size(pca_comp_unstand, axis=0)):

        fig = plt.figure(figsize=(7,7))

        # create plot
        yp = Tmean + pca_comp_unstand[i]
        ym = Tmean - pca_comp_unstand[i]
        plt.plot(yp, depth, linewidth=1.0, color='black', linestyle='--')
        plt.plot(Tmean, depth, linewidth=2.0, color='black', linestyle='-')
        plt.plot(ym, depth, linewidth=1.0, color='black', linestyle='--')

        # fix axes
        ax = plt.gca()
        ax.set_xlim(0,20)
        ax.set_ylim(0,1000)
        ax.set_ylabel("Pressure (dbar)")
#       ax.set_xlabel("Eigenvector Coefficient")
        ax.invert_yaxis()
        ax.grid(True)
#       ax.legend(loc='best')

        # save figures
        plt.savefig(address+"Plots/PCA_Eigenvectors_withMean" + str(i).zfill(2) + \
                    ".pdf",bbox_inches="tight",transparent=True)
        plt.show()

###########

def plotPCAmplitudeCoefficients(address, address_fronts, runIndex):

    # color maps
    colorname = 'RdBu_r'
    colormap = plt.get_cmap(colorname)

    # read reduced columns
    col_reduced = Print.readColreduced(address, runIndex)

    # read the amplitude coefficients
    [lon, lat, dynHeight, X_array, varTime] = None, None, None, None, None
    [lon, lat, dynHeight, X_array, varTime] = Print.readPCAFromFile(address, runIndex, col_reduced=6)

    # subsample
    nsub = 10
    xplot = lon[::nsub]
    yplot = lat[::nsub]

    # make some maps
    for k in range(0,col_reduced):

        cplot = X_array[:,k]
        cplot = cplot[::nsub]

        pmax = np.max([np.abs(np.min(cplot)), np.max(cplot)])
        pmin = -1.0*pmax

        proj = ccrs.SouthPolarStereo()
        proj_trans = ccrs.PlateCarree()

        ax1 = plt.axes(projection=proj)
        ax1.set_extent((-180,180,-90,-30),crs=proj_trans)

        CS = ax1.scatter(xplot, yplot, s = 3.0, lw = 0, c = cplot, \
                         cmap = colormap, transform = proj_trans, vmin = pmin, vmax = pmax)

        # plot fronts 
        SAF, SACCF, SBDY, PF = None, None, None, None
        SAF, SACCF, SBDY, PF = loadFronts(address_fronts)  
        ax1.plot(SAF[:,0], SAF[:,1], lw = 1, ls='-', label='SAF', \
                 color='black', transform=proj_trans)
        ax1.plot(PF[:,0], PF[:,1], lw = 1,ls='-', label='PF', \
                 color='grey', transform=proj_trans)
        ax1.plot(SACCF[:,0], SACCF[:,1], lw = 1,ls='-', label='SACCF', \
                 color='green', transform=proj_trans)
        ax1.plot(SBDY[:,0], SBDY[:,1], lw = 1,ls='-', label='SBDY', \
                 color='blue', transform=proj_trans)

        # compute a circle in axes coordinates, which we can use as a boundary for the map.
        theta = np.linspace(0, 2*np.pi, 100)
        center = [0.5, 0.5]
        radius = 0.52   # 0.46 corresponds to roughly 30S Latitude
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax1.set_boundary(circle, transform=ax1.transAxes)

        # add features
        ax1.gridlines()
        ax1.coastlines()
        ax1.set_extent((-180,180,-90,-30),crs=proj_trans)

        # plot colorbar
        colorbar = plt.colorbar(CS)

        # save figures
        plt.savefig(address+"Plots/PCA_AmplitudeCoefficientMaps" + str(k).zfill(2) + \
                    ".pdf",bbox_inches="tight",transparent=True)
        #plt.show()
        plt.clf()

###########

print('Plot runtime = ', time.clock() - start_time,' s')
