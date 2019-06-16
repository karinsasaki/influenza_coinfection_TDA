#!/usr/bin/env python
# coding: utf-8

#------------------------------------------------------------------
# IMPORT LIBRARIES
#------------------------------------------------------------------

# General
import pandas as pd
import numpy as np
import re
import csv
import itertools
import copy

# import seaborn as sns
# from scipy import stats

# import pickle as pickle
# from pylab import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# from IPython.display import Image
# from mpl_toolkits.mplot3d import Axes3D 

# for KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, euclidean_distances
from sklearn.metrics import pairwise_distances

# for KeplerMapper
import kmapper as km
from kmapper.plotlyviz import *

from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import manifold
from sklearn.neighbors.kde import KernelDensity
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.decomposition import TruncatedSVD


import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objs as go
# from ipywidgets import (HBox, VBox)

# print("libraries imported ...")


#------------------------------------------------------------------
# FUNCTIONS FOR GRID SEARCH FOR VALUES OF THE PARAMETERS OF THE KEPLER MAPPER
#------------------------------------------------------------------

# Python program to print connected components in an undirected graph 
# Reference: https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/

class Graph: 
    
    # init function to declare class variables 
    def __init__(self,V): 
        self.V = V 
        self.adj = [[] for i in range(V)] 

    def DFSUtil(self, temp, v, visited): 

        # Mark the current vertex as visited 
        visited[v] = True

        # Store the vertex to list 
        temp.append(v) 

        # Repeat for all vertices adjacent 
        # to this vertex v 
        for i in self.adj[v]: 
            if visited[i] == False: 
                
                # Update the list 
                temp = self.DFSUtil(temp, i, visited) 
        return temp 

    # method to add an undirected edge 
    def addEdge(self, v, w): 
        self.adj[v].append(w) 
        self.adj[w].append(v) 

    # Method to retrieve connected components 
    # in an undirected graph 
    def connectedComponents(self): 
        visited = [] 
        cc = [] 
        for i in range(self.V): 
            visited.append(False) 
        for v in range(self.V): 
            if visited[v] == False: 
                temp = [] 
                cc.append(self.DFSUtil(temp, v, visited)) 
        return cc 


def count_connected_components(simplicial_complex):
    """
    Description:
        Count the number of connected components of a simplicial complex.
    Input arguments:
        simplicial_complex: output from the Keppler-Mapper map function
    Output variables:
        n_v: total number of vertices of the topological graph
        n_cc: total number of connected components of the topological graph
    """
    
    # nodes of simplicial_complex
    nodes = dict([(y,x) for x,y in enumerate(sorted(set(simplicial_complex['nodes'])))])

    # simplicial complex as a graph
    g = Graph(len(simplicial_complex['nodes'])); 

    for val in simplicial_complex['links']:
        l = len(simplicial_complex['links'][val])
        for i in np.arange(l):
            g.addEdge(nodes[val], nodes[simplicial_complex['links'][val][i]]) 

    # number of connected components
    n_v = len(nodes)
    n_cc = len(g.connectedComponents() )        
    
    return(n_v, n_cc)


def def_lenses_features(df, fs):
    
    mapper = km.KeplerMapper()    
    
    keys = []
    values = []

    for idx,col in enumerate(fs):
        keys.append("lens_{}".format(col))
        values.append(mapper.fit_transform(df[fs].as_matrix(), projection=[idx], scaler=MinMaxScaler()))

    lenses_features = dict(zip(keys, values))
    
    return(lenses_features)


def def_lenses_dimred(df, fs, get_PCA, get_isomap, get_LLE, get_MDS, get_spectral_embedding, get_SVD):
    
    scaler = MinMaxScaler()
    
    mapper = km.KeplerMapper() 
    
    X = df[fs].as_matrix()
    
    keys = []
    values = []

    minmax_scaler = MinMaxScaler()
    df_minmax = minmax_scaler.fit_transform(df[fs].as_matrix())

    # PCA
    if get_PCA==True:
        keys.append('lens_pca_0')
        keys.append('lens_pca_1')
        pca = mapper.fit_transform(df_minmax, projection=PCA(n_components=2), scaler=None)
        values.append(scaler.fit_transform(pca[:,0].reshape(-1,1)))
        values.append(scaler.fit_transform(pca[:,1].reshape(-1,1)))

    # Isomap
    if get_isomap==True:
        keys.append('lens_isomap_0')
        keys.append('lens_isomap_1')
        isomap = manifold.Isomap(n_components=2, n_neighbors=3).fit_transform(df_minmax)
        values.append(scaler.fit_transform(isomap[:,0].reshape(-1,1)))
        values.append(scaler.fit_transform(isomap[:,1].reshape(-1,1)))

    # Locally linear embedding 
    if get_LLE==True:
        keys.append('lens_LLE_0')
        keys.append('lens_LLE_1')
        LLE = manifold.locally_linear_embedding(df_minmax, n_neighbors=3, n_components=2, random_state=0)[0]
        values.append(scaler.fit_transform(LLE[:,0].reshape(-1,1)))
        values.append(scaler.fit_transform(LLE[:,1].reshape(-1,1)))
        
    # Multi-dimensional scaling
    if get_MDS==True:
        keys.append('lens_MDS_0')
        keys.append('lens_MDS_1')
        MDS = manifold.MDS(n_components=2).fit_transform(df_minmax)
        values.append(scaler.fit_transform(MDS[:,0].reshape(-1,1)))
        values.append(scaler.fit_transform(MDS[:,1].reshape(-1,1)))
        
    # Spectral embedding
    if get_spectral_embedding == True:
        keys.append('lens_spectral_embedding_0')
        keys.append('lens_spectral_embedding_1')
        L = manifold.SpectralEmbedding(n_components=2, n_neighbors=1, random_state=0).fit_transform(df_minmax)
        values.append(scaler.fit_transform(L[:,0].reshape(-1,1)))
        values.append(scaler.fit_transform(L[:,1].reshape(-1,1)))
        
    # truncated SVD
    if get_SVD == True:
        keys.append('lens_SVD_0')
        keys.append('lens_SVD_1')
        svd = TruncatedSVD(n_components=2, random_state=42).fit_transform(df_minmax)
        values.append(scaler.fit_transform(svd[:,0].reshape(-1,1)))
        values.append(scaler.fit_transform(svd[:,1].reshape(-1,1)))
                
    lenses_dimred = dict(zip(keys, values))
    
    return(lenses_dimred)


def def_lenses_neighbours(df, fs, labels, metric):

    scaler = MinMaxScaler()
    
    X = df[fs].as_matrix()
    
    keys = []
    values = []
    
    if metric == 'cosine':
        X_cosine_distance = cosine_similarity(X)
        X_dist = np.abs(X_cosine_distance - 1)
        neighcd = KNeighborsClassifier(n_neighbors=3, weights='distance', 
                                   algorithm='brute', metric='precomputed').fit(X_dist, labels)
        dist, ind = neighcd.kneighbors(X_dist, return_distance=True)
    if metric == 'euclidean':
        neighcd = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(X, labels) 
        dist, ind = neighcd.kneighbors(X, return_distance=True)
    if metric == 'correlation':
        neighcd = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='correlation').fit(X, labels) 
        dist, ind = neighcd.kneighbors(X, return_distance=True)
        
    dist.sort(axis=1)
    
    for i in np.arange(np.shape(dist)[-1]):
        keys.append('Neighbour_{}'.format(i))
        values.append( scaler.fit_transform(dist[:,i].reshape(-1,1)) )
    keys.append('Sum')
    values.append(scaler.fit_transform(np.sum(dist, axis=1).reshape(-1,1)))
    
    lenses_nbrs = dict(zip(keys, values))
    
    return(lenses_nbrs)

    
def def_lenses_geometry(df, fs, get_density, get_eccentricity, eccentricity_exponent, 
                        get_inf_centrality, others, metric):

    scaler = MinMaxScaler()
    
    X = df[fs].as_matrix()
    
    if metric == 'cosine':
        X_cosine_distance = cosine_similarity(X)
        X_dist = np.abs(X_cosine_distance - 1)
    if metric == 'euclidean':
        X_dist = euclidean_distances(X)
    if metric == 'correlation':
         X_dist = pairwise_distances(X, metric='correlation')
            
    keys = []
    values = []

    # density - see: https://scikit-learn.org/stable/modules/density.html
    if get_density == True:
        keys.append('lens_density')

        # calc bandwidth using Scott’s Rule, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        n = np.shape(X)[0]
        d = np.shape(X)[1]
        bandwidth = n**(-1./(d+4))

        # calc density
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
        density = kde.score_samples(X)
        values.append(scaler.fit_transform(density.reshape(-1,1)))
    
    # eccentricity
    if get_eccentricity == True:
        keys.append('lens_eccentricity')
        a = X_dist ** eccentricity_exponent
        b = np.sum(a, axis=1)
        c = b/np.shape(X_dist)[0]
        eccentricity = c ** (1/eccentricity_exponent)
        values.append(scaler.fit_transform(eccentricity.reshape(-1,1)))
        
    # inf centrality
    if get_inf_centrality == True:
        keys.append('lens_inf_centrality')
        inf_centrality = np.amax(X_dist, axis=1)
        values.append(scaler.fit_transform(inf_centrality.reshape(-1,1)))
        
    mapper = km.KeplerMapper()    
    
    if others == True:
        for metric in ["sum", "mean", "median", "max", "min", "std", "dist_mean", "l2norm"]:
            keys.append("lens_{}".format(metric))
            values.append(mapper.fit_transform(df[fs].as_matrix(), projection=metric, scaler=MinMaxScaler()))
                    
    lenses_geometry = dict(zip(keys, values))
    
    return(lenses_geometry)


def mapper_parameter_gridsearch(df, fs, labels, metric, lenses_dict, parameter_values, 
                                num_connected_components, filepath):
    
 
    mapper = km.KeplerMapper()
    
    X = np.array(df[fs])
   
        # for dataframe
    df_temp = []

#     idx = 0
    for lens1, lens2, int1, int2, pc1, pc2, eps in parameter_values:

        # Combine lenses
        lens = np.c_[lenses_dict[lens1], lenses_dict[lens2]]
        
        if metric == 'cosine':
            X_cosine_distance = cosine_similarity(X)
            X_dist = np.abs(X_cosine_distance - 1)
            scomplex = mapper.map(lens, 
                          X_dist, 
                          cover=km.cover.Cover(n_cubes=[int1, int2],perc_overlap=[pc1, pc2]),
                          clusterer=DBSCAN(metric='precomputed', eps=eps, min_samples=1),
                          precomputed=True)
        if metric == 'euclidean':
            scomplex = mapper.map(lens, 
                                  X, 
                                  cover=km.cover.Cover(n_cubes=[int1, int2],perc_overlap=[pc1, pc2]),
                                  clusterer=DBSCAN(metric='euclidean', eps=eps, min_samples=1),
                                  precomputed=False)
        if metric == 'correlation':
            scomplex = mapper.map(lens, 
                                  X, 
                                  cover=km.cover.Cover(n_cubes=[int1, int2],perc_overlap=[pc1, pc2]),
                                  clusterer=DBSCAN(metric='correlation', eps=eps, min_samples=1),
                                  precomputed=False)

        # Calculate number of connected components
        n_v, n_cc = count_connected_components(scomplex)

        # Append data to list for dataframe only if the simplex has num_connected_components
        # or less connected components
        if n_cc <= num_connected_components:
            df_temp.append([lens1, lens2, int1, int2, pc1, pc2, eps, n_v, n_cc])
            
    # Create dataframe
    print('Saving to data frame...')
    columns = ['lens1', 'lens2',  
               'lens1_n_cubes', 'lens2_n_cubes', 
               'lens1_perc_overlap', 'lens2_perc_overlap', 
               'eps', 'n_vertices', 'n_connected_components']
    df_sc = pd.DataFrame(data=df_temp, columns=columns)

    # save df to file
    print('Saving to file...')
    df_sc.to_csv(filepath)
        
    print('Done...')
    
    return(df_sc)



#------------------------------------------------------------------
# FUNCTIONS FOR PHASE PLOTS
#------------------------------------------------------------------

def calc_mean(df, feature):
    """
    Description:
        Calculate the mean values of one feature at each time point.
    Input arguments:
        df: name of dataframe of interest
        feature: feature of interest, as a str
            e.g. IAV_IFNgamma_mean = calc_mean(AIV,'IFN-gamma')
    """
    
    data = df[['time point', feature]]        
    data_mean = data.groupby('time point').mean()
    dic = {'time point':np.array(data_mean.index.tolist()), feature:np.array(data_mean[feature].tolist())}
    data_mean = pd.DataFrame.from_dict(dic)
    return(data_mean)




def calc_mean2(df, features):
    """
    Description:
        Calculate the mean values of all features at each time point.
    Input arguments:
        df: name of dataframe of interest
        feature: features of the dataframe as a list of strings
            e.g. IAV_means = calc_mean2(AIV, features)
    """
    
    
    data = df[['time point', features[0]]]        
    data_mean = data.groupby('time point').mean()
    dic = {'time point':np.array(data_mean.index.tolist()), features[0]:np.array(data_mean[features[0]].tolist())}
    data_mean = pd.DataFrame.from_dict(dic)

    for feature in features[1:]:
        data = df[['time point', feature]]        
        data_mean_temp = data.groupby('time point').mean()
        dic = {'time point':np.array(data_mean_temp.index.tolist()), feature:np.array(data_mean_temp[feature].tolist())}
        data_mean_temp = pd.DataFrame.from_dict(dic)
        data_mean[feature] = data_mean_temp[feature]
    return(data_mean)


# calculate if two line segments intersect
# from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

def ccw(A,B,C):
    return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


# e.g. 
#a = Point(0,0)
#b = Point(0,1)
#c = Point(1,1)
#d = Point(1,0)

#print(intersect(a,b,c,d))
#print(intersect(a,c,b,d))
#print(intersect(a,d,b,c))



def PolyArea(x,y):
    """
    Calculate polygon area using the Shoelace formula
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



def polygon_area(df, features):

    """
    Close the loops with a line segment and calculate the area of the polygon
    """
    
    areas = np.empty((len(features),len(features)))
    areas[:] = np.nan
    
    for i,col1 in enumerate(features):
        for j,col2 in enumerate(features):
            
            if i>=j:
                continue
            #elif i==j:
            #    areas[i,j] = np.nan
            else:
                # calc mean values for each time point
                data1_mean = calc_mean(df,col1)
                data2_mean = calc_mean(df,col2)
                #print(data1_mean)
                #print(data2_mean)
                #input()

                # get the point coordinates for the connecting line segment
                a = Point(data1_mean.iloc[0,1],data2_mean.iloc[0,1])
                b = Point(data1_mean.iloc[-1,1],data2_mean.iloc[-1,1])

                # make a list of points
                points = [Point(data1_mean.iloc[x,1],data2_mean.iloc[x,1]) for x in np.arange(len(data1_mean))]

                # make a list of segments
                segments = []
                for k in np.arange(len(points)-1):
                    segments.append([points[k],points[k+1]])

                # check that segments on the phase plot does not intersect the closing segment
                intersections = []
                for l in np.arange(len(points)-1):
                    intersections.append(intersect(points[l],points[l+1],a,b))

                # check no two segments on the phase plots intersect each other
                for m in np.arange(len(segments)):
                    for n in np.arange(len(segments)):
                        if n<=m:
                            continue
                        intersections.append(intersect(segments[m][0],segments[m][1],segments[n][0],segments[n][1]))

                #print(intersections)
                if any(intersections):
                    #areas[i,j] = np.nan
                    continue
                elif PolyArea(data1_mean[col1], data2_mean[col2])==0:
                    continue
                else:
                    # close the polygons calculate and store their area
                    areas[i,j] = PolyArea(data1_mean[col1], data2_mean[col2])

                #print(areas[i,j])
                #input()
    areas = pd.DataFrame(index=features, columns=features, data=areas)
    #areas = areas.round(roundto)
    return(areas)
                



def polygon_plot(df_list, df_names, features, share_yaxes=False, save=False):
    """
    Description:
        polygon phase plots of the infection groups side by side for each feature
    Input arguments:
        df_list: a list of dataframes to compare; all having the same features; 
            e.g. df_list = [IAV, T4, IAVT4]
        df_names: list of the names of the dataframes;
            e.g. df_names = ['IAV', 'T4', 'IAVT4']
        features: list of features to plot;
            e.g. features = features
        polygonal_areas: list of output from polygon_area for each dataframe in df_list;
            e.g. polygonal_areas = [polygon_area(IAV, features), polygon_area(T4, features), polygon_area(IAVT4, features)]
        share_yaxes: specify whether the subplots should share yaxes values
    """
    
    polygonal_areas = [polygon_area(df_list[0], features), polygon_area(df_list[1], features), polygon_area(df_list[2], features)]
    
    for j,col1 in enumerate(features):
        for k,col2 in enumerate(features):
            
            if k<=j:
                continue
            
            # new figure
            fig = plt.figure(figsize = (11,4))
            
            i=1
            for df in df_list:
                
                # create and annotate each subplot            
                if share_yaxes==True:
                    if i==1:
                        ax1 = fig.add_subplot(1,3,i) 
                        ax1.set_xlabel(col1)
                        ax1.set_ylabel(col2) 
                        ax1.set_title('{} \n Area = {:.1e}'.format(df_names[i-1],polygonal_areas[i-1].loc[col1,col2]))
                        plt.setp(ax1.get_yticklabels(), fontsize=10)
                    elif i==2:
                        ax2 = fig.add_subplot(1,3,i, sharex=ax1, sharey=ax1) 
                        ax2.set_xlabel(col1)
                        ax2.set_ylabel(col2)
                        #ax2.set_title('{}'.format(df_names[i-1]))
                        ax2.set_title('{} \n Area = {:.1e}'.format(df_names[i-1],polygonal_areas[i-1].loc[col1,col2]))
                        plt.setp(ax2.get_yticklabels(), visible=False)
                    else:
                        ax3 = fig.add_subplot(1,3,i, sharex=ax1, sharey=ax1) 
                        ax3.set_xlabel(col1)
                        ax3.set_ylabel(col2)
                        #ax3.set_title('{}'.format(df_names[i-1]))
                        ax3.set_title('{} \n Area = {:.1e}'.format(df_names[i-1],polygonal_areas[i-1].loc[col1,col2]))
                        plt.setp(ax3.get_yticklabels(), visible=False)
#                         plt.legend(custom_lines, ['1.5', '6.0', '18.0', '26.0', '31.0', 'closing line'],loc=2, bbox_to_anchor=(1, 1), ncol=1)
                else:
                    ax = fig.add_subplot(1,3,i) 
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2) 
                    ax.set_title('{}'.format(df_names[i-1]))
                    ax.set_title('{} \n Area = {:.1e}'.format(df_names[i-1],polygonal_areas[i-1].loc[col1,col2]))
                    if i==1:
                        plt.setp(ax.get_yticklabels(), fontsize=10)
                    elif i==2:
                        plt.setp(ax.get_yticklabels(), visible=True)
                    else:
                        plt.setp(ax.get_yticklabels(), visible=True)
#                         plt.legend(custom_lines, ['1.5', '6.0', '18.0', '26.0', '31.0', 'closing line'],loc=2, bbox_to_anchor=(1, 1), ncol=1)
                
                # calc mean values for each time point
                data1_mean = calc_mean(df,col1)
                data2_mean = calc_mean(df,col2)

                # get the point coordinates for the connecting line segment
                a = Point(data1_mean.iloc[0,1],data2_mean.iloc[0,1])
                b = Point(data1_mean.iloc[-1,1],data2_mean.iloc[-1,1])
            
                # plot the data points
#                 plt.scatter(data1_mean[col1], 
#                             data2_mean[col2], 
# #                             s=data1_mean['time point']*5, 
#                             #c=sorted_color_names,
#                             alpha=0.4, edgecolors='w', color='blue')

                # connect the datapoints
                plt.plot(data1_mean[col1], 
                         data2_mean[col2], 
                         linestyle='-', 
                         alpha=1, 
                         color='k', linewidth=4)
                
                # close the polygon
                plt.plot([a.x,b.x],[a.y,b.y], linestyle='--', alpha=1, color='gray', linewidth=4)

                i=i+1
                
#                 plt.axis('off')
                
            plt.tight_layout() 
            plt.show() 
            
            if save==True:
                fig.savefig('./phase_plots_{}_vs_{}.pdf'.format(col1,col2), bbox_inches='tight', format='pdf')


        
# colorings

custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='b', label='1.5 hpc', markersize=3, alpha=0.4),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='b', label='6.0 hpc', markersize=5.0, alpha=0.4),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='b', label='18.0 hpc', markersize=7.0, alpha=0.4),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='b', label='26.0 hpc', markersize=9.0, alpha=0.4),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='b', label='31.0 hpc', markersize=10.0, alpha=0.4),
                Line2D([0], [0], linestyle='--', color='blue', label='closing line')
               ]

