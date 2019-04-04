import numpy as np
import pandas as pd
from collections import defaultdict
import random
import operator
from sklearn.preprocessing import StandardScaler, Imputer
from scipy.spatial.distance import euclidean, cosine
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor






class SplitModel():

    def __init__(self, num_f_clusts, num_d_clusts):
        self.num_f_clusts = num_f_clusts
        self.num_d_clusts = num_d_clusts

        self.scap = {
                2005: 39000000,
                2006: 44000000,
                2007: 50300000,
                2008: 56700000,
                2009: 56800000,
                2010: 59400000,
                2011: 64300000,
                2012: 60000000,
                2013: 64300000,
                2014: 69000000,
                2015: 71400000,
                2016: 73000000,
                2017: 75000000,
                2018: 79500000,
                2019: 83000000,
               }
        pass

    def fit(self, players):
        '''
        Pass in forwards and defense (with labels)
        Cluster, then select and fit the best model for each cluster
        X -
        y -

        '''

        self.forwards = players[players.position != 'Defense']
        self.defense = players[players.position == 'Defense']

        # signing_year isn't actually a target, but it's useful to keep with cap_hit
        xcols = []
        ycols = ['length', 'cap_hit', 'cap_pct', 'signing_year']

        xf = self.forwards[xcols]
        yf = self.forwards[ycols]
        xd = self.defense[xcols]
        yd = self.defense[ycols]

        Xf_train, Xf_test, yf_train, yf_test = train_test_split(xf, yf)

        self.forwards = players[players.position != 'Defense']
        self.defense = players[players.position == 'Defense']

        pass

    def predict(self, X_test):
        '''
        Predict salaries, and then contract lengths.
        '''
        #
        pass

    def cluster_fit(self):
        '''
        Assign cluster labels to data
        '''
        # pull out columns to cluster on
        Xf = self.forwards[['mean Total Points/60', '3yr Off. Zone Start %', 'TOI/GP']]
        Xd = self.defense[['mean Total Points/60', '3yr Off. Zone Start %', 'TOI/GP']]

        # Scale data for clustering
        fss = StandardScaler()
        Xfs = fss.fit_transform(Xf)
        dss = StandardScaler()
        Xds = dss.fit_transform(Xd)

        # Instantiate KMeans cluster objects for forwards and defense
        self.fkm = KMeans(n_clusters = self.num_f_clusts)
        self.dkm = Kmeans(n_clusters = self.num_d_clusts)

        # fit the clusters
        fclus = self.fkm.fit_transform(Xfs)
        dclus = self.dkm.fit_transform(Xds)

        # assign cluster labels to original data
        self.forwards['cluster'] = fkm.labels_
        self.defense['cluster'] = dkm.labels_

        pass




    def rmse(self, yhat, y):
    ''' Return root mean squared error of a set of predictions '''
        return np.sqrt(((yhat - y)**2).mean())
