import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean, cosine
from collections import defaultdict
import random


def rmse(yhat, y):
    ''' Return root mean squared error of a set of predictions '''
    return np.sqrt(((yhat - y)**2).mean())


#load in dataframe (already combiend features and targets)
#df = pd.read_csv('../data/thur_data.csv')

def run_clustering(df):
    #split into forwards and defense
    defense = df[df.position == 'Defense']
    forwards = df[df.position != 'Defense']

    #defense clustering
    Xd = defense[['mean Total Points', 'mean TP/60', 'TOI/GP', '3yr CF/60 Rel', '3yr CA/60 Rel',
                 'mean Giveaways/60', 'mean Takeaways/60', '3yr Off.\xa0Zone Starts/60',
                 '3yr Def.\xa0Zone Starts/60', 'mean Shots Blocked/60']]
    yd = defense[['length', 'cap_hit', 'cap_pct']]

    ss = StandardScaler()
    Xds = ss.fit_transform(Xd)
    dkm = KMeans(n_clusters = 3)
    dclus = dkm.fit_transform(Xds)
    #create a cluster feature column
    Xd['cluster'] = dkm.labels_

    dclusts = []
    dcmeans = []
    for i in range(dkm.n_clusters):
        dclusts.append(Xd[dkm.labels_ == i].drop('cluster', axis=1))
        dcmeans.append(dclusts[i].mean(axis=0))

    #plot the defensive clusters
    w = 0.25
    idx = dcmeans[0].index

    xx = np.arange(len(idx))

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    ax.bar(x = xx - w, height = dcmeans[0], width = w, label = 'Cluster 0')
    ax.bar(x = xx, height = dcmeans[1], width = w, label = 'Cluster 1')
    ax.bar(x = xx + w, height = dcmeans[2], width = w, label = 'Cluster 2')
    ax.set_xticks(xx + w/3)
    ax.set_xticklabels(idx, rotation = 60)


    ax.legend()

    ax.set_title('Mean stats of defensive player clusters')
    plt.show()

    #forward clustering
    Xf = forwards[['mean Total Points', 'mean TP/60', 'TOI/GP', '3yr CF/60 Rel', '3yr CA/60 Rel',
                 'mean Giveaways/60', 'mean Takeaways/60', '3yr Off.\xa0Zone Starts/60',
                 '3yr Def.\xa0Zone Starts/60', 'mean Shots Blocked/60']]
    yf = forwards[['length', 'cap_hit', 'cap_pct']]

    ss = StandardScaler()
    Xfs = ss.fit_transform(Xf)

    fkm = KMeans(n_clusters = 4)
    fclus = fkm.fit_transform(Xfs)
    Xf['cluster'] = fkm.labels_

    fclusts = []
    fcmeans = []
    for i in range(fkm.n_clusters):
        fclusts.append(Xf[fkm.labels_ == i].drop('cluster', axis=1))
        fcmeans.append(fclusts[i].mean(axis=0))

    #plot forward clusters
    w = 0.22
    idx = fcmeans[0].index

    xx = np.arange(len(idx))

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    #for i in fcmeans:
    ax.bar(x = xx - 2*w, height = fcmeans[0], width = w, label = 'Cluster 0')
    ax.bar(x = xx - w, height = fcmeans[1], width = w, label = 'Cluster 1')
    ax.bar(x = xx, height = fcmeans[2], width = w, label = 'Cluster 2')
    ax.bar(x = xx + w, height = fcmeans[3], width = w, label = 'Cluster 3')

    ax.set_xticks(xx + w/5)
    ax.set_xticklabels(idx, rotation = 60)


    ax.legend()

    ax.set_title('Mean stats of forward player clusters')
    plt.show()


    #Calculate RMSE for predicting each cluster on its mean cap hit
    fmeancaphit = []
    fyclusts = []
    for x, i in enumerate(fclusts):
        cy = yf[fkm.labels_ == x]
        fyclusts.append(cy)
        print('RMSE for forwards cluster {} mean: ${}'
              .format(x, rmse(cy.cap_hit.mean(), cy.cap_hit)))
        print('Mean cap hit for forwards cluster {}: ${}'.format(x, round(cy.cap_hit.mean())))

    dmeancaphit = []
    dyclusts = []
    for x, i in enumerate(dclusts):
        cy = yd[dkm.labels_ == x]
        dyclusts.append(cy)
        print('RMSE for defense cluster {} mean: ${}'
              .format(x, rmse(cy.cap_hit.mean(), cy.cap_hit)))
        print('Mean cap hit for defense cluster {}: ${}'.format(x, round(cy.cap_hit.mean())))

    #combine forward and defensive clusters to get aggregate stats
    clusts = fclusts + dclusts
    yclusts = fyclusts + dyclusts

    return clusts, yclusts

def run_clustered_models(clusts, yclusts):
    #Run a separate model on each cluster, using 6 different model types to compare performance
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    rfmodels, knnmodels, rnrmodels = [], [], []
    linmodels = []
    adamodels = []
    grbmodels = []
    rfpreds, linpreds, adapreds, grbpreds, knnpreds, rnrpreds = [], [], [], [], [], []
    y_len=0
    for idx, c in enumerate(clusts):
        X_train, X_test, y_train, y_test = train_test_split(c, yclusts[idx]['cap_hit']
                                                            , test_size = 0.15)
        y_len += len(y_test)
        print(y_len)

        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

        rfmod = RandomForestRegressor()
        rfmod.fit(X_train, y_train)
        rfmodels.append(rfmod)
        rfp = rfmod.predict(X_test)
        rfpreds.append(rfp)

        knnmod = KNeighborsRegressor(n_neighbors=5)
        knnmod.fit(X_train, y_train)
        knnmodels.append(knnmod)
        knnp = knnmod.predict(X_test)
        knnpreds.append(knnp)

        rnrmod = RadiusNeighborsRegressor(radius=1.0)
        rnrmod.fit(X_train, y_train)
        rnrmodels.append(rnrmod)
        rnrp = rnrmod.predict(X_test)
        rnrpreds.append(rnrp)

        linmod = LinearRegression()
        linmod.fit(X_train, y_train)
        linmodels.append(linmod)
        linp = linmod.predict(X_test)
        linpreds.append(linp)

        adamod = AdaBoostRegressor()
        adamod.fit(X_train, y_train)
        adamodels.append(adamod)
        adap = adamod.predict(X_test)
        adapreds.append(adap)

        grbmod = GradientBoostingRegressor()
        grbmod.fit(X_train, y_train)
        grbmodels.append(grbmod)
        grbp = grbmod.predict(X_test)
        grbpreds.append(grbp)


    y_test = np.concatenate(y_tests)
    rfpred = np.concatenate(rfpreds)
    knnpred = np.concatenate(knnpreds)
    rnrpred = np.concatenate(rnrpreds)
    linpred = np.concatenate(linpreds)
    adapred = np.concatenate(adapreds)
    grbpred = np.concatenate(grbpreds)

    predlist = {
        'Random Forest': rfpred,
        'k Nearest Neighbors': knnpred,
        'Radius Neighbors': rnrpred,
        'Linear Regression': linpred,
        'AdaBoost': adapred,
        'Gradient Boost': grbpred
    }

    #print predictions and RMSE
    for m in predlist:
        print('Overall RMSE for {} is: ${}'.format(m,
                                                   round(rmse(predlist[m], y_test), 2)))

    return predlist
