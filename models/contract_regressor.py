import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

#Key word arguments for optimal GradientBoostRegressor performance
kwt = {'alpha': 0.9, 'criterion': 'friedman_mse', 'init':None,
             'learning_rate':0.1, 'loss':'ls', 'max_depth':3, 'max_features':None,
             'max_leaf_nodes':None, 'min_impurity_decrease':0.0,
             'min_impurity_split':None, 'min_samples_leaf':1,
             'min_samples_split':2, 'min_weight_fraction_leaf':0.0,
             'n_estimators':100, 'presort':'auto', 'random_state':None,
             'subsample':1.0, 'verbose':0, 'warm_start':False}


class ContractRegressor():
    '''
    A class for multi-target prediction of salary cap percent, and length of
    contract.
    Specific column titles in the pick_cols method are based on stats
    obtainable from Natural Stat Trick.
    '''
    def __init__(self, m = GradientBoostingRegressor, year = 2019, kwargs = kwt):
        self.m = m
        self.kwargs = kwargs

        #A dictionary containing the salary cap ceiling for any given year
        #   Note: 2019 is a speculative estimate based on a press release
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

        self.year = year
        self.year_cap = self.scap[year]
        pass

    def pick_cols(self, df):
        '''
        Pull out feature columns to predict on from a larger dataframe. This function
        assumes that there are no irrelevant columns after the first few.
        '''
        #These columns were causing problems
        x = df.drop(['mean SH%', 'IPP', 'mean Faceoffs pct'], axis =1)
        #pull out relevant columns as a list
        self.xcols = ['forward', 'signing_age', 'signing_status'] + list(x.columns)[-194:]
        #return a dataframe containing only the relevant columns
        return x[self.xcols]

    def drop_duplicate_rows(self, X):

        #Drop any duplicate rows
        X.reset_index(inplace=True)
        X.drop_duplicates(subset = ['Season_Player'], inplace=True)
        X.set_index(X.Season_Player, inplace=True)
        X.drop('Season_Player', axis=1, inplace=True)
        return X

    def impute(self, X):
        '''
        Handle any NaN values in the dataframe. Returns an imputed version of
        the feature dataframe.
        '''
        #Store index for later
        idx = X.index

        #My data tended to contain a few '-' values instead of 0s, fix that.
        X.replace('-', 0, inplace=True)

        #Replace True/False with 1/0
        X.replace([True, False, 'True', 'False'], [1, 0, 1, 0], inplace=True)

        #Enforce conversion to float or int for any columns that had '-' and were
        #dtype: object
        X = X.apply(pd.to_numeric)


        #perform imputing for any columns still missing values
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose = 1)
        Ximp = imp.fit_transform(X)

        return pd.DataFrame(Ximp, index=idx, columns=self.xcols)

    def fit(self, X_train, yp_train, yl_train):
        '''
        Fit a model (default sklearn's GradientBoostingRegressor) for salary cap_pct
        and contract length
        '''
        #Store training data
        self.X_train = X_train
        self.yp_train = yp_train
        self.yl_train = yl_train

        #instantiate and fit a model for cap_pct
        self.sal_model = self.m(**self.kwargs)
        self.sal_model.fit(X_train, yp_train)

        #Add predicted cap_pct as a feature for the length model
        Xl_train = X_train.copy()
        Xl_train = np.hstack((Xl_train, self.sal_model.predict(Xl_train).reshape(-1,1)))

        #instantiate and fit a model for contract length
        self.len_model = self.m(**self.kwargs)
        self.len_model.fit(Xl_train, yl_train)
        pass

    def predict(self, X):
        '''
        Predict salary cap_pct and contract length
        Returns two arrays of predictions for cap_pct and length
        '''
        #predict cap_pct
        self.cap_preds = self.sal_model.predict(X)

        #add predicted cap_pct as a feature for the length model
        Xl_test = X.copy()
        Xl_test = np.hstack((Xl_test, self.cap_preds.reshape(-1,1)))

        #predict length
        self.length_preds = self.len_model.predict(Xl_test)

        return self.cap_preds, self.length_preds

    def score(self, X_test, yp_test, yl_test):
        '''
        Return RMSE for salary cap_pct, equivalent salary in real dollars for 2019 (default),
        and RMSE for contract length.
        Prints output and returns RMSE for both as well as cap hit in real dollars.
        '''
        #get predictions
        cap_preds, length_preds = self.predict(X_test)

        #get RMSE score for cap_pct
        self.p_rmse = self.rmse(cap_preds, yp_test, 2)
        #convert cap_pct to real dollars
        self.cap_hit = self.p_rmse * self.year_cap // 100
        #get RMSE score for length
        self.l_rmse = self.rmse(length_preds, yl_test, 1)
        #print output
        print('RMSE Cap_pct: {}%'.format(self.p_rmse))
        print('    translates to {} Cap Hit of: ${}'.format(self.year, self.cap_hit))
        print('RMSE Length: {} years'.format(self.l_rmse))
        return self.p_rmse, self.cap_hit, self.l_rmse

    def rmse(self, yhat, y, round_to = None):
        '''
        Return root mean squared error of a set of predictions with an
        optional rounding term.
        Inputs:
        yhat: an array of predictions
        y: a corresponding array of actual values
        round_to: an optional rounding term (default = None)
        Returns:
        err: A single RMSE value
        '''
        err = np.sqrt(((yhat - y)**2).mean())
        if round_to:
            return round(err, round_to)
        else:
            return err

    def predict_global_mean(self, yp_train, yp_test, yl_train, yl_test):
        '''
        Print the RMSE if the mean values were to be predicted every time.
        '''
        pcmp = yp_train.mean()
        p_rmse = self.rmse(pcmp, yp_test, 2)
        print('RMSE pick mean cap_pct: {}%'.format(p_rmse))
        print('    translates to 2019 Cap Hit of: ${}'.format(p_rmse * self.year_cap // 100))
        pcml = yl_train.mean()
        print('RMSE pick mean length: {} years'.format(self.rmse(pcml, yl_test, 1)))
        return pcmp, pcml

def prediction_cleaning(sal_preds, len_preds, idx):
    '''
    Output cap hit and length predictions into a readable pandas dataframe.
    Inputs:
    sal_preds: 1-D numpy array
    len_preds: 1-D numpy array
    idx: the index of the original data's DataFrame
    Returns:
    A 3 column pandas dataframe
    '''
    #create dataframe
    preds_df = pd.DataFrame(index=idx)
    preds_df['cap_pct'] = sal_preds
    preds_df['length'] = len_preds

    #A helper function that improves the readability of our output for cap_hit in dollars
    def int_to_mon_str(x):
        s = str(x)
        if x < 1000000:
            s = '$' + s[:3] + ',' + s[3:]
        elif x < 10000000:
            s = '$' + s[0] + ',' + s[1:4] + ',' + s[4:]
        else:
            s = '$' + s[0:2] + ',' + s[2:5] + ',' + s[5:]
        return s

    #add column for cap_hit in dollars
    preds_df['2019_cap_hit'] = fa_preds_df.cap_pct * 83000000 // 100
    preds_df['2019_cap_hit'] = fa_preds_df['2019_cap_hit'].apply(lambda x: int(round(x, -3)))
    preds_df['2019_cap_hit'] = fa_preds_df['2019_cap_hit'].apply(int_to_mon_str)

    #Improve readability of prediction columns
    preds_df['cap_pct'] = fa_preds_df['cap_pct'].apply(lambda x: round(x, 2))
    preds_df['length'] = fa_preds_df['length'].apply(lambda x: round(x, 1))

    return preds_df
