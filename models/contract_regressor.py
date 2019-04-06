import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from collections import defaultdict


class ContractPredictor():
    def __init__(self, m = GradientBoostingRegressor, year = 2019):
        self.m = m

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

    def fit(self, X_train, yp_train, yl_train):
        '''
        Fit a model (default sklearn's GradientBoostingRegressor) for salary cap_pct
        and contract length
        '''
        self.X_train = X_train
        self.yp_train = yp_train
        self.yl_train = yl_train

        self.sal_model = self.m()
        self.sal_model.fit(X_train, yp_train)

        Xl_train = X_train.copy()
        Xl_train = np.hstack((Xl_train, self.sal_model.predict(Xl_train).reshape(-1,1)))

        self.len_model = self.m()
        self.len_model.fit(Xl_train, yl_train)
        pass

    def predict(self, X):
        '''
        Predict salary cap_pct and contract length
        '''
        self.cap_preds = self.sal_model.predict(X)
        Xl_test = X.copy()
        Xl_test = np.hstack((Xl_test, self.cap_preds.reshape(-1,1)))
        self.length_preds = self.len_model.predict(Xl_test)

        return self.cap_preds, self.length_preds

    def score(self, X_test, yp_test, yl_test):
        '''
        Return RMSE for salary cap_pct, equivalent salary in $ for 2019 (default),
        and RMSE for contract length
        '''
        cap_preds, length_preds = self.predict(X_test)
        self.p_rmse = self.rmse(cap_preds, yp_test, 2)
        self.cap_hit = self.p_rmse * self.year_cap // 100
        self.l_rmse = self.rmse(length_preds, yl_test, 1)
        print('RMSE Cap_pct: {}%'.format(self.p_rmse))
        print('    translates to {} Cap Hit of: ${}'.format(self.year, self.cap_hit))
        print('RMSE Length: {} years'.format(self.l_rmse))
        return self.p_rmse, self.cap_hit, self.l_rmse

    def rmse(self, yhat, y, round_to = None):
        ''' Return root mean squared error of a set of predictions '''
        err = np.sqrt(((yhat - y)**2).mean())
        if round_to:
            return round(err, round_to)
        else:
            return err
