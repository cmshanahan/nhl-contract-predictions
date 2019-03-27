import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean_data(file = None):
    #load in dataframe from csv
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv('../data/Contract_Details_PuckPedia_Mar_2019_Confidential.csv')

    #Drop columns I won't be using
    df.drop(['buyout_id', 'base_salary', 'p_bonuses', 's_bonuses',
            'total_salary'], axis=1, inplace=True)

    #rename Total Value column to a format I prefer
    df['total_value'] = df[' Total Value ']
    df.drop(' Total Value ', axis=1, inplace=True)

    #create a column for full player name
    df['Player'] = df['first_name'] + ' ' + df['last_name']

    #make a function to convert money strings to ints and apply
    money_to_int = lambda x: int(x.strip().strip('$()').replace(',', ''))
    df['cap_hit'] = df['cap_hit'].apply(money_to_int)
    df['total_value'] = df['total_value'].apply(money_to_int)

    #convert signing date to date format and pull out the signing year
    df.signing_date = pd.to_datetime(df.signing_date)
    df['signing_date'] = pd.DatetimeIndex(df['signing_date']).date
    df['signing_year'] = pd.DatetimeIndex(df.signing_date).year

    #remove row that had NaN for signing year (Defenseman Fyodor Tyutin had
    #the contract listed in other rows also so no data is lost
    df.drop(6991, axis=0, inplace=True)

    #Make signing year an int instead of a float
    df.signing_year = df.signing_year.apply(int)

    #Manually entered league salary cap history
    scap = {
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

    #Add a column for the total salary cap in a contract's year signed
    df['signing_year_cap'] = df['signing_year'].apply(lambda x: scap[x])
    # and a column for the percentage of the cap in the year signed
    df['cap_pct'] = round(100 * df.cap_hit / df.signing_year_cap, 2)

    #convert birthdate to pandas datetime
    df.birthdate = pd.to_datetime(df.birthdate)

    #get age at contract signing_date (not accounting for leap days)
    df['signing_age'] = df.signing_date - df.birthdate
    df.signing_age = df.signing_age.apply(lambda x: x.days // 365)

    #convert current season and contract_end to single year ints for ease of
    #calculations
    df.season = df.season.apply(lambda x: int(x[:4]))
    df.contract_end = df.contract_end.apply(lambda x: int(x[:4]))

    #Drop some non-standard situations (i.e. suspensions, season-opening IR,
    # salary retention adjustments)
    df.drop(df[df.first_name.map(len) > 15].index, inplace=True)

    #Drop one extremely young player
    df.drop(df[df.ufa_year > 2030].index, inplace = True)

    #Drop where ufa_year is null
    df.drop(df[df.ufa_year.isnull()].index, inplace=True)
    #Turn the remaining ufa years into integers
    df.ufa_year = df.ufa_year.apply(int)

    #Drop any contracts less than league minimum in 2009
    df.drop(df[df.cap_hit < 500000].index, inplace=True)

    #Replace the position names for two players whose positions don't match the rest
    df.replace(['RW', 'LW'], ['Right Wing', 'Left Wing'], inplace=True)

    #drop entry level contracts
    df.drop(df[df.contract_level == 'entry_level'].index, inplace=True)

    #Add a column for whether the contract was signed as UFA or RFA
    dff['signing_status'] = dff['signing_year'] >= dff['ufa_year']
    dff['signing_status'] = dff['signing_status'].apply(ufa_check)


    return df

#auxillary function to check if contract was signed as UFA or RFA
def ufa_check(x):
    if x == True:
        return 'UFA'
    else:
        return 'RFA'
