import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2 as pg2
from collections import defaultdict

def clean_FA():
    df = pd.read_csv('../data/free_agents_2019.csv')
    df['signing_status'] = df['ufa_status'].apply(lambda x: 1 if 'UFA' else 0)
    df = df[(df.position != 'G') & (df.position != 27) & (df.position != 30)]
    df['forward'] = (df.position != 'D').replace([True, False], [1, 0])
    df['Season_Player'] = '2018 ' + df['Player']
    df.set_index(df['Season_Player'], inplace=True)
    df.drop('Season_Player', axis=1, inplace=True)
    fa = df[['Player', 'signing_age', 'forward', 'signing_status']]
    fa['signing_year'] = 2019
    return fa

def clean_contracts_data(file = None, goalies = False):
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

    #convert signing date and birthdate to date format and pull out the signing year
    df.signing_date = pd.to_datetime(df.signing_date)
    df.birthdate = pd.to_datetime(df.birthdate)
    df['signing_year'] = pd.DatetimeIndex(df.signing_date).year

    #get age at contract signing_date (not accounting for leap days)
    df['signing_age'] = df.signing_date - df.birthdate
    df.signing_age = df.signing_age.apply(lambda x: x.days // 365)

    #convert signing_date to a date instead of a datetime
    df['signing_date'] = pd.DatetimeIndex(df['signing_date']).date

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

    #add dummy variables for positions
    df['forward'] = (df.position != 'Goaltender') & (df.position != 'Defense')
    df.replace([True, False], [1, 0], inplace=True)

    #drop entry level contracts
    df.drop(df[df.contract_level == 'entry_level'].index, inplace=True)

    #Add a column for whether the contract was signed as UFA or RFA
    df['signing_status'] = df['signing_year'] >= df['ufa_year']
    #df['signing_status'] = df['signing_status'].apply(ufa_check)

    #eliminate duplicate rows with the same contract
    df = df.groupby('contract_id').head(1)

    #eliminate goalie contracts or tag them
    if goalies == False:
        df = df[df.position != 'Goaltender']
    else:
        df['skater'] = df.position != 'Goaltender'

    return df

def clean_contracts_data2(file = None, goalies = False):
    #load in dataframe from csv
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv('../data/Contract_Details_PuckPedia_Mar_2019_Confidential_2.csv')

    #Drop columns I won't be using
    df.drop(['buyout_id', 'base_salary', 'p_bonuses', 's_bonuses',
            'total_salary'], axis=1, inplace=True)

    #rename Total Value column to a format I prefer
    df['total_value'] = df[' value ']
    df.drop(' value ', axis=1, inplace=True)

    #create a column for full player name
    df['Player'] = df['first_name'] + ' ' + df['last_name']

    #set index to be contract_id
    df.set_index(df.contract_id, inplace=True)

    #drop column with missing cap hit/length (Alexander Radulov, contract no. 2132)
    df.drop(2132, inplace=True)

    #make a function to convert money strings to ints and apply
    money_to_int = lambda x: int(x.strip().strip('$()').replace(',', ''))
    df['cap_hit'] = df['cap_hit'].apply(money_to_int)
    df['total_value'] = df['total_value'].apply(money_to_int)

    #convert signing date and birthdate to date format and pull out the signing year
    df.signing_date = pd.to_datetime(df.signing_date)
    df.birthdate = pd.to_datetime(df.birthdate)
    df['signing_year'] = pd.DatetimeIndex(df.signing_date).year
    df.dropna(axis=0, subset=['signing_year'], inplace=True)

    #get age at contract signing_date (not accounting for leap days)
    df['signing_age'] = df.signing_date - df.birthdate
    df.signing_age = df.signing_age.apply(lambda x: x.days // 365)

    #convert signing_date to a date instead of a datetime
    df['signing_date'] = pd.DatetimeIndex(df['signing_date']).date

    #remove row that had NaN for signing year (Defenseman Fyodor Tyutin had
    #the contract listed in other rows also so no data is lost
#     df.drop(4236, axis=0, inplace=True)

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

    #Drop any contracts signed before 2009 since I have no stats
    df = df[df.signing_year > 2009]

    #Add a column for the total salary cap in a contract's year signed
    df['signing_year_cap'] = df['signing_year'].apply(lambda x: scap[x])
    # and a column for the percentage of the cap in the year signed
    df['cap_pct'] = round(100 * df.cap_hit / df.signing_year_cap, 2)

    #convert birthdate to pandas datetime
    df.birthdate = pd.to_datetime(df.birthdate)

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
    #df.drop(df[df.cap_hit < 500000].index, inplace=True)

    #Replace the position names for two players whose positions don't match the rest
    df.replace(['RW', 'LW'], ['Right Wing', 'Left Wing'], inplace=True)

    #add dummy variables for positions
    df['skater'] = df.position != 'Goaltender'
    df['forward'] = (df.position != 'Goaltender') & (df.position != 'Defense')
    df.replace([True, False], [1, 0], inplace=True)

    #drop entry level contracts
    df.drop(df[df.contract_level == 'entry_level'].index, inplace=True)

    #Add a column for whether the contract was signed as UFA or RFA
    df['signing_status'] = df['signing_year'] >= df['ufa_year']
    df['signing_status'] = df['signing_status'].apply(ufa_check)

    #eliminate duplicate rows with the same contract
    df = df.groupby('contract_id').head(1)

    #eliminate goalie contracts
    if goalies == False:
        df = df[df.position != 'Goaltender']

    return df

#auxillary function to check if contract was signed as UFA or RFA
def ufa_check(x):
    if x == True:
        return 'UFA'
    else:
        return 'RFA'

def ufa_to_binary(x):
    if x == 'UFA':
        return 1
    else:
        return 0


def clean_features_data(sql=True, new_fas = False):
    if sql:
        #read in player season total stats from SQL
        conn = pg2.connect(dbname='nhl', user='postgres', host='localhost', port='5435')
        cur = conn.cursor()
        query = '''
                SELECT *
                FROM pst;
                '''
        cur.execute(query)
        pst = list(cur)

        pst_cols = ['Season_Player', 'Player', 'Season', 'Position', 'GP', 'TOI', 'Goals',
                    'Total Assists', 'First Assists', 'Second Assists', 'Total Points',
                   'IPP', 'Shots', 'SH%', 'iCF', 'iFF', 'iSCF', 'iHDCF', 'Rush Attempts',
                   'Rebounds Created', 'PIM', 'Total Penalties', 'Minor', 'Major',
                   'Misconduct', 'Penalties Drawn', 'Giveaways', 'Takeaways', 'Hits',
                   'Hits Taken', 'Shots Blocked', 'Faceoffs Won', 'Faceoffs Lost',
                   'Faceoffs %'
                    ]

        df = pd.DataFrame(pst, columns = pst_cols)
    else:
        df = pd.read_csv('../data/up_all_pst.csv')

    dfsummable = df.drop(['SH%', 'Faceoffs %', 'IPP'], axis=1)

    summable_stats = ['GP', 'TOI', 'Goals',
                'Total Assists', 'First Assists', 'Second Assists', 'Total Points',
               'Shots', 'iCF', 'iFF', 'iSCF', 'iHDCF', 'Rush Attempts',
               'Rebounds Created', 'PIM', 'Total Penalties', 'Minor', 'Major',
               'Misconduct', 'Penalties Drawn', 'Giveaways', 'Takeaways', 'Hits',
               'Hits Taken', 'Shots Blocked', 'Faceoffs Won', 'Faceoffs Lost']

    #read in on_ice relative stats per season and over 3 year window
    oirel = pd.read_csv('../data/up_all_oirel.csv')
    woirel = pd.read_csv('../data/up_all_woirel.csv')

    #fix column labeling
    oirel['Season'] = oirel['Unnamed: 0.1']
    oirel.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
    woirel['Season'] = woirel['Unnamed: 0.1']
    woirel.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)

    #set indices to be season/player combo
    oirel['Season_Player'] = oirel.Season.apply(str)
    oirel.Season_Player = oirel.Season_Player + ' ' + oirel.Player
    woirel['Season_Player'] = woirel.Season.apply(str)
    woirel.Season_Player = woirel.Season_Player + ' ' + woirel.Player
    oirel.set_index(oirel.Season_Player, inplace=True)
    woirel.set_index(woirel.Season_Player, inplace=True)

    #read in target data to narrow down required player seasons
    if new_fas:
        contracts = clean_FA()
    else:
        contracts = clean_contracts_data()

    #set up default dictionaries to hold individual player stats
    allstats = defaultdict(pd.DataFrame)
    allsumstats = defaultdict(pd.DataFrame)
    allmeanstats = defaultdict(pd.DataFrame)

    #Give each player it's own data frame of stats all linked together in a dictionary with
    #player names as keys
    #Take season total stats and aggregate them over a 3 year window
    for player in contracts['Player'].unique():
        allstats[player] = dfsummable[df.Player == player]
        allstats[player].sort_values(by='Season', inplace=True)
        allstats[player]['Season_index'] = pd.date_range(end='2019',
                                                   periods = allstats[player].shape[0],
                                                   freq='Y')
        allstats[player].set_index(allstats[player].Season_index, inplace=True)
        allsumstats[player] = (allstats[player][summable_stats].rolling(window=3, min_periods=1)
                      .agg(np.sum))
        allmeanstats[player] = (allstats[player][summable_stats].rolling(window=3, min_periods=1)
                  .agg(np.mean))
        allmeanstats[player]['SH%'] = allsumstats[player]['Shots'] / allsumstats[player]['Goals']
        allmeanstats[player]['Faceoffs %'] = allsumstats[player]['Faceoffs Won'] / (
        allsumstats[player]['Faceoffs Won'] + allsumstats[player]['Faceoffs Lost'])

    scols, mcols = [], []
    #Relabeling sum and mean columns for clarity, arbitrarily using Pavelski for convenience,
    #could be any player
    for i in allsumstats['Joe Pavelski'].columns:
        scols.append('sum '+i)
    for i in allmeanstats['Joe Pavelski'].columns:
        mcols.append('mean '+i)
    for p in allstats:
        allsumstats[p].columns = scols
        allmeanstats[p].columns = mcols
        allstats[p] = pd.concat([allstats[p], allsumstats[p], allmeanstats[p]], axis=1)

    #get a combined dataframe with all relevant player/contract years
    players = list(allstats.keys())
    allallstats = pd.DataFrame()

    for p in players:
        allallstats = pd.concat([allallstats, allstats[p].set_index(allstats[p].Season_Player)])

    #drop duplicate column names
    m_oirel = oirel.drop(['Player', 'Team', 'Position', 'GP', 'TOI', 'Season'], axis=1)
    m_woirel = woirel.drop(['Player', 'Team', 'Position', 'GP', 'TOI', 'Season'], axis=1)

    #relabel 3yr window columns for clarity
    wcols = []
    for i in m_woirel.columns:
        wcols.append('3yr ' + i)
    m_woirel.columns = wcols

    #merge on ice relative stats to the other stats
    allallstats = pd.merge(allallstats, m_oirel, on = 'Season_Player')
    allallstats = pd.merge(allallstats, m_woirel, on = 'Season_Player')

    #remove contracts signed before 2010 (lack of stats)
    table = contracts[contracts.signing_year > 2009]#.set_index('contract_id')

    #add column to line up contract years and stats years
    table['year_match'] = table.signing_year - 1

    #merge contracts and stats into a single table
    table = pd.merge(table, allallstats,
                how = 'left', left_on = ['Player', 'year_match'],
                right_on = ['Player', 'Season'])

    #drop rows that had missing seasons / indexing issues leading to NaNs
    #1700 rows down to 1440
    table.dropna(thresh = 150, inplace=True)

    #Only take contracts signed after 2013 to eliminate survivorship bias in older
    #contracts signed before the last CBA.
    table = table[table.Season > 2013]


    #bring back IPP stat
    df = pd.merge(table, df[['Season_Player', 'IPP']], how='left', on = 'Season_Player')

    #drop duplicate Position column
    df.drop('Position', axis=1, inplace=True)

    #set index to be player/season
    df.set_index(df['Season_Player'], inplace=True)

    #extra feature engineering
    df['Giveaways/60'] = (df['Giveaways'] / df['TOI']) * 60
    df['mean Giveaways/60'] = (df['sum Giveaways'] / df['sum TOI']) * 60
    df['Takeaways/60'] = (df['Takeaways'] / df['TOI']) * 60
    df['mean Takeaways/60'] = (df['sum Takeaways'] / df['sum TOI']) * 60
    df['Shots Blocked/60'] = (df['Shots Blocked']/df['TOI']) * 60
    df['mean Shots Blocked/60'] = (df['sum Shots Blocked'] / df['sum TOI']) * 60
    df['mean Total Points/60'] = (df['sum Total Points']/df['sum TOI']) * 60
    df['Total Points/60'] = (df['Total Points']/df['TOI']) * 60
    df['Goals/60'] = (df['Goals']/df['TOI']) * 60
    df['mean Goals/60'] = (df['sum Goals']/df['sum TOI']) * 60
    df['Shots/60'] = (df['Shots']/df['TOI']) * 60
    df['mean Shots/60'] = (df['sum Shots']/df['sum TOI']) * 60
    df['Hits/60'] = (df['Hits']/df['TOI']) * 60
    df['mean Hits/60'] = (df['sum Hits']/df['sum TOI']) * 60
    df['PIM/60'] = (df['PIM']/df['TOI']) * 60
    df['mean PIM/60'] = (df['sum PIM']/df['sum TOI']) * 60
    df['Penalties Drawn/60'] = (df['Penalties Drawn']/df['TOI']) * 60
    df['mean Penalties Drawn/60'] = (df['sum Penalties Drawn']/df['sum TOI']) * 60
    df['mean Faceoffs pct'] = (df['sum Faceoffs Won'] / (df['sum Faceoffs Won'] + df['sum Faceoffs Lost']))
    df['Goalness'] = df['Goals']/(df['Total Points'] + 1)
    df['mean Goalness'] = df['sum Goals']/(df['sum Total Points'] + 1)

    df.drop(['3yr Season_Player', 'year_match', 'Team', 'Season', 'Season_Player', 'Season_index'], axis=1, inplace=True)

    df.rename({'3yr Off.\xa0Zone Starts/60': '3yr Off. Zone Starts/60',
                 '3yr Neu.\xa0Zone Starts/60': '3yr Neu. Zone Starts/60',
                 '3yr Def.\xa0Zone Starts/60': '3yr Def. Zone Starts/60',
                 '3yr On\xa0The\xa0Fly Starts/60': '3yr On The Fly Starts/60',
                 '3yr Off.\xa0Zone Start %': '3yr Off. Zone Start %',
                 '3yr Off.\xa0Zone Faceoffs/60': '3yr Off. Zone Faceoffs/60',
                 '3yr Neu.\xa0Zone Faceoffs/60': '3yr Neu. Zone Faceoffs/60',
                 '3yr Def.\xa0Zone Faceoffs/60': '3yr Def. Zone Faceoffs/60',
                 '3yr Off.\xa0Zone Faceoff %': '3yr Off. Zone Faceoff %',
                 'Off.\xa0Zone Starts/60': 'Off. Zone Starts/60',
                 'Neu.\xa0Zone Starts/60': 'Neu. Zone Starts/60',
                 'Def.\xa0Zone Starts/60': 'Def. Zone Starts/60',
                 'On\xa0The\xa0Fly Starts/60': 'On The Fly Starts/60',
                 'Off.\xa0Zone Start %': 'Off. Zone Start %',
                 'Off.\xa0Zone Faceoffs/60': 'Off Zone Faceoffs/60',
                 'Neu.\xa0Zone Faceoffs/60': 'Neu. Zone Faceoffs/60',
                 'Def.\xa0Zone Faceoffs/60': 'Def. Zone Faceoffs/60',
                 'Off.\xa0Zone Faceoff %': 'Off Zone Faceoff %',}
                 , axis = 1, inplace=True)

    return df
