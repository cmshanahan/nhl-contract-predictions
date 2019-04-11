## nhl-contract-predictions
### Predict NHL player contract terms (salary and length) from their in-game stats

This project looks at predicting two targets, NHL player salary cap hits and length of contract.

### Table of Contents:
 * [Background](#Background)
 * [Data](#Data)
 * [Model](#the-model)
    - [Clustering](#clustering)
    - [Gradient Boosting Regressor](#gradient-boosting-regressor)
    - [Modeling Choices](#modeling-choices)
    - [Important Features](#important-features)
 * [Results](#results)
 * [Conclusion](#conclusion)
    - [Tools used](#tools-used)
    - [Special Thanks](#special-thanks)


## Background:
#### The NHL salary cap:
One of the defining features of most modern sports leagues (including the NHL) is a salary cap. The salary cap sets a hard limit on how much each team can spend on its players’ contracts in a given year. It enforces parity and keeps leagues competitive and interesting. Previously successful or independently wealthy teams cannot enrich themselves further by outspending poorer teams. This ensures wider appeal in the entire league, and for the most part, prevents the same few teams from constantly being in power.  
As such, managing a team’s contracts and salary cap space is extremely important for any team that wants to even pretend to be competitive. Most continually successful teams are very good at not overpaying their high-end talent and identifying key players that can be valuable without putting too big a dent in the team’s salary cap situation. Teams are always looking to get better for cheaper.
More data on the salary cap is available from [Wikipedia here](https://en.wikipedia.org/wiki/NHL_salary_cap).


## Data:
Contracts were obtained with permission from [PuckPedia.com](https://puckpedia.com/) and included every player under an active NHL contract in the 2017-2018 and 2016-2017 seasons. In all, I had ~1200 contracts to work with once I eliminated goalies, entry level players, and those who signed contracts under the previous collective bargaining agreement.
Stats were downloaded in csv format from Natural Stat Trick.
I used Pandas rolling and aggregate functions to calculate average stats over the prior 3-year span.
The stats data was then merged with the contracts data so that every row contained a player contract and that player's stats over the season prior to signing and aggregated over 3 years prior to signing.
The raw data and the cleaned / featurized / merged data were then stored in SQL databases using a Postgres image on a Docker container.

<img src="images/cap_length_box.png" alt="drawing" width="600"/>

Here's a chart illustrating some survivorship bias in my data:
<img src="images/Avg_cap_pct_over_time.png" alt="drawing" width="600"/>  
The average percentage of salary cap value of contracts increases as you go farther back in time since my data only contains active contracts for the last 2 seasons. The only contracts still active from those older years are for higher tier players.

Features and trends that stood out:
 - Goals and Assists had a clear and significant positive correlation to salary
 - Penalties in general did not have a very visible effect, but major penalties had a strong negative effect (Skilled players tend to fight less).
 - Higher cap hit corresponds to higher cumulative stats (this makes sense because they play more).
 - Older players tend to sign shorter contracts, as do younger players with lower stats.
 - Elite players who are younger or in their prime usually get signed to max length contracts. This makes sense as the team gets to lock up the player's talent long term, and the player gets financial security.
 - The data distribution is clearly weighted towards many contracts of lower salary and shorter length, but averages are brought up by the best players being paid significantly more.

<img src="images/sal_hist.png" alt="drawing" width="600"/>
<img src="images/len_hist.png" alt="drawing" width="600"/>


## Model:


### Selecting an Error Metric and a Baseline
To evaluate my model I chose Root Mean Squared Error (RMSE) due to its interpretability and applicability to regression problems. One main advantage of RMSE over some other error metrics is that it can be expressed in the same units as our targets, dollars and years.  
<RMSE =  \sqrt{\frac{1}{n} \ast   \sum (prediction - target)^{2}}>

### kMeans Clustering:
One notion I had going into this project was that there are different types of players who would have different stats valued differently when it comes to contract negotiations. I hypothesized that
Cluster plots here

### Gradient Boosting Regressor:
The cleaned and compiled data was run through sklearn's Gradient Boosting Regressor algorithm to generate a predictive model. I tried several different regression models but ultimately found that Gradient Boosting provided the best and most consistent scores.


### Modeling Choices:
* Since the salary cap changes inconsistent amounts from year to year (it has always moved upwards, but in theory it could shrink) predictions are made against the contract's percentage of the salary cap at year of signing.
  - The cap hit in real dollars was then calculated by referencing the listed salary cap for that year on [wikipedia](https://en.wikipedia.org/wiki/NHL_salary_cap).
* Stats were aggregated over a 3-year window because this is a student project and that was the available window for relative on-ice statistics from Natural Stat Trick without paying. While I could have aggregated cumulative stats over a larger window I chose not to for consistency.
* In the cases where a player did not have 3 years of NHL stats before signing his contract, I chose to fill in aggregates with stats over the available time window.
* Entry Level Contracts were excluded from my model's training set as I am only predicting standard level contracts once a player has time played in the NHL.
* Contracts signed before 2010 were excluded as Natural Stat Trick's data only goes back to 2007, thus there is no 3 year window.
* On top of that I decided to exclude all contracts signed before the last Collective Bargaining Agreement in 2013 to eliminate bias from contracts signed under a different set of rules.

<img src="images/cap_ht_scat.png" alt="drawing" width="600"/>
<img src="images/len_ht_scat.png" alt="drawing" width="600"/>

## Important Features

### Permutation importance
I calculated feature importances using the Random Forest Permutation Importance (RFPimp) module. The permutation importance of a feature is calculated as the change in model score that arises from randomly scrambling the values for that feature while holding all others the same.

### Salary Features
* Total Points (1 year)
* Time on Ice (TOI) (1 year total)
* TOI / Game (3 year mean)
* TOI / Game (1 year)
* iCF (Player shot attempts - 1 year)

### Length Features
* Predicted Cap %
* Player Age

## Results:
RMSE pick mean cap_pct: 2.9%
    translates to 2019 Cap Hit of: $2,407,000
RMSE pick mean length: 1.9 years

RMSE Cap_pct: 0.97% (**Improvement: 67%**)
    translates to 2019 Cap Hit of: $805,100
RMSE Length: 1.0 years (**Improvement: 47%**)

RMSE for predicting mean Total Value: $17,899,000
RMSE for Gradient Boosted Model Total Value: $7,523,000
**Improvement: 58.0%**

### Tools and Resources used:  
 - Python
 - sklearn  
 - pandas  
 - matplotlib
 - seaborn
 - Tableau
 - SQL
 - PuckPedia (contract data)
 - Natural Stat Trick (statistics data)
 - NHL API  

### Special Thanks:
 * Thanks to PuckPedia for sharing their contracts database with me
 * Natural Stat Trick for having advanced stats data freely available for download
 * The instructors and my fellow classmates at Galvanize
