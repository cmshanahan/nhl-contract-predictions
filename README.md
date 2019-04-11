## nhl-contract-predictions
### Predict NHL player contract terms (salary and length) from their in-game stats

This project looks at predicting two targets, NHL player salary cap hits and length of contract.

### Table of Contents:
 * [Background](#Background)
 * [Data](#Data)
 * [Model](#the-model)
    - [Modeling Choices](#modeling-choices)
    - [Error Metric and Baseline](#error-and-baseline)
    - [Clustering](#clustering)
    - [Nearest Neighbors Regressors](#nearest-neighbors-regressors)
    - [Gradient Boosting Regressor](#gradient-boosting-regressor)
 * [Conclusion](#conclusion)
    - [Important Features](#important-features)
    - [Results](#results)
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
The stats data was then merged with the contracts data so that every row contained a player contract and that player's stats over the season prior to signing and aggregated over 3 years prior to signing. After all of this, my data contained roughly 200 columns.
The raw data and the cleaned / featurized / merged data were then stored in SQL databases using a Postgres image on a Docker container.  
One major challenge in dealing with the data in this problem was the relatively low sample size. This resulted in high variance models, making it difficult to evaluate the impact of individual changes.

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

### Selecting an Error Metric and a Baseline
To evaluate my model I chose Root Mean Squared Error (RMSE) due to its interpretability and applicability to regression problems. One main advantage of RMSE over some other error metrics is that it can be expressed in the same units as our targets, dollars and years.  

### kMeans Clustering:
One notion I had going into this project was that there are different types of players who would have different stats valued differently when it comes to contract negotiations. I hypothesized that these inherent players groups could be separated and a more accurate model could be achieved by running separate linear models on each cluster independently.  
I ultimately had to reject this hypothesis as I found no method of clustering the players that resulted in cleanly separable groups. Running independent models on these clusters did no better than running a global non-parametric model. In fact by further segmenting my already small dataset, the variance problem became even worse.  
Another factor reducing the effectiveness of clustering was the high dimensionality of the data, which often made computed distances end up being completely arbitrary. I tried selecting features that I thought would well-define player usage (such as Offensive Zone Start % and TOI/GP) with mediocre results. Objective dimensionality reduction using Principal Component Analysis (PCA) did not help either.

<img src="images/intuit_clusters.png" alt="drawing" width="600"/>
<img src="images/pca_clusters.png" alt="drawing" width="600"/>

### Nearest Neighbors Regressors
Nearest neighbors regressors were evaluated as another possible metric for evaluating salaries (k Nearest Neighbors and Radius Neighbors). In fact this was how I believed salaries were evaluated going into this process. However as encountered before with clustering, due to the high dimensionality of the data, a player's "nearest neighbors" would often have little to do with him in the way of actually meaningful performance statistics, or would often have few to none neighbors in a predetermined "radius". This made for wildly inconsistent results when it came to a predictive model.

### Gradient Boosting Regressor:
I tried several different regression models and found that ensembled decision tree based models such as Random Forest Regression and Gradient Boosting Regression consistently outperformed both linear models and neighbor based models. Gradient Boosting Regression proved to ultimately be the most consistent algorithm and usually outperformed the other models. The cleaned and compiled data was run through sklearn's Gradient Boosting Regressor algorithm to generate a predictive model.   
One thing that made the decision tree models so much more consistently effective than neighbors regressors is their independence from distance metrics. Their ability to handle unique situations and non-linearities in the data trends was also incredibly valuable.


## Conclusions

### Permutation Importance
I calculated feature importances using the Random Forest Permutation Importance (RFPimp) module. The permutation importance of a feature is calculated as the change in model score that arises from randomly scrambling the values for that feature while holding all others the same.

#### Important Salary Features
* Total Points (1 year)
* Time on Ice (TOI) (1 year total)
* TOI / Game (3 year mean)
* TOI / Game (1 year)
* iCF (Player shot attempts - 1 year)

#### Important Length Features
* Predicted Cap %
* Player Age

### Results:
RMSE pick mean cap_pct: 2.9%
    translates to 2019 Cap Hit of: $2,407,000
RMSE pick mean length: 1.9 years

RMSE Cap_pct: 0.97% (**Improvement: 67%**)
    translates to 2019 Cap Hit of: $805,100
RMSE Length: 1.0 years (**Improvement: 47%**)

RMSE for predicting mean Total Value: $17,899,000
RMSE for Gradient Boosted Model Total Value: $7,523,000
**Improvement: 58.0%**

<img src="images/free_agent_2019_preds.png" alt="drawing" width="600"/>

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

## About the Author:  
Colin Shanahan is a San Francisco based former mechanical engineer turned Data Scientist and a student at Galvanize's Data Science Immersive program. He is also a huge hockey fan.  
You can reach him on [LinkedIn](https://www.linkedin.com/in/c-shan).
