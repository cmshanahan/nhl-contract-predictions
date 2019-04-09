## nhl-contract-predictions
### Predict NHL player contract terms (salary and length) from their in-game stats

This project looks at predicting NHL player salary cap hits and length of contract.

### Table of Contents:
 * [Background](#Background)
 * [Data](#Data)
 * [Modeling Choices](#Modeling-Choices)
 * [Alternate paths](#Alternate-paths)
 * [Conclusion](#Conclusion)
  - [Tools used](#tools-used)
  - [Thanks](#special-thanks)


## Background:
#### The NHL salary cap:
One of the defining features of most modern sports leagues (including the NHL) is a salary cap. The salary cap sets a hard limit on how much each team can spend on its players’ contracts in a given year. It enforces parity and keeps leagues competitive and interesting. Previously successful or independently wealthy teams cannot enrich themselves further by outspending poorer teams. This ensures wider appeal in the entire league, and for the most part, prevents the same few teams from constantly being in power.  
As such, managing a team’s contracts and salary cap space is extremely important for any team that wants to even pretend to be competitive. Most continually successful teams are very good at not overpaying their high-end talent and identifying key players that can be valuable without putting too big a dent in the team’s salary cap situation. Teams are always looking to get better for cheaper.
More data is available from [Wikipedia here](https://en.wikipedia.org/wiki/NHL_salary_cap).


## Data:
Contracts were obtained with permission from PuckPedia.com and included every player under an NHL contract in the 2017-2018 and 2016-2017 seasons. In all, I have 1460 contracts to work with.
Stats were downloaded in csv format from Natural Stat Trick.
I used Pandas rolling and aggregate functions to calculate average stats over the prior 3-year span.
The stats data was then merged with the contracts data so that every row contained a player contract and that player's stats over the season prior to signing and aggregated over 3 years prior to signing.
The raw data and the cleaned / featurized / merged data were then stored in SQL databases using a Postgres image on a Docker container.

Here's a chart illustrating some survivorship bias in my data:
<img src="images/Avg_cap_pct_over_time.png" alt="drawing" width="600"/>  
The average percentage of salary cap value of contracts increases as you go farther back in time since my data only contains active contracts for the last 2 seasons. The only contracts still active from those older years are for higher tier players.

Features that stood out:
 - Goals and Assists obviously had a significant positive correlation to salary
 - Penalties in general did not have a very visible effect, but major penalties had a strong negative effect (Skilled players tend to fight less)
 - Higher cap hit corresponds to higher cumulative stats (this makes sense because they play more)

## The Model:
The cleaned and compiled data was run through sklearn's Gradient Boosting Regressor algorithm to generate a predictive model.

### Modeling Choices:
* Since the salary cap changes inconsistent amounts from year to year (it has always moved upwards, but in theory it could shrink) predictions are made against the contract's percentage of the salary cap at year of signing.
  - The cap hit in real dollars was then calculated by referencing the listed salary cap for that year on [wikipedia](https://en.wikipedia.org/wiki/NHL_salary_cap).
* Stats were aggregated over a 3-year window because this is a student project and that was the available window for relative on-ice statistics from Natural Stat Trick without paying. While I could have aggregated cumulative stats over a larger window I chose not to for consistency.
* In the cases where a player did not have 3 years of NHL stats before signing his contract, I chose to fill in aggregates with stats over the available time window.
* Entry Level Contracts were excluded from my model's training set as I am only predicting standard level contracts once a player has time played in the NHL.
* Contracts signed before 2010 were excluded as Natural Stat Trick's data only goes back to 2007, thus there is no 3 year window.
* On top of that I decided to exclude all contracts signed before the last Collective Bargaining Agreement in 2013 to eliminate bias from contracts signed under a different set of rules.



### Special Thanks:
 * Thanks to PuckPedia for sharing their contracts database with me
 * Natural Stat Trick for having advanced stats data freely available for download
 * The instructors and my fellow classmates at Galvanize

### Tools used:  
 - Python  
 - sklearn  
 - pandas  
 - SQL  
 - NHL API  
 - Contract data from excel  
