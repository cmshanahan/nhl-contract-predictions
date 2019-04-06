## nhl-contract-predictions
### Predict NHL player contract terms (salary and length) from their in-game stats

This project looks at predicting NHL player salary cap hits and length of contract.

### Table of Contents:



## Background:
The NHL salary cap....
Since the salary cap changes inconsistent amounts from year to year (it has always moved upwards, but in theory it could shrink) predictions are made against the contract's percentage of the salary cap at year of signing.

##Data:
Contracts were obtained with permission from PuckPedia.com and included every player under an NHL contract in the 2017-2018 and 2016-2017 seasons.
Stats were downloaded in csv format from Natural Stat Trick.
I used Pandas rolling and aggregate functions to calculate average stats over the prior 3-year span.
The stats data was then merged with the contracts data so that every row contained a player contract and that player's stats over the season prior to signing and aggregated over 3 years prior to signing.
The raw data and the cleaned / featurized / merged data were then stored in SQL databases using a Postgres image on a Docker container.

Here's a chart illustrating some survivorship bias in my data:
<img src="images/Avg_cap_pct_over_time.png" alt="drawing" width="1000"/>  
The average percentage of salary cap value of contracts increases as you go farther back in time since my data only contains active contracts for the last 2 seasons. The only contracts still active from those older years are for higher tier players.

Features that stood out:
 - Goals and Assists obviously had a significant positive correlation to salary
 - Penalties in general did not have a very visible effect, but major penalties had a strong negative effect (Skilled players tend to fight less)



### Special Thanks:
 * Thanks to PuckPedia for sharing their contracts database with me
 * Natural Stat Trick for having advanced stats data freely available for download
 * The instructors and my fellow classmates at Galvanize
