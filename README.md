# nhl-contract-predictions
Predict NHL player contract terms - Galvanize Data Science Immersive Capstone Project


# Day 1:  
Cleaned contract data obtained via excel sheet from PuckPedia.
Made assumption of $83 million salary cap for 2019 season based on Sportsnet article

# Day 2:  
Forget everything else you were going to use for stats sources, just use Natural Stat Trick! They provide CSVs!
For now I'm going to limit stats to 'All Situations' for simplicity. Next steps will hopefully involve adding power play / penalty kill.
I got an extremely dumb linear regression model based solely on age and player position. The root mean squared error (RMSE) was 3%, which is also what the standard deviation of my test set was, so yeah, not a great model. But it's something.
Now I need to work on a way to combine the rest of my features and my targets.

# Day 3 (or How I learned to stop worrying and love Luongo):  
I think the best route for me to go is to implement sklearn's Pipeline somehow. I will also need to have some sort of pandas rolling window setup. I have base data for every player / season combo, and for some of the relative stats I have 3 year windows already setup. A lot of the players in my stats data aren't going to show up anywhere in my contracts data.
I want to use previous contract salary as a feature. If a player has no previous contracts in the data, then assume he was on an ELC (I'm not going to be predicting ELCs.) Median ELC cap_hit in the dataset is $792,500.
I think we'll ignore total career stats for now as the data I have only goes back to 2007 and that won't cover everyone's entire career.
-- Maybe what I'm really after is a RadiusNeighborsRegressor not kNNregressor
Hey I got df.rolling to work!

# Day 4:  
Merged the tables together! I now have a row for every player's contract, with their stats for the year prior to signing and the 3 year window before it in the same row.
I'm having some problems with players missing seasons creating NaNs in my merge. 300 rows affected, I'll come up with a fix later, but for now I've dropped them. Ran a basic linear model on signing age, position (F or D), points in last season, mean points in last 3 seasons, sum TOI over 3 seasons. RMSE is 1.38% of cap_pct.
On running the same model for both cap_pct and cap_hit (including signing_year_cap as a feature) apparently it predicts cap_hit better than cap_pct. I wonder if this means cap_pct isn't staying consistent, or maybe because the vast majority of the contracts are only over a small time window.
Random Forest Regressor has a RMSE of ~$750,000.
RadiusNeighborsRegressor w/ radius = 0.67 (using sklearn's StandardScaler), gives a RMSE of $665,000. -> Also RMSE of 0.88% for cap_pct.
I should run Grid Search on RNR.
If I drop radius to 0.25, the RMSE drops to $43,000, but I don't know if I trust that result.
In fact, when I remove the constant random state from my train test split, a lot of my answers seem to end up all over the place (mostly at lower radii). A radius of ~2/3 might be the sweet spot where it stays relatively consistent
I think I need to run some Ridge or Lasso regression to help w/ feature selection. I also want to keep forward / defense models separate.
Even more than that... I should try and get around to that player clustering I was thinking about and implement different models for each cluster.
Radius Neighbors Regressor gives inconsistent results between uniform weights and distance weights.
