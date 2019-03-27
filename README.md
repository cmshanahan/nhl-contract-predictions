# nhl-contract-predictions
Predict NHL player contract terms - Galvanize Data Science Immersive Capstone Project


Day 1:
Cleaned contract data obtained via excel sheet from PuckPedia.
Made assumption of $83 million salary cap for 2019 season based on Sportsnet article

Day 2:
Forget everything else you were going to use for stats sources, just use Natural Stat Trick! They provide CSVs!
For now I'm going to limit stats to 'All Situations' for simplicity. Next steps will hopefully involve adding power play / penalty kill.
I got an extremely dumb linear regression model based solely on age and player position. The root mean squared error (RMSE) was 3%, which is also what the standard deviation of my test set was, so yeah, not a great model. But it's something.
Now I need to work on a way to combine the rest of my features and my targets.

Day 3 (or How I learned to stop worrying and love Luongo):
