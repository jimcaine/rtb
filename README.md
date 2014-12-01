rtb
===
Jim Caine - November, 2014 - DePaul University - Computational Advertising

A project written in November, 2014.  Historical RTB data published by iPinYou is leveraged to explore RTB bidding algorithms.  Machine learning algorithms (logistic regression, decision tree classification, LDA, naive-bayes) are used to predict the propensity of a click for any given impression.  The model (using logistic regression) is then tested by simulating an ad campaign by bidding arbitrary cost per click (CPC) goal values multiplied by the propensity to click given by the model.  The utility of the model is evaluated by comparing actual CPC values for similar budgets.  The bidding algorithm is then extended to handle fixed budgets by using a Monte Carlo resampling procedure to dynamically change the bid urgency (scaling factor of the bid, similar to goal) throughout the campaign.

Python with sci-kit learn and pandas is used extensively for the computation in rtb.py.