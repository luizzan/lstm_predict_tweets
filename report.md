# Analysis of Tweets Correlation and Prediction - Summary of findings

1. Function `tweets_analysis` in `tweets.py` loads tweets from `tweets.json`.

1. Reshaping of data OK:
	1. Columns OK.
	1. Index OK.
	1. Cell content OK.
	1. Correlation calculation OK.
	1. Correlation is only performed and printed if enough data is available: at least two bins and two columns. If the number of bins is lower than 25, a warning is printed regarding calculations on low amount of data. The data is presented as a table containing two columns for the pair of tags analysed and one for the correlation coefficient between them. This table is only printed if there is at least one pair of tags with correlation higher than the specified threshold.

1. Program extended to enable custom bin frequency.

1. Tests can be run from `tests.py`. They test possible input errors in the `tweets_analysis` function: format of parameters, availability of bins for the calculation of correlation, etc. In a real application further tests could be added for which the function is provided a known dataset and its outputs are compared to verified results.

1. This step was part of the Data Engineer application, to which I was not applying. However, I have previously developed a module for tweet scraping. Its output is either to return a dataframe or to save a local CSV file. Both can be easily transformed into the JSON format required as the input of `tweets_analysis`. Code available here: [tweetdigger](https://github.com/luizzan/tweetdigger)

1. The model for tweet prediction given a hashtag was created in the function `predict_tweet_counts` in `tweets.py`. An LSTM model was developed in Keras. For a given hashtag, it splits the available tweet counts into two thirds for training and one third for testing. The method used was to, provided a window size, create a feature set containing labeled sequences for training. The predictions are performed one step at a time, providing the most recent data available, with the same window size. The evaluation of results is performed via RMSE. A dataframe containing datetimes, the actual number of tweets and the predicted tweet counts is returned. The data provided is scarce, so model accuracy is not optimal. In a real application the model could be automatically optimized as more data is made available, tuning window size, epochs and other model parameters.