# Analysis of Tweets Correlation and Prediction - Instructions

In tweets.py write a simple program to:

1. Load 8000 Twitter tweets from tweets.json downloaded from Twitter API.

1. Shape data into Pandas.DataFrame so that:
	1. Pandas.DataFrame columns will contain tweets' tag attribute (i.e. ["#ai", "#analytics", "conference", "acquisition"]);
	1. Pandas.DataFrame index will contain tweets' created_at binned into 10-minutes buckets (i.e. ["2019-02-11 11:00:00", "2019-02-11 10:50:00"]) sorted in descending order;
	1. Pandas.DataFrame cells will contain count of tweets tagged with particular tag (column) at particular created_at bin (index);
	1. Performs correlation analysis between all pairs of tags;
	1. Determine and provide reasoning if any significant correlation between tags was observed.

1. Extend program to enable custom bin frequency.

1. Write tests.

1. Get the data for step 1 yourself: Write a code that loads 8000 consecutive Twitter tweets starting from 2019-02-01 16:30:00 with tags:
	1. #acquisition
	1. #ai
	1. #analytics
	1. #bigdata
	1. #conference
	1. #meetup
	1. #ml
	1. #startup

1. Split the dataset into thirds (ordered by time). Use first two thirds to train a model predicting counts of tweets given a tag (from those present in the data) and a time bin. Evaluate model performance using last third of data, discuss results.