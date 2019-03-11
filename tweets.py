import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def tweets_analysis(path, tags=[], bin_frequency='600s', stat_sign=0.6, print_corr=True):
    
    """
    Returns a pandas dataframe containing the count of tweets grouped into
        time interval bins (index) and hashtags (columns).
    
    Keyword arguments:
    path -- Path to the JSON file containing tweets.
    tags -- Hashtags to be included in the analysis.
    bin_frequency -- Interval of time into which tweets will be grouped.
    stat_sign -- Statistical significance threshold. As we are dealing with
        social media data and the use of hashtags, we will consider by
        default that any absolute correlation higher than 0.6 is significant.
    print_corr -- Print correlation analysis.
    """
    
    # Check types and formats
    assert os.path.isfile(path), 'Not a JSON file.'
    assert type(tags) == list, '"tags" must be a list'
    assert type(bin_frequency) == str, 'bin_frequency must be a string'
    assert bin_frequency[-1] == 's', 'bin_frequency must be in the format XXXs'
    assert bin_frequency[:-1].strip().isdigit(), 'bin_frequency must contain an integer'
    assert type(stat_sign) == int or type(stat_sign) == float, '"stat_sign" must be a number'
    assert stat_sign >= 0 and stat_sign <= 1, '"stat_sign" must be a float between 0 and 1'
    assert type(print_corr) == bool
    
    # Import JSON file
    try:
        df = pd.read_json(path)
    except:
        raise Exception('Error loading JSON file.')
    assert 'created_at' in df.columns and \
        'tag' in df.columns, \
        'JSON file must contain objects including "created_at" and "tag" keys.'
    
    # Check selected tags
    all_tags = sorted(df.tag.unique().tolist())
    if tags == []:
        tags = all_tags
    else:
        unknown_tags = [tag for tag in tags if tag not in all_tags]
        if unknown_tags:
            raise Exception('Unknown tags detected: {}'.format(unknown_tags))
            
    # Keep only the required tags to speed up the next steps
    df = df[df.tag.isin(tags)]
    
    # Prepare bins
    bin_frequency = int(bin_frequency.replace('s', ''))
    rounded_max_dt = df.created_at.max() + dt.timedelta(hours=1)
    rounded_max_dt = rounded_max_dt.replace(minute=0, second=0)
    min_dt = df.created_at.min()

    bin_dt = rounded_max_dt
    bins = [bin_dt]
    while bin_dt > min_dt:
        bin_dt -= dt.timedelta(seconds=bin_frequency)
        bins.append(bin_dt)
    bins = sorted(bins)
    
    # Replace created_at by bins
    if len(bins) == 1:
        # In case all the items can be grouped into a single bin
        df['created_at'] = bins[0]
    else:
        df['created_at'] = pd.cut(
            df.created_at,
            bins=bins,
            labels=bins[:-1],
            right=False,
        )
    
    # Aggregate by custom frequency bin
    df = df[['created_at', 'tag']]
    df['dummy'] = 1
    df = df.pivot_table(index='created_at', columns='tag', values='dummy', aggfunc='count')
    df = df[tags]
    df.sort_index(ascending=False, inplace=True)
    
    # Analyse correlation
    if print_corr:
        
        assert len(df) >= 2 and len(df.columns) >= 2, 'Not enough data to calculate correlations.'
        
        corrs = df.corr()

        tags_rs = []  # Store the tags and R values if significant
        # Iterate over the lower triangle of the correlation matrix
        ii, jj = np.tril_indices(corrs.shape[0], -1, corrs.shape[1])
        for i, j in zip(ii, jj):
            corr = corrs.iloc[i, j]
            if abs(corr) >= stat_sign:
                tags_rs.append([corrs.index[i], corrs.columns[j], corr])
        
        if len(df) < 25:
            'Warning! The correlation is being calculated on a low amount of data points: {}'.format(len(df))
            
        print('Statistical significance threshold: {}\n'.format(stat_sign))
        if tags_rs:
            print('Correlations found!')
            print(pd.DataFrame(tags_rs, columns=['Hashtag 1', 'Hashtag 2', 'Pearson R']))
        else:
            print('No significant correlation found.')
            
    return df

def predict_tweet_counts(path, tag, time, window_size, epochs=50):
    """
	Returns a pandas dataframe containing the count of tweets grouped into
        time interval bins (index) and predictions for the latest third of data.

    path: path to the JSON file.
    tag: selected tag for the analysis.
    time: time for aggregation.
    window_size: number of steps.
    epochs: epochs for model training.
    """
    
    raw_df = tweets_analysis(
        path,
        tags=[tag],
        bin_frequency=time,
        print_corr=False,
    )
    raw_df.sort_index(inplace=True)
    raw_df = raw_df.dropna()
    
    # Create df with linear timeline
    # Fill missing tweet counts
    min_time_gap = np.diff(raw_df.index)[0]
    min_time = raw_df.index.min()
    max_time = raw_df.index.max()

    times = []
    time = min_time
    while time < max_time:
        times.append(time)
        time += min_time_gap

    df = pd.DataFrame(index=times)
    df = df.join(raw_df, how='left')
    df.fillna(0, inplace=True)
    
    col = df.columns.tolist()[0]
    df[col] = df[col].astype(float)
    
    # Train-test split
    split_index = round((2/3)*len(df))
    train_data = df[df.columns[0]].tolist()[:split_index]
    test_data = df[df.columns[0]].tolist()[split_index:]
    
    # Normalize train data
    scaler = MinMaxScaler(feature_range = (0, 1))
    train_reshaped = np.array(train_data).reshape(len(train_data), 1)
    train_scaled = scaler.fit_transform(train_reshaped)
    
    # Build feature set
    feature_set = []
    labels = []
    for i in range(window_size, len(train_scaled)):
        feature_set.append(train_scaled[i-window_size:i, 0])
        labels.append(train_scaled[i, 0])
    
    feature_set, labels = np.array(feature_set), np.array(labels)
    feature_set = np.reshape(
        feature_set,
        (feature_set.shape[0], feature_set.shape[1], 1),
    )
    
    # Build model
    model = Sequential()
    
    model.add(LSTM(
        units=50,
        return_sequences=True,
        input_shape=(feature_set.shape[1], 1),
    ))
    model.add(Dropout(0.2))
    
    model.add(LSTM(
        units=50,
        return_sequences=True,
    ))
    model.add(Dropout(0.2))
    
    model.add(LSTM(
        units=50,
        return_sequences=True,
    ))
    model.add(Dropout(0.2))
    
    model.add(LSTM(
        units=50,
    ))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
    )
    
    # Train model
    model.fit(
        feature_set,
        labels,
        epochs=epochs,
        batch_size=32,
        verbose=1,
    )
    
    # Normalize test data
    full_test_data = train_data[-window_size:] + test_data
    full_test_reshaped = np.array(full_test_data).reshape(len(full_test_data), 1)
    full_test_scaled = scaler.transform(full_test_reshaped)
    
    # Build test feature set
    test_feature_set = []
    for i in range(window_size, len(full_test_scaled)):
        test_feature_set.append(full_test_scaled[i-window_size:i, 0])
    
    test_feature_set = np.array(test_feature_set)
    test_feature_set = np.reshape(
        test_feature_set,
        (test_feature_set.shape[0], test_feature_set.shape[1], 1),
    )
    
    # Predict
    predictions = model.predict(test_feature_set)
    predictions = scaler.inverse_transform(predictions)
    df['predicted'] = split_index*[None]+list(np.reshape(predictions, len(predictions), 1))
    
    print('RMSE of predictions:', np.sqrt(mean_squared_error(test_data, predictions)))
    
    #Plot actual data and predictions
    plt.figure(figsize=(10,6))
    plt.plot(df.index[split_index:], test_data, label='Actual data')
    plt.plot(df.index[split_index:], predictions , label='Predicted data')
    plt.title('Prediction - {}'.format(tag))
    plt.xlabel('Date')
    plt.ylabel('Number of tweets')
    plt.legend()
    plt.show()
    
    # Return sorted df
    df.sort_index(ascending=False, inplace=True)
    
    return df
