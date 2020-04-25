import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from haversine import haversine
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from catboost import CatBoostRegressor
from catboost import cv
from catboost import Pool

np.random.seed(0)


def train_pca(df):
    coords = np.vstack((df[['latitude', 'longitude']].values,
                        df[['del_latitude', 'del_longitude']].values))

    pca = PCA(random_state=0)
    pca.fit(coords)

    return pca


def clusterize(df, n_clusters=100, batch_size=10000, sample_size=500000):
    coords = np.vstack((df[['latitude', 'longitude']].values,
                        df[['del_latitude', 'del_longitude']].values))

    sample_ind = np.random.permutation(len(coords))[:sample_size]
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=0).fit(coords[sample_ind])

    return kmeans


def create_features(df, features_to_use, pca, kmeans, train=False):
    # time features
    df['OrderedDate_datetime'] = pd.to_datetime(df['OrderedDate'])
    df['month'] = df['OrderedDate_datetime'].dt.month
    df['hour'] = df['OrderedDate_datetime'].dt.hour
    df['week_of_year'] = df['OrderedDate_datetime'].dt.weekofyear
    df['day_of_year'] = df['OrderedDate_datetime'].dt.dayofyear
    df['day_of_week'] = df['OrderedDate_datetime'].dt.dayofweek

    # geo features
    df['haversine'] = df.apply(lambda row: haversine((row['latitude'], row['longitude']),
                                                     (row['del_latitude'], row['del_longitude'])), axis=1)

    # maneuvers
    #     df['n_turns'] = df['step_maneuvers'].apply(lambda s: Counter(s.split('|'))['turn'])

    #     df['n_left_directions'] = df['step_direction'].apply(lambda s: Counter(s.split('|'))['left'])
    #     df['n_right_directions'] = df['step_direction'].apply(lambda s: Counter(s.split('|'))['right'])

    # PCA features
    pickup_pca_features = pca.transform(df[['latitude', 'longitude']])
    df['pickup_pca0'] = pickup_pca_features[:, 0]
    df['pickup_pca1'] = pickup_pca_features[:, 1]

    dropoff_pca_features = pca.transform(df[['del_latitude', 'del_longitude']])
    df['dropoff_pca0'] = dropoff_pca_features[:, 0]
    df['dropoff_pca1'] = dropoff_pca_features[:, 1]

    # kmeans features
    df['pickup_cluster'] = kmeans.predict(df[['latitude', 'longitude']])
    df['dropoff_cluster'] = kmeans.predict(df[['del_latitude', 'del_longitude']])

    if train:
        return df[features_to_use + ['RTA']]

    return df[features_to_use]


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def train(df):
    X = df.drop('RTA', axis=1)
    y = np.log(df['RTA'])
    categorical_features_indicies = [features_to_use.index(feat) for feat in categorical_features]

    model = CatBoostRegressor(task_type="CPU", loss_function='MAPE', random_seed=0)
    model.fit(
        X, y,
        cat_features=categorical_features_indicies,
        early_stopping_rounds=10,
        verbose=True,
        plot=True
    )

    importances = model.get_feature_importance(prettified=True)
    print('Important features\n', importances)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--train_val_merge', type=int, default=1)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file)
    print('Train loaded')
    val_df = pd.read_csv(args.val_file)
    print('Validation loaded')
    test_df = pd.read_csv(args.test_file)
    print('Test loaded')

    if args.train_val_merge:
        train_df = pd.concat([train_df, val_df], sort=False)

    print('train_df shape: ', train_df.shape[0])

    # Train stage
    pca = train_pca(train_df)
    kmeans = clusterize(train_df)

    features_to_use = [
        'main_id_locality',
        'ETA', 'month',
        'hour',
        'week_of_year',
        'day_of_year',
        'day_of_week',
        'haversine',
        'pickup_pca0',
        'pickup_pca1',
        'dropoff_pca0',
        'dropoff_pca1',
        'pickup_cluster',
        'dropoff_cluster'
    ]

    categorical_features = [
        'main_id_locality',
        'month',
        'hour',
        'week_of_year',
        'day_of_week',
        'pickup_cluster',
        'dropoff_cluster'
    ]

    train_df = create_features(train_df, features_to_use, pca, kmeans, True)
    print('Train DataFrame: \n', train_df.head())

    model = train(train_df)

    if not args.train_val_merge:
        val_df = create_features(val_df, features_to_use, pca, kmeans, True)
        val_df['predict'] = np.exp(model.predict(val_df))
        mape = mean_absolute_percentage_error(val_df['ETA'], val_df['predict'])
        print('Validation MAPE: ', mape)

    # Test stage
    test_df = create_features(test_df, features_to_use, pca, kmeans, False)
    test_df['predict'] = np.exp(model.predict(test_df))

    test_df = test_df.reset_index()
    test_df = test_df.rename(columns={'index': 'Id', 'predict': 'Prediction'})
    test_df[['Id', 'Prediction']].to_csv('/app/submission/baseline.csv', sep=',', index=False, header=True)

    print(test_df[['Id', 'Prediction']].head())
