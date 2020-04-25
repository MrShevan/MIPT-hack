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
    df['day_of_week'] = df['OrderedDate_datetime'].dt.dayofweek
    df['week_of_year'] = df['OrderedDate_datetime'].dt.weekofyear
    df['day_of_year'] = df['OrderedDate_datetime'].dt.dayofyear

    # geo features
    df['haversine'] = df.apply(lambda row: haversine((row['latitude'], row['longitude']),
                                                     (row['del_latitude'], row['del_longitude'])), axis=1)

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

def load_data(mode, path):
    '''
    mode: {'train', 'val', 'test'}
    path: path to data file
    '''
    train_df = pd.read_csv(path)
    train_df_ext = pd.read_csv(f'../data/{mode}_extended.csv')
    train_df = pd.concat([train_df, train_df_ext], axis=1)
    print(f'{mode} loaded')
    return train_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--separate_cities', type=int, default=0)

    parser.add_argument('--train_val_merge', type=int, default=1)
    args = parser.parse_args()

    train_df = load_data('train', args.train_file)
    val_df = load_data('val', args.val_file)
    test_df = load_data('test', args.test_file)

    if args.train_val_merge:
        train_df = pd.concat([train_df, val_df], sort=False)

    print('train_df shape: ', train_df.shape[0])

    # Train stage
    pca = train_pca(train_df)
    kmeans = clusterize(train_df)

    features_to_use = [
        'main_id_locality',
        'ETA',
        'hour',
        'month',
        'day_of_week',
        'week_of_year',
        'day_of_year',
        'haversine',
        'pickup_pca0',
        'pickup_pca1',
        'dropoff_pca0',
        'dropoff_pca1',
        'pickup_cluster',
        'dropoff_cluster',
        'start_offset',
        'finish_offset',
        'koeff_overroute',
        'parts_count',
        'parts_distance_sum',
        'parts_distance_avg'
    ]

    categorical_features = [
        'main_id_locality',
        'month',
        'hour',
        'week_of_year', 'day_of_week',
        'pickup_cluster',
        'dropoff_cluster'
    ]

    # koeff = (train_df["RTA"].sum() + train_df["RTA"].shape[0]) / train_df["ETA"].sum()

    train_df = create_features(train_df, features_to_use, pca, kmeans, True)
    # train_df["ETA"] = koeff * train_df["ETA"]
    print('Train DataFrame: \n', train_df.head())

    if args.separate_cities:
        models = {}
        for main_id_locality in train_df["main_id_locality"].unique():
            city_df = train_df[train_df["main_id_locality"] == main_id_locality]
            model = train(city_df)
            models[main_id_locality] = model

        if not args.train_val_merge:
            val_df = create_features(val_df, features_to_use, pca, kmeans, True)
            # val_df["ETA"] = koeff * val_df["ETA"]

            predicts = []
            for main_id_locality in val_df["main_id_locality"].unique():
                city_df = val_df[val_df["main_id_locality"] == main_id_locality]
                city_df['predict'] = np.exp(models[main_id_locality].predict(city_df))
                predicts.append(city_df)

            val_df = pd.concat(predicts, axis=0)

            mape = mean_absolute_percentage_error(val_df['RTA'], val_df['predict'])
            print('Validation MAPE: ', mape)

    else:

        model = train(train_df)

        if not args.train_val_merge:
            val_df = create_features(val_df, features_to_use, pca, kmeans, True)
            # val_df["ETA"] = koeff * val_df["ETA"]

            val_df['predict'] = np.exp(model.predict(val_df))
            mape = mean_absolute_percentage_error(val_df['RTA'], val_df['predict'])
            print('Validation MAPE: ', mape)

        # Test stage
        test_df = create_features(test_df, features_to_use, pca, kmeans, False)
        # test_df["ETA"] = koeff * test_df["ETA"]

        test_df['predict'] = np.exp(model.predict(test_df))

        test_df = test_df.reset_index()
        test_df = test_df.rename(columns={'index': 'Id', 'predict': 'Prediction'})
        test_df[['Id', 'Prediction']].to_csv('submission/submission_time.csv', sep=',', index=False, header=True)

        print(test_df[['Id', 'Prediction']].head())
