import argparse
import pandas as pd
import polyline
from tqdm import tqdm
from haversine import haversine


def get_route_features(row):
    features = {}

    try:
        parts = polyline.decode(row["route"])

        parts_count = len(parts)
        parts_distance_sum = 0

        for i in range(0, len(parts) - 1):
            parts_distance_sum += haversine(parts[i], parts[i + 1])

        features["parts_count"] = parts_count
        features["parts_distance_sum"] = parts_distance_sum
        features["parts_distance_avg"] = parts_distance_sum / parts_count

    except:
        features["parts_count"] = None
        features["parts_distance_sum"] = None
        features["parts_distance_avg"] = None

    return features


def get_new_features(train_df):
    # Геогрфическое расстояние от начальной до конечной точки
    train_df.loc[:, 'haversine'] = train_df.apply(lambda row: haversine((row['latitude'], row['longitude']),
                                                                        (row['del_latitude'], row['del_longitude'])),
                                                  axis=1)

    # Удаленность от центра города начала маршрута
    train_df.loc[:, 'start_offset'] = train_df.apply(lambda row: haversine((row['latitude'], row['longitude']),
                                                                           (row['center_latitude'], row['center_longitude'])), axis=1)

    # Удаленность от центра города конца маршрута
    train_df.loc[:, 'finish_offset'] = train_df.apply(lambda row: haversine((row['del_latitude'], row['del_longitude']),
                                                                            (row['center_latitude'], row['center_longitude'])), axis=1)

    # Коэффициент перепробега (отношение реальной длины к расстоянию между точками)
    train_df.loc[:, "koeff_overroute"] = train_df.apply(lambda row: row['EDA'] / row['haversine'], axis=1)

    train_route_features = []
    for idx, row in tqdm(train_df.iterrows()):
        train_route_features.append(get_route_features(row))

    train_route_features_df = pd.DataFrame(train_route_features)

    values = dict(train_route_features_df.mean())
    train_route_features_df = train_route_features_df.fillna(value=values)

    train_route_features_df.parts_count = train_route_features_df.parts_count.astype(int)
    train_route_features_df.parts_distance_sum = train_route_features_df.parts_distance_sum.astype(float)
    train_route_features_df.parts_distance_avg = train_route_features_df.parts_distance_avg.astype(float)

    train_routes_df = pd.concat([train_df, train_route_features_df], axis=1)

    train_routes_df["koeff_overroute_dist"] = train_routes_df["parts_distance_sum"] / train_routes_df["haversine"]
    train_routes_df["koeff_overroute_rel"] = train_routes_df["EDA"] / train_routes_df["parts_distance_sum"]

    feature_names = ["start_offset", "finish_offset", "koeff_overroute", "parts_count",
                     "parts_distance_sum", "parts_distance_avg", "koeff_overroute_dist",
                     "koeff_overroute_rel"]

    new_features_df = train_routes_df[feature_names]

    return new_features_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--test_add_file', type=str, required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file)
    print('Train loaded')
    print('Train df get features ...')
    train_df_extended = get_new_features(train_df)
    train_df_extended.to_csv("/data/train_extended.csv", index=False)
    print()

    val_df = pd.read_csv(args.val_file)
    print('Validation loaded')
    print('Validation df get features ...')
    valid_df_extended = get_new_features(val_df)
    valid_df_extended.to_csv("/data/val_extended.csv", index=False)
    print()

    test_df = pd.read_csv(args.test_file)
    test_additional_df = pd.read_csv(args.test_add_file)
    test_df["route"] = test_additional_df["route"]

    print('Test loaded')
    print('Test df get features ...')
    test_df_extended = get_new_features(test_df)
    test_df_extended.to_csv("/data/test_extended.csv", index=False)
