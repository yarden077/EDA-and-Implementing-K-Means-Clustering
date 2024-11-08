import pandas as pd
from datetime import datetime


def load_data(path):
    """ reads and returns the pandas DataFrame """
    df = pd.read_csv(path)
    return df


def data_analysis(df):
    """ prints statistics on the transformed df """
    print("Part A:")
    print('describe output:')
    print(df.describe().to_string())
    print()
    print('corr output:')
    corr = df.corr()
    print(corr.to_string())
    print()
    dict = get_correlation(df)
    dict = sort_dictionary_by_correlation(dict)
    print_statistics(dict)
    print()
    print_mean(df)


def add_new_columns(df):
    """ adds columns to df and returns the new df """
    df['season_name'] = df['season'].apply(name_the_season)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    df['Hour'] = df['timestamp'].apply(get_hour)
    df['Day'] = df['timestamp'].apply(get_day)
    df['Month'] = df['timestamp'].apply(get_month)
    df['Year'] = df['timestamp'].apply(get_year)
    df['is_weekend_holiday'] = df.apply(check_weekend_holiday, axis=1)
    df['t_diff'] = df.apply(calculate_difference, axis=1)
    return df


def check_weekend_holiday(value):
    """ function returns the value of is_weekend_holiday according to the conditions """
    if value.is_holiday == 0 and value.is_weekend == 0:
        return 0
    if value.is_holiday == 0 and value.is_weekend == 1:
        return 1
    if value.is_holiday == 1 and value.is_weekend == 0:
        return 2
    return 3


def name_the_season(value):
    """ function gets the column of season in the dataframe and sets a name for every season by a numeric value """
    if value == 0:
        return "spring"
    if value == 1:
        return "summer"
    if value == 2:
        return "fall"
    if value == 3:
        return "winter"


def calculate_difference(value):
    """calculates the difference between t2 and t1 features """
    return value.t2 - value.t1


def get_hour(value):
    return datetime.time(value).hour


def get_month(value):
    return datetime.date(value).month


def get_year(value):
    return datetime.date(value).year


def get_day(value):
    return datetime.date(value).day


def get_correlation(df):
    """
    :param df: The given data frame.
    In this function we set a dictionary that the keys are pairs of features, and the values are the correlation
    between the features.
    :return: The dictionary
    """
    dict1 = {}
    features = df.columns.values.tolist()
    number_of_cols = df.shape[1]
    for i in range(number_of_cols):
        for j in range(i + 1, number_of_cols):
            if legal(features[i], features[j]):
                dict1[(features[i], features[j])] = abs(df[features[i]].corr(df[features[j]]))
    return dict1


def legal(feature1, feature2):
    """ calculating correlation of timestamp or season name is not wanted """
    if feature1 == 'timestamp' or feature1 == 'season_name' or feature2 == 'timestamp' or feature2 == 'season_name':
        return False
    return True


def sort_dictionary_by_correlation(dict1):
    """ Function sorts the dictionary by value from the highest correlation to lowest """
    dict1 = (dict(sorted(dict1.items(), key=lambda value: value[1], reverse=True)))
    return dict1


def print_statistics(dict):
    """ printing the statistics of the data """
    len_dict = len(dict)
    counter = 1
    print("Highest correlated are:")
    for i in range(5):
        print(f"{i+1}. {list(dict)[i]} with {format(list(dict.values())[i], '.6f')}")
    print()
    print("Lowest correlated are:")
    for i in range(len_dict - 1, len_dict - 6, -1):
        print(f"{counter}. {list(dict)[i]} with {format(list(dict.values())[i], '.6f')}")
        counter += 1


def print_mean(df):
    """ function calculates t_diff mean in every season and also the mean of t_diff """
    counter = 0
    seasons = ['fall', 'spring', 'summer', 'winter']
    df1 = df.groupby(['season_name'])['t_diff'].mean()
    for i in seasons:
        print(f"{i} average t_diff is {format(df1[counter], '.2f')}")
        counter += 1
    print(f"All average t_diff is {format(df['t_diff'].mean(), '.2f')}")
    





