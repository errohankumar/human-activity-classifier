import os
import warnings
import sys

from glob import glob

import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import sklearn

import seaborn as sns
import matplotlib.pyplot as plt


import pickle
import errno


# Reading the individual csv files within the subjects and finding features such as mean, median, iqr, zcr etc
def main(train_label_df, feature, flag):
    # list all subject's data floder
    folders = os.listdir('/Users/rohankumar/Desktop/BBDC 2019/Data/' + flag)

    df = pd.DataFrame()

    for folder in set(folders):
        # fetch files from each directory and calculate the new features
        data_df = pd.DataFrame(fetch_train_data(folder, feature, train_label_df, flag))

        # append each subject's dataframe to a common dataframe
        df = df.append(data_df, ignore_index=True)

    df = df.add_prefix(feature + "_")

    # Write new features into separate files
    out_file = '/Users/rohankumar/Desktop/BBDC 2019/Data/pickle/' + flag + '/' + feature + '.pickle'
    if not os.path.exists(os.path.dirname(out_file)):
        try:
            os.makedirs(os.path.dirname(out_file))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    pickle_out = open(out_file, "wb")
    pickle.dump(df, pickle_out)

    print("Excuted Successfully")


def fetch_train_data(folder, feature, train_label_df, flag):
    column_names = ['EMG1'
        , 'EMG2'
        , 'EMG3'
        , 'EMG4'
        , 'Airborne'
        , 'ACC upper X'
        , 'ACC upper Y'
        , 'ACC upper Z'
        , 'Goniometer X'
        , 'ACC lower X'
        , 'ACC lower Y'
        , 'ACC lower Z'
        , 'Goniometer Y'
        , 'gyro upper X'
        , 'gyro upper Y'
        , 'gyro upper Z'
        , 'gyro lower X'
        , 'gyro lower Y'
        , 'gyro lower Z'
                    ]
    # define a dataframe
    dataframe = pd.DataFrame()

    # filenames: holds all the activity files given subject
    file_names = glob('/Users/rohankumar/Desktop/BBDC 2019/Data/' + flag + '/' + folder + '/' + '*.csv')

    for file_name in file_names:
        df = pd.read_csv(file_name, names=column_names, header=None)

        # append the processed dataset to dataframe
        dataframe = dataframe.append(extract_features(df, feature, folder, file_name, train_label_df, flag),
                                     ignore_index=True)

    return dataframe


def extract_features(df, feature, folder, file_name, train_label_df, flag):
    new_df = pd.DataFrame()
    new_df = df
    new_df = feature_extract(new_df, feature)
    new_df = new_df.transpose()
    if flag == 'train':
        new_df['datafile'] = folder + '/' + os.path.basename(
            file_name)  ## os.path.basename extracts the filename only
        stats_df['activity'] = train_label_df[train_label_df.Datafile == new_df.datafile].Label.item()
    return (new_df)


def feature_extract(switcher_df, switcher_feature):
    switcher = {
        'mean': switcher_df.mean(),
        'median': switcher_df.median(),
        'min': switcher_df.min(),
        'max': switcher_df.max(),
        'std': switcher_df.std(),
        'variance': switcher_df.var(),
        'mad': switcher_df.mad(),
        'rms': np.sqrt(np.sum(np.power((switcher_df), 2)) / len(switcher_df)),
        'zcr': np.diff(np.signbit(switcher_df)).sum(),
        'iqr': switcher_df.quantile(0.75) - switcher_df.quantile(0.25),
        'pe': switcher_df.quantile(0.75),
        'kurtosis': kurtosis(switcher_df),
        'skew': skew(switcher_df)
    }
    return switcher.get(switcher_feature, "Invalid feature")


if __name__ == '__main__':
    # Initialise the train.csv for label extraction
    train_label_df = pd.read_csv('/Users/rohankumar/Desktop/BBDC 2019/Data/train.csv')
    print(train_label_df.shape)

    # flag is to determine test and train data
    flag = 'train'
    features = ['mean', 'median', 'min', 'max', 'std', 'variance', 'mad', 'rms', 'iqr', 'pe']

    for feature in features:
        main(train_label_df, feature, flag)