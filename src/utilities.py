import pandas
import math
from os import path, mkdir
import shutil
import numpy as np
from tensorflow.keras.losses import MeanSquaredError, Reduction


def calculate_reconstruction_error(original_data, reconstructed_data):
    mean_squared_error_instance = MeanSquaredError(reduction=Reduction.NONE)
    return mean_squared_error_instance(original_data, reconstructed_data).numpy()


def calculate_point_by_point_error(original_window, reconstructed_window):
    errors = list()
    for index, _ in enumerate(original_window):
        errors.append(calculate_reconstruction_error(
            [original_window[index]],
            [reconstructed_window[index]]
        ))
    weighted_average = np.average(errors, weights=np.divide(errors, np.max(errors)))
    return errors, weighted_average


def point_by_point_error(original_window, reconstructed_window):
    errors, weighted_average = calculate_point_by_point_error(original_window, reconstructed_window)
    print('Errors P96-100: {}'.format([np.percentile(errors, percentile) for percentile in np.arange(96, 100)]))
    print('Errors P50: {}'.format(np.percentile(errors, 50)))
    print('Errors W.AVG: {}'.format(weighted_average))
    print('Errors MEAN: {}'.format(np.mean(errors)))
    print('Errors SUM: {}'.format(np.sum(errors)))
    return errors


def count_downloaded_experiments_to_csv(file_to_save, directory, configuration, header):
    amount_of_experiments = 0
    downloaded_experiments = 0
    dataframe = pandas.DataFrame(columns=header)
    for zip_file, parameters in configuration.get_experiments():
        amount_of_experiments += 1
        if path.exists(path.join(directory, zip_file)):
            downloaded_experiments += 1
            parameters.append('SI')
        else:
            parameters.append('NO')
        dataframe.loc[amount_of_experiments] = parameters
    dataframe.to_csv(file_to_save, index=False, sep=';', decimal=',')
    print('Amount of experiments: {} (the half is {})'.format(amount_of_experiments,
                                                              math.floor(amount_of_experiments / 2)))
    print('Downloaded experiments: {}'.format(downloaded_experiments))
    print('Remaining experiments: {}'.format(amount_of_experiments - downloaded_experiments))


def move_experiments_to_directory(current_directory, target_directory, configuration):
    if not path.exists(target_directory):
        mkdir(target_directory)
    counter = 0
    for zip_file, _ in configuration.get_experiments():
        file_path = path.join(current_directory, zip_file)
        if path.exists(file_path):
            counter += 1
            shutil.move(file_path, path.join(target_directory, zip_file))
    print('{} files moved!'.format(counter))
