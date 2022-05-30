import sys
from os import path, mkdir
from math import floor

DATA_FILES = {
    'GB1': 'grillos_y_ballena_v2_1porciento.csv',
    'GB2': 'grillos_y_ballena_v2_2porciento.csv',
    'GB3': 'grillos_y_ballena_v2_3porciento.csv',
    'GB5': 'grillos_y_ballena_v2_5porciento.csv',
    'GB7': 'grillos_y_ballena_v2_7porciento.csv',
    'GB10': 'grillos_y_ballena_v2_10porciento.csv',
    'Simple-1-1P': 'senoidal_anomala_1.1porciento_40.csv',
    'Simple-2P': 'senoidal_anomala_2porciento.csv',
    'Simple-1-1P-Nois20': 'senoidal_anomala_1.1porciento_40_noisy0.2.csv',
    'Simple-1-1P-Nois10': 'senoidal_anomala_1.1porciento_40_noisy0.1.csv',
    'Simple-1-1P-Nois05': 'senoidal_anomala_1.1porciento_40_noisy0.05.csv',
    'Simple-1-1P-Nois025': 'senoidal_anomala_1.1porciento_40_noisy0.025.csv',
    'Simple-2P-Nois20': 'senoidal_anomala_2porciento_noisy0.2.csv',
    'Simple-2P-Nois10': 'senoidal_anomala_2porciento_noisy0.1.csv',
    'Simple-2P-Nois05': 'senoidal_anomala_2porciento_noisy0.05.csv',
    'Simple-2P-Nois025': 'senoidal_anomala_2porciento_noisy0.025.csv',
    'Comb-1-1P': 'senoidal_combinada_anomala_1.1porciento_40.csv',
    'Comb-2P': 'senoidal_combinada_anomala_2porciento_40.csv',
    'Comb-1-1P-Nois20': 'senoidal_combinada_anomala_1.1porciento_40_noisy0.2.csv',
    'Comb-1-1P-Nois10': 'senoidal_combinada_anomala_1.1porciento_40_noisy0.1.csv',
    'Comb-1-1P-Nois05': 'senoidal_combinada_anomala_1.1porciento_40_noisy0.05.csv',
    'Comb-1-1P-Nois025': 'senoidal_combinada_anomala_1.1porciento_40_noisy0.025.csv',
    'Comb-2P-Nois20': 'senoidal_combinada_anomala_2porciento_40_noisy0.2.csv',
    'Comb-2P-Nois10': 'senoidal_combinada_anomala_2porciento_40_noisy0.1.csv',
    'Comb-2P-Nois05': 'senoidal_combinada_anomala_2porciento_40_noisy0.05.csv',
    'Comb-2P-Nois025': 'senoidal_combinada_anomala_2porciento_40_noisy0.025.csv'
}


class ExperimentsCreator:

    def __init__(self, configuration, skip_training=False):
        self.configuration = configuration
        self.downloaded_experiments = 0
        self.remaining_experiments = 0
        self.amount_of_experiments = 0
        self.skip_training = skip_training

    def __count_experiments(self, directory, experiment_names):
        self.amount_of_experiments = len(experiment_names)
        for zip_file, _ in experiment_names:
            if path.exists(path.join(directory, zip_file)):
                self.downloaded_experiments += 1
        self.remaining_experiments = self.amount_of_experiments - self.downloaded_experiments
        print('Amount of experiments: {} (the half is {})'.format(
            self.amount_of_experiments,
            floor(self.amount_of_experiments / 2))
        )
        print('Downloaded experiments: {}'.format(self.downloaded_experiments))
        print('Remaining experiments: {}'.format(self.remaining_experiments))

    def generate_functions_calls(self, directory, function_name, amount_of_threads, batch_number=None):
        experiment_names = self.configuration.get_experiments()
        self.__count_experiments(directory, experiment_names)
        if self.remaining_experiments > 0:
            batch_size = max(1, floor(self.remaining_experiments / amount_of_threads))
            print('Batch size: {} for {} threads'.format(batch_size, amount_of_threads))
            batchs_directory = 'batchs_{}'.format(function_name)
            if not path.exists(batchs_directory):
                mkdir(batchs_directory)
            batch_counter = 1
            if batch_number is None:
                batch_number = batch_counter
            batch_content = list()
            amount_of_calls_created = 0
            for zip_file, parameters in experiment_names:
                if path.exists(path.join(directory, zip_file)):
                    continue
                batch_content.append(parameters)
                if batch_counter > amount_of_threads:
                    continue
                if len(batch_content) == batch_size:
                    amount_of_calls_created += len(batch_content)
                    self.__create_batch_file(batch_number, batchs_directory, function_name, batch_content)
                    batch_content = list()
                    batch_counter += 1
                    batch_number += 1
            if len(batch_content) > 0:
                amount_of_calls_created += len(batch_content)
                self.__spread_remaining_content(amount_of_threads, batchs_directory, function_name, batch_content)
            print('Created {} calls'.format(amount_of_calls_created))

    def __create_batch_file(self, batch_number, batchs_directory, function_name, batch_content):
        original_stdout = sys.stdout
        with open(path.join(batchs_directory, 'batch_{}.py'.format(batch_number)), 'w') as output:
            sys.stdout = output
            print('from experiments_runners import {}'.format(function_name))
            for parameters in batch_content:
                self.__add_function_call_to_batch(parameters, function_name)
        sys.stdout = original_stdout

    def __add_function_call_to_batch(self, parameters, function_name):
        prefix, line = self.configuration.get_string(parameters)
        if self.skip_training:
            print("{}(data_file='{}', {}, skip_training=True)".format(function_name, DATA_FILES[prefix], line))
        else:
            print("{}(data_file='{}', {})".format(function_name, DATA_FILES[prefix], line))

    def __spread_remaining_content(self, amount_of_threads, batchs_directory, function_name, batch_content):
        print('Spreading {} calls...'.format(len(batch_content)))
        original_stdout = sys.stdout
        batch_number = amount_of_threads
        for parameters in batch_content:
            with open(path.join(batchs_directory, 'batch_{}.py'.format(batch_number)), 'a') as output:
                sys.stdout = output
                self.__add_function_call_to_batch(parameters, function_name)
            batch_number = batch_number - 1 if batch_number > 0 else amount_of_threads
        sys.stdout = original_stdout
