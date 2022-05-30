from typing import List

import pandas
from zipfile import ZipFile
from os import path
from abc import ABC, abstractmethod
import numpy


def read_csv_file(filename):
    return pandas.read_csv(filename, sep=';', decimal=',')


def create_csv_with_dataset_length(data_length, windows_list, displacements_list, file):
    data = list()
    for window in windows_list:
        data.append([
            len(range(0, data_length - window, displacement_fn(window))) for __, displacement_fn in displacements_list
        ])
    write_two_variable_table_to_csv_file(data, file, [label for label, __ in displacements_list], windows_list)


def write_two_variable_table_to_csv_file(data, file: str, columns: List[str], rows: List[str]):
    pandas.DataFrame(
        data,
        index=[row for row in rows],
        columns=[column for column in columns]
    ).to_csv(
        file,
        index=True,
        sep=';',
        decimal=',',
        encoding='utf-8-sig'
    )


def flat_list_of_lists(nested_list):
    data = list()
    for sublist in nested_list:
        data.extend(sublist)
    return data


def extract_files_from_zip(zip_file, directory):
    extracted_zip, _ = path.splitext(zip_file)
    experiment_folder = path.join(directory, extracted_zip)
    if not path.exists(experiment_folder):
        with ZipFile(path.join(directory, zip_file), 'r') as file:
            file.extractall(directory)
    return experiment_folder, extracted_zip


class CsvCreator(ABC):

    def __init__(self, configuration):
        self.configuration = configuration

    def combine_csv_files_from_experiment(self, data_directory, results_directory):
        counter = 0
        for zip_file, parameters in self.configuration.get_experiments():
            counter += 1
            experiment_folder, experiment_name = extract_files_from_zip(zip_file, data_directory)
            experiment_data = self.get_data(
                read_csv_file(path.join(experiment_folder, '{}.csv'.format(experiment_name))),
                parameters
            )

            model_filename = path.join(results_directory, self.get_filename(parameters))

            if path.exists(model_filename):
                experiment_data.to_csv(model_filename, index=False, sep=';', decimal=',', mode='a', header=False)
            else:
                experiment_data.to_csv(model_filename, index=False, sep=';', decimal=',')
        print('Finished !! {} models'.format(counter))

    def get_MCC_data(self, experiment):
        column = 'MCC'
        best_row = experiment[column].idxmax()
        return (experiment[column].min(),
                experiment[column].mean(),
                experiment[column].max(),
                experiment.at[best_row, 'id_modelo'])

    def sum_up_csv_files(self, csv_directory, sumup_filename):
        for parameters in self.get_model_parameters():
            sum_up = self.get_sum_up_by_group_dataframe(
                read_csv_file(path.join(csv_directory, self.get_filename(parameters))),
                parameters
            )
            if path.exists(sumup_filename):
                sum_up.to_csv(sumup_filename, index=False, sep=';', decimal=',', mode='a', header=False)
            else:
                sum_up.to_csv(sumup_filename, index=False, sep=';', decimal=',')

    def find_best_models(self, sumup_filename, table_filename, legend_function):
        sum_up = read_csv_file(sumup_filename)
        best_models = dict()
        for mode, pattern in zip(['promedio', 'mejor caso'], ['promedio MCC ({})', 'mejor MCC ({})']):
            print('Mejores modelos ({})... ({})'.format(mode, sumup_filename))
            best_models[mode] = list()
            for prefix in self.configuration.groups:
                column = pattern.format(prefix)
                best_row = sum_up[column].idxmax()
                best_models[mode].append(self.get_sum_up(best_row, column, sum_up, legend_function(prefix)))
        column = 'promedio MCC general'
        best_row = sum_up[column].idxmax()
        best_models['general'] = list()
        best_models['general'].append(self.get_sum_up(best_row, column, sum_up, 'general'))
        write_two_variable_table_to_csv_file(
            data=flat_list_of_lists([sublist for __, sublist in best_models.items()]),
            file=table_filename,
            columns=self.get_columns_for_best_model_csv(),
            rows=flat_list_of_lists([numpy.repeat(key, len(sublist)) for key, sublist in best_models.items()]),
        )

    @abstractmethod
    def get_data(self, experiment, parameters):
        pass

    def get_filename(self, parameters):
        return self.configuration.get_csv_filename(parameters)

    @abstractmethod
    def get_sum_up(self, best_row, column, sum_up, prefix):
        pass

    @abstractmethod
    def get_columns_for_best_model_csv(self):
        pass

    @abstractmethod
    def get_sum_up_by_group_dataframe(self, model_data, parameters):
        pass

    @abstractmethod
    def get_model_parameters(self):
        pass


class MLP2CsvCreator(CsvCreator):

    def __init__(self, configuration):
        super().__init__(configuration)

    def get_data(self, experiment, parameters):
        worst_MCC, average_MCC, best_MCC, best_model = self.get_MCC_data(experiment)
        return pandas.DataFrame(data={
            'unidades capa1': [parameters[0]],
            'ratio de dropout': [parameters[1]],
            'serie': [parameters[2]],
            'ventana': [parameters[3]],
            'desplazamiento': [parameters[4]],
            'peor MCC': [worst_MCC],
            'promedio MCC': [average_MCC],
            'mejor MCC': [best_MCC],
            'mejor modelo': [best_model]
        })

    def get_sum_up(self, best_row, column, sum_up, prefix):
        print('Mejor para {}: {:2} {} ... {:.2f}'.format(
            prefix,
            sum_up.at[best_row, 'unidades capa1'],
            sum_up.at[best_row, 'ratio de dropout'],
            sum_up.at[best_row, column]
        ))
        return [prefix,
                sum_up.at[best_row, 'unidades capa1'],
                sum_up.at[best_row, 'ratio de dropout'],
                sum_up.at[best_row, column]
                ]

    def get_columns_for_best_model_csv(self):
        return ['serie', 'Capa 1', 'Dropout', 'MCC']

    def get_model_parameters(self):
        parameters = list()
        for layer1_units in self.configuration.layer1_units:
            for dropout_rate in self.configuration.dropout_rate:
                parameters.append([
                    layer1_units,
                    dropout_rate
                ])
        return parameters

    def get_sum_up_by_group_dataframe(self, model_data, parameters):
        column = 'promedio MCC'
        groupby = model_data.groupby('serie')
        data = dict()
        data['unidades capa1'] = [parameters[0]]
        data['ratio de dropout'] = [parameters[1]]
        data['promedio MCC general'] = [model_data[column].mean()]
        for group in self.configuration.groups:
            data['promedio MCC ({})'.format(group)] = [groupby.get_group(group)[column].mean()]
            data['mejor MCC ({})'.format(group)] = [groupby.get_group(group)[column].max()]
        return pandas.DataFrame(data=data)


class MLP3CsvCreator(CsvCreator):

    def __init__(self, configuration):
        super().__init__(configuration)

    def get_data(self, experiment, parameters):
        worst_MCC, average_MCC, best_MCC, best_model = self.get_MCC_data(experiment)
        return pandas.DataFrame(data={
            'unidades capa1': [parameters[0]],
            'unidades capa2': [parameters[1]],
            'ratio de dropout': [parameters[2]],
            'serie': [parameters[3]],
            'ventana': [parameters[4]],
            'desplazamiento': [parameters[5]],
            'peor MCC': [worst_MCC],
            'promedio MCC': [average_MCC],
            'mejor MCC': [best_MCC],
            'mejor modelo': [best_model]
        })

    def get_sum_up(self, best_row, column, sum_up, prefix):
        print('Mejor para {}: {:2} {:2} {} ... {:.2f}'.format(
            prefix,
            sum_up.at[best_row, 'unidades capa1'],
            sum_up.at[best_row, 'unidades capa2'],
            sum_up.at[best_row, 'ratio de dropout'],
            sum_up.at[best_row, column]
        ))
        return [
            prefix,
            sum_up.at[best_row, 'unidades capa1'],
            sum_up.at[best_row, 'unidades capa2'],
            sum_up.at[best_row, 'ratio de dropout'],
            sum_up.at[best_row, column]
        ]

    def get_columns_for_best_model_csv(self):
        return ['serie', 'Capa 1', 'Capa 2', 'Dropout', 'MCC']

    def get_model_parameters(self):
        parameters = list()
        for layer1_units in self.configuration.layer1_units:
            for layer2_units in self.configuration.layer2_units:
                for dropout_rate in self.configuration.dropout_rate:
                    parameters.append([
                        layer1_units,
                        layer2_units,
                        dropout_rate
                    ])
        return parameters

    def get_sum_up_by_group_dataframe(self, model_data, parameters):
        column = 'promedio MCC'
        groupby = model_data.groupby('serie')
        data = dict()
        data['unidades capa1'] = [parameters[0]]
        data['unidades capa2'] = [parameters[1]]
        data['ratio de dropout'] = [parameters[2]]
        data['promedio MCC general'] = [model_data[column].mean()]
        for group in self.configuration.groups:
            data['promedio MCC ({})'.format(group)] = [groupby.get_group(group)[column].mean()]
            data['mejor MCC ({})'.format(group)] = [groupby.get_group(group)[column].max()]
        return pandas.DataFrame(data=data)


class AutoencoderCsvCreator(CsvCreator):

    def __init__(self, configuration):
        super().__init__(configuration)

    def get_data(self, experiment, parameters):
        worst_MCC, average_MCC, best_MCC, best_model = self.get_MCC_data(experiment)
        return pandas.DataFrame(data={
            'capa intermedia': parameters[0],
            'bottleneck': parameters[1],
            'serie': parameters[2],
            'ventana': parameters[3],
            'desplazamiento': parameters[4],
            'peor MCC': [worst_MCC],
            'promedio MCC': [average_MCC],
            'mejor MCC': [best_MCC],
            'mejor modelo': [best_model]
        })

    def get_sum_up(self, best_row, column, sum_up, prefix):
        print('Mejor para {}: {:2} {:2} ... {:.2f}'.format(
            prefix,
            sum_up.at[best_row, 'capa intermedia'],
            sum_up.at[best_row, 'bottleneck'],
            sum_up.at[best_row, column]
        ))
        return [
            prefix,
            sum_up.at[best_row, 'capa intermedia'],
            sum_up.at[best_row, 'bottleneck'],
            sum_up.at[best_row, column]
        ]

    def get_columns_for_best_model_csv(self):
        return ['serie', 'Capa Intermedia', 'Bottleneck', 'MCC']

    def get_model_parameters(self):
        parameters = list()
        for intermediate_units in self.configuration.intermediate:
            for bottleneck_units in self.configuration.bottleneck:
                parameters.append([
                    intermediate_units,
                    bottleneck_units
                ])
        return parameters

    def get_sum_up_by_group_dataframe(self, model_data, parameters):
        column = 'promedio MCC'
        groupby = model_data.groupby('serie')
        data = dict()
        data['capa intermedia'] = [parameters[0]]
        data['bottleneck'] = [parameters[1]]
        data['promedio MCC general'] = [model_data[column].mean()]
        for group in self.configuration.groups:
            data['promedio MCC ({})'.format(group)] = [groupby.get_group(group)[column].mean()]
            data['mejor MCC ({})'.format(group)] = [groupby.get_group(group)[column].max()]
        return pandas.DataFrame(data=data)
