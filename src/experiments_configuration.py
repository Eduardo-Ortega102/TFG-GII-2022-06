from math import floor
from abc import ABC, abstractmethod
from typing import List


def get_natural_windows() -> List[int]:
    return [10, 20, 40, 60, 90]


def get_senoidal_windows() -> List[int]:
    return [5, 10, 20, 40, 60]


def get_natural_windows_autoencoder() -> List[int]:
    return [20, 40, 60, 90]


def get_senoidal_windows_autoencoder() -> List[int]:
    return [20, 40, 60]


def get_natural_prefixes() -> List[str]:
    return ['GB1', 'GB2', 'GB3', 'GB5', 'GB7', 'GB10']


def get_senoidal_prefixes() -> List[str]:
    return ['Simple-1-1P', 'Simple-1-1P-Nois025', 'Simple-1-1P-Nois05', 'Simple-1-1P-Nois10', 'Simple-1-1P-Nois20',
            'Simple-2P', 'Simple-2P-Nois025', 'Simple-2P-Nois05', 'Simple-2P-Nois10', 'Simple-2P-Nois20',
            'Comb-1-1P', 'Comb-1-1P-Nois025', 'Comb-1-1P-Nois05', 'Comb-1-1P-Nois10', 'Comb-1-1P-Nois20',
            'Comb-2P', 'Comb-2P-Nois025', 'Comb-2P-Nois05', 'Comb-2P-Nois10', 'Comb-2P-Nois20'
            ]


def get_senoidal_noiseless_prefixes() -> List[str]:
    def prefix_without_noise(prefix) -> bool:
        return 'Nois' not in prefix

    return list(filter(prefix_without_noise, get_senoidal_prefixes()))


def get_senoidal_1percent_prefixes() -> List[str]:
    def prefix_with_1_1P(prefix) -> bool:
        return '1-1P' in prefix

    return list(filter(prefix_with_1_1P, get_senoidal_prefixes()))


def get_senoidal_2percent_prefixes() -> List[str]:
    def prefix_with_2P(prefix) -> bool:
        return '2P' in prefix

    return list(filter(prefix_with_2P, get_senoidal_prefixes()))


def get_senoidal_1percent_prefixes_by_kind(kind) -> List[str]:
    def prefix_with_kind(prefix) -> bool:
        return kind in prefix

    return list(filter(prefix_with_kind, get_senoidal_1percent_prefixes()))


def get_senoidal_prefixes_by_kind(kind) -> List[str]:
    def prefix_with_kind(prefix) -> bool:
        return kind in prefix

    return list(filter(prefix_with_kind, get_senoidal_prefixes()))


def get_senoidal_2percent_prefixes_by_kind(kind) -> List[str]:
    def prefix_with_kind(prefix) -> bool:
        return kind in prefix

    return list(filter(prefix_with_kind, get_senoidal_2percent_prefixes()))


class Configuration(ABC):

    def __init__(self, groups, windows, displacements, getting_baseline):
        self.groups = groups
        self.windows = windows
        self.displacements = displacements
        self.baseline_string = '_base' if getting_baseline else ''

    def get_iteration(self, window_length, displacement):
        return self.get_displacements(window_length).index(displacement)

    def get_displacements(self, window_length):
        if self.displacements is None:
            return [window_length, floor(window_length / 2), 1]
        return self.displacements

    @abstractmethod
    def get_csv_files(self):
        pass

    @abstractmethod
    def get_experiments(self):
        pass

    @abstractmethod
    def get_string(self, parameters):
        pass

    @abstractmethod
    def get_csv_filename(self, parameters):
        pass


class MLP2Configuration(Configuration):

    def __init__(self, groups, layer1_units=[25, 20, 15], dropout_rate=[0.2, 0.3], windows=None,
                 displacements=None, getting_baseline=False):
        super().__init__(groups, windows, displacements, getting_baseline)
        self.layer1_units = layer1_units
        self.dropout_rate = dropout_rate

    def get_csv_files(self):
        files = list()
        for layer1_units in self.layer1_units:
            for dropout_rate in self.dropout_rate:
                files.append(self.get_csv_filename([layer1_units, dropout_rate]))
        return files

    def get_csv_filename(self, parameters):
        return 'MLP2_model_{}_{}{}.csv'.format(parameters[0], parameters[1], self.baseline_string)

    def get_string(self, parameters):
        hidden1 = parameters[0]
        dropout_rate = parameters[1]
        prefix = parameters[2]
        window_length = parameters[3]
        displacement = parameters[4]
        return prefix, "prefix='{}', window_length={}, displacement={}, h1_units={}, rate={}".format(
            prefix,
            window_length,
            displacement,
            hidden1,
            dropout_rate
        )

    def get_experiments(self):
        experiments = list()
        for layer1_units in self.layer1_units:
            for dropout_rate in self.dropout_rate:
                for prefix in self.groups:
                    for window in self.windows:
                        for displacement in self.get_displacements(window):
                            file_name = '#{}{}_Wind{}_Displ{}_OF{}.zip'.format(
                                prefix,
                                '-L{}-D{}-'.format(layer1_units, dropout_rate),
                                window,
                                displacement,
                                'binary_crossentropy'
                            )
                            data = [
                                layer1_units, dropout_rate, prefix, window, displacement
                            ]
                            experiments.append((file_name, data))
        return experiments


class MLP3Configuration(Configuration):

    def __init__(self, groups, layer1_units=[25, 20, 15], layer2_units=[15, 10, 5], dropout_rate=[0.2, 0.3],
                 windows=None, displacements=None, getting_baseline=False):
        super().__init__(groups, windows, displacements, getting_baseline)
        self.layer1_units = layer1_units
        self.layer2_units = layer2_units
        self.dropout_rate = dropout_rate

    def get_csv_files(self):
        files = list()
        for layer1_units in self.layer1_units:
            for layer2_units in self.layer2_units:
                for dropout_rate in self.dropout_rate:
                    files.append(self.get_csv_filename([layer1_units, layer2_units, dropout_rate]))
        return files

    def get_csv_filename(self, parameters):
        return 'MLP3_model_{}_{}_{}{}.csv'.format(parameters[0], parameters[1], parameters[2], self.baseline_string)

    def get_string(self, parameters):
        hidden1 = parameters[0]
        hidden2 = parameters[1]
        dropout_rate = parameters[2]
        prefix = parameters[3]
        window_length = parameters[4]
        displacement = parameters[5]
        return prefix, "prefix='{}', window_length={}, displacement={}, h1_units={}, h2_units={}, rate={}".format(
            prefix,
            window_length,
            displacement,
            hidden1,
            hidden2,
            dropout_rate
        )

    def get_experiments(self):
        experiments = list()
        for layer1_units in self.layer1_units:
            for layer2_units in self.layer2_units:
                for dropout_rate in self.dropout_rate:
                    for prefix in self.groups:
                        for window in self.windows:
                            for displacement in self.get_displacements(window):
                                file_name = '#{}{}_Wind{}_Displ{}_OF{}.zip'.format(
                                    prefix,
                                    '-L{}-M{}-D{}-'.format(layer1_units, layer2_units, dropout_rate),
                                    window,
                                    displacement,
                                    'binary_crossentropy'
                                )
                                data = [
                                    layer1_units, layer2_units, dropout_rate, prefix, window, displacement
                                ]
                                experiments.append((file_name, data))
        return experiments


class AutoencoderConfiguration(Configuration):

    def __init__(self, autoencoder_type, groups, intermediate=[25, 20, 15], bottleneck=[15, 10, 5], windows=None,
                 displacements=None, getting_baseline=False):
        super().__init__(groups, windows, displacements, getting_baseline)
        self.autoencoder_type = autoencoder_type
        self.intermediate = intermediate
        self.bottleneck = bottleneck

    def get_csv_files(self):
        files = list()
        for intermediate_units in self.intermediate:
            for bottleneck_units in self.bottleneck:
                files.append(self.get_csv_filename([intermediate_units, bottleneck_units]))
        return files

    def get_csv_filename(self, parameters):
        return '{}_model_{}_{}{}.csv'.format(self.autoencoder_type, parameters[0], parameters[1], self.baseline_string)

    def get_string(self, parameters):
        intermediate = parameters[0]
        bottleneck = parameters[1]
        prefix = parameters[2]
        window_length = parameters[3]
        displacement = parameters[4]
        return prefix, "prefix='{}', window_length={}, displacement={}, intermediate={}, bottleneck={}".format(
            prefix,
            window_length,
            displacement,
            intermediate,
            bottleneck
        )

    def get_experiments(self):
        experiments = list()
        for intermediate_units in self.intermediate:
            for bottleneck_units in self.bottleneck:
                for prefix in self.groups:
                    for window in self.windows:
                        for displacement in self.get_displacements(window):
                            file_name = '#{}{}_Wind{}_Displ{}_OF{}.zip'.format(
                                prefix,
                                '-IN{}-BO{}-'.format(intermediate_units, bottleneck_units),
                                window,
                                displacement,
                                'mean_squared_error'
                            )
                            data = [
                                intermediate_units, bottleneck_units, prefix, window, displacement
                            ]
                            experiments.append((file_name, data))
        return experiments
