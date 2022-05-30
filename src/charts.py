import math
import random
from statistics import mean
from typing import List
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from abc import ABC
from csv_creator import read_csv_file
from os import path


class Displacements:
    @staticmethod
    def one(window: int) -> int:
        return 1

    @staticmethod
    def one_half(window: int) -> int:
        return math.floor(window / 2)

    @staticmethod
    def window(window: int) -> int:
        return window

    @staticmethod
    def values():
        return [
            ('Un punto', Displacements.one),
            ('Mitad de ventana', Displacements.one_half),
            ('Ventana', Displacements.window)
        ]


def get_MLP2_model_data(file, layer1_units, dropout_rate):
    data = read_csv_file(file)
    row = data[(data['unidades capa1'] == layer1_units) &
               (data['ratio de dropout'] == dropout_rate)
               ]
    return row


def get_MLP3_model_data(file, layer1_units, layer2_units, dropout_rate):
    data = read_csv_file(file)
    row = data[(data['unidades capa1'] == layer1_units) &
               (data['unidades capa2'] == layer2_units) &
               (data['ratio de dropout'] == dropout_rate)
               ]
    return row


def get_autoencoder_model_data(file, intermediate, bottleneck):
    data = read_csv_file(file)
    row = data[(data['capa intermedia'] == intermediate) &
               (data['bottleneck'] == bottleneck)
               ]
    return row


def get_autoencoder_matrix_config(csv_directory, autoencoder_type, intermediate, bottleneck):
    return Matrix2x2Configuration(
        architecture=autoencoder_type,
        dataframe=get_autoencoder_model_data(
            intermediate=intermediate,
            bottleneck=bottleneck,
            file=path.join(csv_directory, '{}_resumen.csv'.format(autoencoder_type))
        ),
        base_dataframe=get_autoencoder_model_data(
            intermediate=intermediate,
            bottleneck=bottleneck,
            file=path.join(csv_directory, '{}_resumen_base.csv'.format(autoencoder_type))
        )
    )


def get_autoencoder_matrix_config_from_file(csv_directory, autoencoder_type, csv_file):
    return Matrix2x2Configuration(
        architecture=autoencoder_type,
        dataframe=read_csv_file(path.join(csv_directory, autoencoder_type, '{}.csv'.format(csv_file))),
        base_dataframe=read_csv_file(path.join(csv_directory, autoencoder_type, '{}_base.csv'.format(csv_file)))
    )


def get_anomaly_percentage(prefix):
    return '2' if '2P' in prefix else '1.1'


def get_kind_of_serie(prefix):
    return 'Simple' if 'Simple' in prefix else 'Combinada'


NOISE_LABEL_AND_PERCENTAGE_MAPPING = {
    'Nois20': '20',
    'Nois10': '10',
    'Nois05': '5',
    'Nois025': '2.5',
}


def build_legend_entry(prefix):
    def get_noise_percentage():
        for label in NOISE_LABEL_AND_PERCENTAGE_MAPPING:
            if label in prefix:
                return '(ruido {}%)'.format(NOISE_LABEL_AND_PERCENTAGE_MAPPING[label])
        return '(sin ruido)'

    return '{}% anomalías, Senoidal {} {}'.format(
        get_anomaly_percentage(prefix),
        get_kind_of_serie(prefix),
        get_noise_percentage()
    )


def get_legend_label_for_displacements_chart(prefix):
    def get_noise_percentage():
        for label in NOISE_LABEL_AND_PERCENTAGE_MAPPING:
            if label in prefix:
                return '(ruido {}%)'.format(NOISE_LABEL_AND_PERCENTAGE_MAPPING[label])
        return ''

    return '{}% anomalías, Senoidal {} {}'.format(
        get_anomaly_percentage(prefix),
        get_kind_of_serie(prefix),
        get_noise_percentage()
    )


def build_legend_entry_for_animals(prefix):
    def get_percentage():
        if '10' in prefix:
            return '10'
        if '2' in prefix:
            return '2'
        if '3' in prefix:
            return '3'
        if '5' in prefix:
            return '5'
        if '7' in prefix:
            return '7'
        return '1'

    return 'Grillo-Ballena, {}%'.format(
        get_percentage()
    )


class BarChart(ABC):

    def __init__(self):
        self.figsize = (11, 6.5)

    def get_colors(self, amount):
        def r():
            return random.randint(0, 255)

        colors = ['cyan', 'orange', 'darkgray', 'violet', 'darkkhaki', 'orangered', 'gold', 'pink', 'palegreen', 'peru']
        while amount > len(colors):
            colors.append('#{:02x}{:02x}{:02x}'.format(r(), r(), r()))
        return colors

    def add_bars(self, axes, fill_data_callback):
        axes.set_ylim([0, 1.10])
        axes.grid(axis='y')
        axes.set_ylabel('C.C. Matthews ', fontsize=12)
        axes.set_yticks([y / 10 for y in range(0, 11, 1)])
        self.annotate_bars(fill_data_callback(axes), axes)

    def annotate_bars(self, bars, axes):
        if bars is not None:
            number_pattern = '{:.2f}'
            for bar, baseline in bars:
                for rectangle in bar.patches:
                    height = number_pattern.format(rectangle.get_height())
                    x_position = rectangle.get_x() + rectangle.get_width() / 2
                    axes.annotate(
                        text=height,
                        xy=(x_position, float(height)),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center',
                        va='bottom'
                    )
                    axes.plot(x_position, baseline, "k*")
                    axes.annotate(
                        text=(number_pattern.format(baseline)),
                        xy=(x_position, float(baseline)),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center',
                        va='bottom'
                    )

    def get_legend_from_figure(self, figure):
        star = mlines.Line2D([], [], color='black', marker='*', linestyle='None')
        handles, labels = figure.gca().get_legend_handles_labels()
        handles.append(star)
        labels.append('Valor de referencia (modelo sin entrenar)')
        return dict(zip(labels, handles))


class FourRowsChart(BarChart):

    def __init__(self):
        super().__init__()

    def plot(self, title, first_callback, second_callback, third_callback, fourth_callback):
        figure = plt.figure(constrained_layout=True, figsize=self.figsize)
        gs = figure.add_gridspec(nrows=5, ncols=2, height_ratios=[1, 1, 1, 1, 0.25])
        ax_1 = figure.add_subplot(gs[0, :])
        ax_2 = figure.add_subplot(gs[1, :])
        ax_3 = figure.add_subplot(gs[2, :])
        ax_4 = figure.add_subplot(gs[3, :])
        figure.suptitle(title, fontsize=16)

        self.add_bars(ax_1, first_callback)
        self.add_bars(ax_2, second_callback)
        self.add_bars(ax_3, third_callback)
        self.add_bars(ax_4, fourth_callback)

        dictionary = self.get_legend_from_figure(figure)
        figure.legend(
            handles=dictionary.values(),
            labels=dictionary.keys(),
            bbox_to_anchor=(0.5, 0.01),  # to move down, decrease second number
            loc='center',
            ncol=2
        )
        plt.show()


class Matrix2x2Configuration:
    def __init__(self, architecture, dataframe, base_dataframe):
        self.architecture = architecture
        self.dataframe = dataframe
        self.base_dataframe = base_dataframe


class Matrix2x2Chart(BarChart):

    def __init__(self):
        super().__init__()

    def plot(self, title, row_1_title, row_2_title, config1: Matrix2x2Configuration, config2: Matrix2x2Configuration,
             first_callback, second_callback, third_callback, fourth_callback):
        figure = plt.figure(constrained_layout=True, figsize=self.figsize)
        gs = figure.add_gridspec(nrows=3, ncols=2, height_ratios=[1, 1, 0.25])
        ax_1 = figure.add_subplot(gs[0, 0])
        ax_2 = figure.add_subplot(gs[1, 0])
        ax_3 = figure.add_subplot(gs[0, 1])
        ax_4 = figure.add_subplot(gs[1, 1])
        figure.suptitle(title, fontsize=16)
        ax_1.set_title('{}, {}'.format(row_1_title, config1.architecture), fontsize=14)
        ax_2.set_title('{}, {}'.format(row_2_title, config1.architecture), fontsize=14)
        ax_3.set_title('{}, {}'.format(row_1_title, config2.architecture), fontsize=14)
        ax_4.set_title('{}, {}'.format(row_2_title, config2.architecture), fontsize=14)

        self.add_bars(ax_1, lambda ax: first_callback(ax, config1.dataframe, config1.base_dataframe))
        self.add_bars(ax_2, lambda ax: second_callback(ax, config1.dataframe, config1.base_dataframe))
        self.add_bars(ax_3, lambda ax: third_callback(ax, config2.dataframe, config2.base_dataframe))
        self.add_bars(ax_4, lambda ax: fourth_callback(ax, config2.dataframe, config2.base_dataframe))

        dictionary = self.get_legend_from_figure(figure)
        figure.legend(
            handles=dictionary.values(),
            labels=dictionary.keys(),
            bbox_to_anchor=(0.5, -0.01),  # to move down, decrease second number
            loc='center',
            ncol=3
        )
        plt.show()


class GeneralPerformanceChart(Matrix2x2Chart):

    def __init__(self):
        super().__init__()

    def fill_in_general_bars(self, ax, column, dataframe, base_dataframe, series):
        width = 0.10
        bars = []
        ax.set_xlabel('Señales', fontsize=12)
        ax.set_xticklabels([])
        x_positions = [width * i for i in range(len(series))]
        colors = self.get_colors(len(series))
        for i in range(len(series)):
            legend_entry, serie = series[i]
            mcc__format = '{} MCC ({})'.format(column, serie)
            bar = ax.bar(x_positions[i], dataframe[mcc__format], width=width / 1.5, color=colors[i], label=legend_entry)
            bars.append((bar, base_dataframe[mcc__format][0]))
        return bars

    def plot_dataframe(self, title, row_1_title, row_2_title, row_1_column, row_2_column,
                       config1: Matrix2x2Configuration, config2: Matrix2x2Configuration, series):
        super().plot(
            title,
            row_1_title,
            row_2_title,
            config1,
            config2,
            lambda ax, dataframe, base_dataframe: self.fill_in_general_bars(ax, row_1_column, dataframe, base_dataframe,
                                                                            series),
            lambda ax, dataframe, base_dataframe: self.fill_in_general_bars(ax, row_2_column, dataframe, base_dataframe,
                                                                            series),
            lambda ax, dataframe, base_dataframe: self.fill_in_general_bars(ax, row_1_column, dataframe, base_dataframe,
                                                                            series),
            lambda ax, dataframe, base_dataframe: self.fill_in_general_bars(ax, row_2_column, dataframe, base_dataframe,
                                                                            series)
        )


class WindowsPerformanceChart(FourRowsChart):

    def __init__(self, config_1, config_2, windows, bar_width=0.20, chart_width=12, chart_height=6.5):
        super().__init__()
        self.config_1 = config_1
        self.config_2 = config_2
        self.figsize = (chart_width, chart_height)
        self.bar_width = bar_width
        self.windows = windows

    def fill_in_bars(self, ax, config, subtitle, prefixes, column):
        bars = []
        data = config.dataframe
        baseline = config.base_dataframe
        colors = self.get_colors(len(prefixes))
        ax.set_title('{}, {}'.format(subtitle, config.architecture), fontsize=14)
        prefix_index = 0
        for legend_entry, prefix in prefixes:
            window_index = 0
            for window in self.windows:
                bar = ax.bar(self.bar_width * prefix_index + window_index,
                             mean(data[(data['serie'] == prefix) & (data['ventana'] == window)][column]),
                             color=colors[prefix_index], label=legend_entry, width=self.bar_width / 1.5,
                             edgecolor='white')
                bars.append(
                    (bar, mean(baseline[(baseline['serie'] == prefix) & (baseline['ventana'] == window)][column])))
                window_index += 1
            prefix_index += 1
        ax.set_xlabel('Tamaño de Ventana', fontsize=12)
        ax.set_xticks([x + 0.3 for x in range(len(self.windows))])
        ax.set_xticklabels(self.windows)
        return bars

    def plot_dataframe(self, title, subtitle1, subtitle2, prefixes):
        super().plot(
            title,
            lambda ax: self.fill_in_bars(ax, self.config_1, subtitle1, prefixes, 'promedio MCC'),
            lambda ax: self.fill_in_bars(ax, self.config_2, subtitle1, prefixes, 'promedio MCC'),
            lambda ax: self.fill_in_bars(ax, self.config_1, subtitle2, prefixes, 'mejor MCC'),
            lambda ax: self.fill_in_bars(ax, self.config_2, subtitle2, prefixes, 'mejor MCC'),
        )


class RowsDisplacementsChart(FourRowsChart):

    def __init__(self, config_1, config_2, windows, bar_width=0.20, chart_width=12, chart_height=6.5):
        super().__init__()
        self.config_1 = config_1
        self.config_2 = config_2
        self.figsize = (chart_width, chart_height)
        self.bar_width = bar_width
        self.windows = windows

    def fill_in_bars(self, ax, config, subtitle, prefixes, column):
        bars = []
        data = config.dataframe
        baseline = config.base_dataframe
        displacements = Displacements.values()
        colors = self.get_colors(len(displacements))
        ax.set_title('{}, {}'.format(subtitle, config.architecture), fontsize=14)
        prefix_index = 0
        for prefix in prefixes:
            displacement_index = 0
            for displacement_label, displacement_fn in displacements:
                bar = ax.bar(self.bar_width * displacement_index + prefix_index,
                             get_average_for_displacement(self.windows, column, data, displacement_fn, [prefix]),
                             color=colors[displacement_index],
                             label='Desplazamiento = {}'.format(displacement_label.lower()), width=self.bar_width / 1.5,
                             edgecolor='white')
                bars.append(
                    (bar, get_average_for_displacement(self.windows, column, baseline, displacement_fn, [prefix])))
                displacement_index += 1
            prefix_index += 1
        ax.set_xlabel('Serie', fontsize=12)
        ax.set_xticks([x + 0.25 for x in range(len(prefixes))])
        ax.set_xticklabels([build_legend_entry_for_animals(prefix) for prefix in prefixes])
        return bars

    def plot_dataframe(self, title, subtitle1, subtitle2, prefixes):
        super().plot(
            title,
            lambda ax: self.fill_in_bars(ax, self.config_1, subtitle1, prefixes, 'promedio'),
            lambda ax: self.fill_in_bars(ax, self.config_2, subtitle1, prefixes, 'promedio'),
            lambda ax: self.fill_in_bars(ax, self.config_1, subtitle2, prefixes, 'mejor'),
            lambda ax: self.fill_in_bars(ax, self.config_2, subtitle2, prefixes, 'mejor'),
        )


def get_average_for_displacement(windows: List[int], column: str, dataframe, displacement_fn, prefixes: List[str]):
    mcc_averages = list()
    for window in windows:
        mcc_averages.append(mean(
            dataframe[
                (dataframe['ventana'] == window) &
                (dataframe['desplazamiento'] == displacement_fn(window)) &
                (dataframe['serie'].isin(prefixes))
                ]['{} MCC'.format(column)]
        ))
    return mean(mcc_averages)


class MatrixDisplacementsChart(Matrix2x2Chart):

    def __init__(self, windows):
        super().__init__()
        self.windows = windows

    def fill_in_bars(self, ax, column, dataframe, base_dataframe, series, prefixes):
        width = 0.10
        bars = []
        ax.set_xlabel('Desplazamientos', fontsize=12)
        ax.set_xticklabels([])
        x_positions = [width * i for i in range(len(series))]
        colors = self.get_colors(len(series))
        for i in range(len(series)):
            legend_entry, legend_filter, displacement_fn = series[i]
            filtered_prefixes = [prefix for prefix in prefixes if legend_filter in prefix]
            bar = ax.bar(x_positions[i],
                         get_average_for_displacement(self.windows, column, dataframe, displacement_fn,
                                                      filtered_prefixes),
                         width=width / 1.5,
                         color=colors[i],
                         label=legend_entry)
            bars.append((bar, get_average_for_displacement(self.windows, column, base_dataframe, displacement_fn,
                                                           filtered_prefixes)))
        return bars

    def plot_dataframe(self, title, row_1_title, row_2_title, row_1_column, row_2_column,
                       config1: Matrix2x2Configuration, config2: Matrix2x2Configuration,
                       prefixes, legend_filter1, legend_filter2):
        series = self.get_legend_entries_for_filter(legend_filter1) + self.get_legend_entries_for_filter(legend_filter2)
        super().plot(
            title,
            row_1_title,
            row_2_title,
            config1,
            config2,
            lambda ax, dataframe, base_dataframe: self.fill_in_bars(ax, row_1_column, dataframe, base_dataframe, series,
                                                                    prefixes),
            lambda ax, dataframe, base_dataframe: self.fill_in_bars(ax, row_2_column, dataframe, base_dataframe, series,
                                                                    prefixes),
            lambda ax, dataframe, base_dataframe: self.fill_in_bars(ax, row_1_column, dataframe, base_dataframe, series,
                                                                    prefixes),
            lambda ax, dataframe, base_dataframe: self.fill_in_bars(ax, row_2_column, dataframe, base_dataframe, series,
                                                                    prefixes)
        )

    def get_legend_entries_for_filter(self, legend_filter):
        series = list()
        for displacement in Displacements.values():
            series.append((
                '{} [{}]'.format(get_legend_label_for_displacements_chart(legend_filter), displacement[0]),
                legend_filter,
                displacement[1]
            ))
        return series
