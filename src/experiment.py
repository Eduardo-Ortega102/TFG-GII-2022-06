from trainer import KFoldTrainer
from reporter import MLPReporter, AutoencoderReporter
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import os
import shutil
import numpy as np
import pandas
import matplotlib.pyplot as plt
from datetime import datetime
from abc import ABC, abstractmethod
from utilities import calculate_reconstruction_error, point_by_point_error


class AbstractExperiment(ABC):

    def __init__(self, id, content_directory, window_length, skip_training, objective_function='binary_crossentropy'):
        self.id = id
        self.content_directory = content_directory
        self.window_length = window_length
        self.objective_function = objective_function
        self.skip_training = skip_training

    def execute_single(self, signal_file, model_function, displacement):
        print('\nTimeStamp: {}'.format(datetime.now()))
        X, y = self.load_data(signal_file, displacement)

        X_train, X_test, y_train, y_test = self.split_train_test(X, y, percentage=20)
        model_function_wrapper = lambda: model_function(input_dimension=self.window_length,
                                                        objective_function=self.objective_function)
        print('Cantidad de anomalías (positivos): {}'.format(np.sum(y_test)))

        experiment_name = self.get_iteration_name(displacement)
        print(experiment_name)
        os.mkdir(experiment_name)
        path = os.path.join(experiment_name, experiment_name)
        dataframe = pandas.DataFrame(columns=['ventana', 'desplazamiento', 'id_modelo', 'TP', 'FN', 'TN', 'FP', 'MCC'])

        if not self.skip_training:
            trainer = KFoldTrainer(model_function_wrapper)
            self.train(trainer, X_train, y_train)
            for model_id, model in enumerate(trainer.get_all_models()):
                basename = '{}_MOD{}'.format(path, model_id)
                trainer.save_learning_curve(basename, model_id)
                self.save_model_data(X_test, basename, dataframe, displacement, model, model_id, y_test)
        else:
            for model_id in range(10):
                basename = '{}_BASE{}'.format(path, model_id)
                model = model_function_wrapper()
                self.save_model_data(X_test, basename, dataframe, displacement, model, model_id, y_test)

        dataframe.to_csv('{}.csv'.format(path), index=False, sep=';', decimal=',')
        np.savez('{}_test.npz'.format(path), X_test=X_test, y_test=y_test)
        zip_file_name = '{}.zip'.format(experiment_name)
        with ZipFile(zip_file_name, 'w') as zip_file:
            for _, _, filenames in os.walk(experiment_name):
                for filename in filenames:
                    zip_file.write(os.path.join(experiment_name, filename))
        shutil.rmtree(experiment_name)
        shutil.move(zip_file_name, os.path.join(self.content_directory, zip_file_name))

    @abstractmethod
    def train(self, trainer: KFoldTrainer, X_train, y_train):
        pass

    @abstractmethod
    def save_model_data(self, X_test, basename, dataframe, displacement, model, model_id, y_test):
        pass

    def save_model_data_internal(self, basename, dataframe, displacement, model, model_id, confusion_matrix, weights_only=False):
        if weights_only:
            model.save_weights('{}.weights'.format(basename), save_format='tf')
        else:
            model.save('{}.h5'.format(basename))
        dataframe.loc[model_id] = [self.window_length, displacement, model_id] + confusion_matrix

    def load_data(self, signal_file, displacement):
        serieDataFrame = pandas.read_csv(signal_file)
        windows, labels = self.load_windows(
            self.interpolate(serieDataFrame['data'].values),
            serieDataFrame['label'].values,
            displacement
        )
        print('{} ventanas y {} etiquetas'.format(len(windows), len(labels)))
        print('{} ventanas anomalas y {} ventanas normales'.format(np.sum(labels), len(labels) - np.sum(labels)))
        return windows, labels

    def load_windows(self, data, labels, displacement):
        windows = list()
        windows_labels = list()
        for start in range(0, len(data) - self.window_length, displacement):
            end = start + self.window_length
            windows.append(data[start:end])
            windows_labels.append(1 if 1 in labels[start:end] else 0)
        return np.array(windows), np.array(windows_labels)

    def interpolate(self, data, minimum=-1, maximum=1):
        if data.min() == minimum and data.max() == maximum:
            return data
        print('min = {}'.format(data.min()))
        print('max = {}'.format(data.max()))
        return np.interp(data, (data.min(), data.max()), (minimum, maximum))

    def plot_window(self, basename, data, title):
        fig, axes = plt.subplots()
        axes.plot(data, 'bo')
        axes.set_title(title)
        axes.set_xlabel('tiempo')
        axes.set_ylabel('señal')
        axes.grid(True)
        fig.savefig('{}_{}.plot.png'.format(basename, title.replace(' ', '_')))
        plt.close('all')

    def split_train_test(self, data, labels, percentage):
        data_train, data_test, labels_train, labels_test = train_test_split(
            data,
            labels,
            test_size=(percentage / 100),
            shuffle=False
        )
        print('Test shape: {}'.format(np.array(data_test).shape))
        print('Train shape: {}'.format(np.array(data_train).shape))
        print('Conjunto de test = {}, Etiquetas = {}. Ventanas anómalas = {}.'.format(
            len(data_test),
            len(labels_test),
            np.sum(labels_test))
        )
        print('Conjunto de entrenamiento = {}, Etiquetas = {}. Ventanas anómalas = {}.'.format(
            len(data_train),
            len(labels_train),
            np.sum(labels_train))
        )
        return data_train, data_test, labels_train, labels_test

    def get_iteration_name(self, displacement):
        return '#{}_Wind{}_Displ{}_OF{}'.format(
            self.id,
            self.window_length,
            displacement,
            self.objective_function
        )


class MLPExperiment(AbstractExperiment):

    def train(self, trainer: KFoldTrainer, X_train, y_train):
        trainer.train(X_train, y_train)

    def save_model_data(self, X_test, basename, dataframe, displacement, model, model_id, y_test):
        reporter = MLPReporter(['normal', 'anomaly'], y_test, model.predict(X_test))
        # reporter.plot_confusion_matrix()
        confusion_matrix = reporter.get_confusion_matrix()
        self.save_model_data_internal(basename, dataframe, displacement, model, model_id, confusion_matrix)


class AutoencoderExperiment(AbstractExperiment):

    def train(self, trainer: KFoldTrainer, X_train, y_train):
        clean_data = X_train[np.where(y_train == 0)]
        trainer.train(clean_data, clean_data)

    def save_model_data(self, x_test, basename, dataframe, displacement, model, model_id, y_test):
        x_reconstructed = model.predict(x_test)
        # self.debug_misclassification(basename, x_reconstructed, x_test, y_test)
        reporter = AutoencoderReporter(['normal', 'anomaly'], x_test, y_test, x_reconstructed)
        # reporter.plot_confusion_matrix()
        confusion_matrix = reporter.get_confusion_matrix()
        self.save_model_data_internal(basename, dataframe, displacement, model, model_id, confusion_matrix, weights_only=True)

    def debug_misclassification(self, basename, x_reconstructed, x_test, y_test):
        anomaly_indexes = np.where(y_test == 1)
        test_window = x_test[anomaly_indexes][0]
        reconstructed_window = x_reconstructed[anomaly_indexes][0]
        self.plot_window(basename, test_window, 'Ventana anomala original')
        self.plot_window(basename, reconstructed_window, 'Ventana anomala reconstruida')
        self.plot_window(basename, point_by_point_error(test_window, reconstructed_window), 'Errores punto a punto')
        reconstruction_error = calculate_reconstruction_error(test_window, reconstructed_window)
        print('reconstruction_error: {}'.format(reconstruction_error))
        self.plot_window(basename, reconstruction_error, 'MSE ventana')
