from os import path
from experiment import MLPExperiment, AutoencoderExperiment
from models import create_MLP2_model, create_MLP3_model, create_AE_model, create_VAE_model

G_DRIVE = '/content/drive/My Drive/TFG/results_9'


def run_MLP2_experiment(data_file, prefix, window_length, displacement, h1_units, rate, skip_training=False):
    __run_MLP_experiment_internal(
        data_file,
        prefix,
        window_length,
        displacement,
        __create_MLP2_model_wrapper(h1_units, rate),
        '-L{}-D{}-'.format(h1_units, rate),
        skip_training
    )


def run_MLP3_experiment(data_file, prefix, window_length, displacement, h1_units, h2_units, rate, skip_training=False):
    __run_MLP_experiment_internal(
        data_file,
        prefix,
        window_length,
        displacement,
        __create_MLP3_model_wrapper(h1_units, h2_units, rate),
        '-L{}-M{}-D{}-'.format(h1_units, h2_units, rate),
        skip_training
    )


def run_AE_experiment(data_file, prefix, window_length, displacement, intermediate, bottleneck, skip_training=False):
    __run_AE_experiment_internal(
        data_file,
        prefix,
        window_length,
        displacement,
        __create_AE_model_wrapper(intermediate, bottleneck),
        '-IN{}-BO{}-'.format(intermediate, bottleneck),
        skip_training
    )


def run_VAE_experiment(data_file, prefix, window_length, displacement, intermediate, bottleneck, skip_training=False):
    __run_AE_experiment_internal(
        data_file,
        prefix,
        window_length,
        displacement,
        __create_VAE_model_wrapper(intermediate, bottleneck),
        '-IN{}-BO{}-'.format(intermediate, bottleneck),
        skip_training
    )


def __run_MLP_experiment_internal(data_file, prefix, window_length, displacement, model_function, sufix, skip_training):
    validateStorageExists()
    MLPExperiment(
        id='{}{}'.format(prefix, sufix),
        content_directory=G_DRIVE,
        window_length=window_length,
        skip_training=skip_training
    ).execute_single(
        signal_file=data_file,
        model_function=model_function,
        displacement=displacement
    )


def __run_AE_experiment_internal(data_file, prefix, window_length, displacement, model_function, sufix, skip_training):
    validateStorageExists()
    AutoencoderExperiment(
        id='{}{}'.format(prefix, sufix),
        content_directory=G_DRIVE,
        window_length=window_length,
        skip_training=skip_training,
        objective_function='mean_squared_error'
    ).execute_single(
        signal_file=data_file,
        model_function=model_function,
        displacement=displacement
    )


def validateStorageExists():
    if not path.exists(G_DRIVE):
        raise Exception('The storage does not exist!')


def __create_MLP2_model_wrapper(h1_units, rate):
    return (lambda input_dimension, objective_function:
            create_MLP2_model(
                input_dimension=input_dimension,
                objective_function=objective_function,
                hidden1_units=h1_units,
                dropout_rate=rate
            )
            )


def __create_MLP3_model_wrapper(h1_units, h2_units, rate):
    return (lambda input_dimension, objective_function:
            create_MLP3_model(
                input_dimension=input_dimension,
                objective_function=objective_function,
                hidden1_units=h1_units,
                hidden2_units=h2_units,
                dropout_rate=rate
            )
            )


def __create_AE_model_wrapper(intermediate, bottleneck):
    return (lambda input_dimension, objective_function:
            create_AE_model(
                input_dimension=input_dimension,
                intermediate=intermediate,
                bottleneck=bottleneck,
                objective_function=objective_function
            )
            )


def __create_VAE_model_wrapper(intermediate, bottleneck):
    return (lambda input_dimension, objective_function:
            create_VAE_model(
                input_dimension=input_dimension,
                intermediate=intermediate,
                bottleneck=bottleneck
            )
            )
