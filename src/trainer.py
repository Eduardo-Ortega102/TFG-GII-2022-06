import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class KFoldTrainer:

    def __init__(self, model_function, amount_of_folds=10):
        self.model_function = model_function
        self.amount_of_folds = amount_of_folds
        self.model_list = list()
        self.best_model = -1
        self.loss_list = list()

    def get_best_model(self):
        print('Best Model: {}'.format(self.best_model))
        return self.model_list[self.best_model][0]

    def get_all_models(self):
        return [model_tuple[0] for model_tuple in self.model_list]

    def get_all_scores(self):
        return [model_tuple[1] for model_tuple in self.model_list]

    def save_learning_curve(self, file_to_save, model_id):
        loss, validation_loss = self.loss_list[model_id]
        self.__plot_learning_curve('loss', loss, validation_loss, model_id, file_to_save)

    def train(self, training_values, training_labels):
        iteration = 0
        best_score = np.inf

        for train_index, test_index in KFold(self.amount_of_folds, shuffle=False).split(training_values):
            X_train_k, y_train_k = training_values[train_index], training_labels[train_index]
            X_validation, y_validation = training_values[test_index], training_labels[test_index]

            print('Iter. {}: {} elements'.format(iteration, len(test_index)))

            model = self.model_function()
            history = model.fit(
                X_train_k,
                y_train_k,
                batch_size=32,
                epochs=200,
                shuffle=False,
                validation_data=(X_validation, y_validation),
                verbose=0
            )
            score = model.evaluate(X_validation, y_validation, verbose=0)
            print('Score {}: {}'.format(iteration, score))
            self.model_list.append((model, score))
            self.loss_list.append((history.history['loss'], history.history['val_loss']))

            if score < best_score:
                best_score = score
                self.best_model = iteration
            iteration += 1

    def __plot_learning_curve(self, name, training_values, validation_values, iteration, file_to_save=None):
        fig, axes = plt.subplots(constrained_layout=True, figsize=(13, 5))
        fig.suptitle('Learning Curve Model {}'.format(iteration))
        axes.set_ylabel(name)
        axes.set_xlabel('epoch')
        axes.plot(training_values, label='train')
        axes.plot(validation_values, label='validation')
        axes.legend(loc='upper right')
        if file_to_save is not None:
            fig.savefig('{}.LC.png'.format(file_to_save))
        else:
            fig.show()
        plt.close('all')
