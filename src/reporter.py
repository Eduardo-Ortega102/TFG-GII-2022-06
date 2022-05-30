import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from abc import ABC, abstractmethod
from utilities import calculate_reconstruction_error


class AbstractReporter(ABC):

    def __init__(self, classes, ground_truth):
        self.classes = classes
        self.ground_truth = ground_truth
        self.threshold = None
        self.threshold_score = None

    def get_confusion_matrix(self):
        tn, fp, fn, tp = confusion_matrix(
            self.ground_truth,
            self.make_predictions(self.threshold),
            normalize='true'
        ).ravel()
        return [tp, fn, tn, fp, self.threshold_score]

    @abstractmethod
    def make_predictions(self, threshold):
        pass

    def plot_confusion_matrix(self, file_to_save=None, normalize='true'):
        df_cm = pd.DataFrame(
            confusion_matrix(self.ground_truth, self.make_predictions(self.threshold), normalize=normalize),
            index=self.classes,
            columns=self.classes
        )
        plt.figure()
        plt.title('Confusion Matrix (MCC = {:.3f})'.format(self.threshold_score))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=('d' if normalize is None else '.3f'))  # font size
        if file_to_save is not None:
            plt.savefig('{}_CMatrix.png'.format(file_to_save), bbox_inches='tight')
        else:
            plt.show()
        plt.close('all')


class MLPReporter(AbstractReporter):

    def __init__(self, classes, ground_truth, probabilities):
        super().__init__(classes, ground_truth)
        self.probabilities = probabilities
        self.threshold, self.threshold_score = self.__calculate_best_threshold()

    def __calculate_best_threshold(self):
        thresholds = np.unique(self.probabilities)
        mcc_scores = [
            matthews_corrcoef(self.ground_truth, self.make_predictions(threshold)) for threshold in thresholds
        ]
        index = np.argmax(mcc_scores)
        return thresholds[index], mcc_scores[index]

    def make_predictions(self, threshold):
        return [0 if probability[0] < threshold else 1 for probability in self.probabilities]


class AutoencoderReporter(AbstractReporter):

    def __init__(self, classes, original_data, ground_truth, reconstructed_data):
        super().__init__(classes, ground_truth)
        self.reconstruction_errors = calculate_reconstruction_error(original_data, reconstructed_data)
        self.threshold, self.threshold_score = self.__calculate_threshold()

    def __calculate_threshold(self):
        thresholds = list()
        mcc_scores = list()
        for percentile in np.arange(80, 100):
            threshold = np.percentile(self.reconstruction_errors, percentile)
            score = matthews_corrcoef(self.ground_truth, self.make_predictions(threshold))
            thresholds.append(threshold)
            mcc_scores.append(score)
        index = np.argmax(mcc_scores)
        return thresholds[index], mcc_scores[index]

    def make_predictions(self, threshold):
        return [0 if error < threshold else 1 for error in self.reconstruction_errors]
