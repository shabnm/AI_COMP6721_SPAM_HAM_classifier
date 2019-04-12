import os
import unittest
import matplotlib.pyplot as plt

from src.data_provider import DataProvider
from src.naive_bayes_model import NaiveBayesModel


class ClassifierTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.full_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/"
        self.short_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data_tiny/"
        self.short2_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data_tiny2/"

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_pipeline_on_full_data(self):
        self.run_pipeline(self.full_data_dir)

    # def test_pipeline_on_short_data(self):
    #     self.run_pipeline(self.short_data_dir)
    #
    # def test_pipeline_on_short2_data(self):
    #     self.run_pipeline(self.short2_data_dir)
    #

    def run_pipeline(self, datadir):

        markers_ham = {
            'Accuracy': 'o',
            'Precision': 's',
            'Recall': '^',
            'F1': 'D',
        }
        markers_spam = {
            'Accuracy': 'o',
            'Precision': 'x',
            'Recall': 'v',
            'F1': '*',
        }
        print('Data', datadir)
        ##############################################################

        print('experiment 1 baseline')
        model = NaiveBayesModel(smoothing=0.5)
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability()
        model.save_model_to_file(datadir + '/out/baseline-model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        self.print_cm(cm)
        model.save_results_to_file(results, datadir + '/out/baseline-result.txt')

        ##############################################################

        print('experiment 2 stop_words')
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../English_stop_word.txt', 'r', ) as f:
            stop_words = [l.strip() for l in f.readlines()]
        model = NaiveBayesModel(smoothing=0.5, stop_words=stop_words)
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability()
        model.save_model_to_file(datadir + '/out/stopword-model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        self.print_cm(cm)
        model.save_results_to_file(results, datadir + '/out/stopword-result.txt')

        ##############################################################

        print('experiment 3 length')
        model = NaiveBayesModel(smoothing=0.5, min_len_filter=2, max_len_filter=9)
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability()
        model.save_model_to_file(datadir + '/out/wordlength-model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        self.print_cm(cm)
        model.save_results_to_file(results, datadir + '/out/wordlength-result.txt')

        ##############################################################

        plot_x = []
        plot_y_ham = {}
        plot_y_spam = {}
        freq_to_remove = [0, 1, 5, 10, 15, 20]
        for f in freq_to_remove:
            plot_x.append(f)
            print('experiment 4 low frequency_{}'.format(f))
            model = NaiveBayesModel(smoothing=0.5, cutoff_low_count=f)
            model.create_inverted_index(DataProvider(datadir, source='train'))
            model.calc_probability()
            results, cm = model.inference(DataProvider(datadir, source='test'))
            self.print_cm(cm)
            if len(plot_y_ham) == 0:
                plot_y_ham = {k: [] for k, v in cm['ham'].items()}
                plot_y_spam = {k: [] for k, v in cm['spam'].items()}
            for k, v in cm['ham'].items():
                plot_y_ham[k].append(v)
            for k, v in cm['spam'].items():
                plot_y_spam[k].append(v)
        plt.grid()
        plt.xticks(freq_to_remove)
        # plt.yticks([0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
        plt.xlabel("Word frequency to remove")
        # plt.ylabel("F1")
        labels = ['Accuracy', 'Precision', 'Recall', 'F1']
        legends = []
        for k in labels:
            legend = "ham {}".format(k)
            plt.plot(plot_x, plot_y_ham[k], marker=markers_ham[k], label=legend)
            legends.append(legend)
        for k in labels:
            legend = "spam {}".format(k)
            plt.plot(plot_x, plot_y_spam[k], '--', marker=markers_spam[k], label=legend)
            legends.append(legend)
        plt.legend(legends, bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()

        plot_x = []
        plot_y_ham = {}
        plot_y_spam = {}
        fraq_to_remove = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        for f in fraq_to_remove:
            plot_x.append(f)
            print('experiment 4 high frequency_{}'.format(f))
            model = NaiveBayesModel(smoothing=0.5, cutoff_top_frequent_words_fraction=f)
            model.create_inverted_index(DataProvider(datadir, source='train'))
            model.calc_probability()
            results, cm = model.inference(DataProvider(datadir, source='test'))
            self.print_cm(cm)
            if len(plot_y_ham) == 0:
                plot_y_ham = {k: [] for k, v in cm['ham'].items()}
                plot_y_spam = {k: [] for k, v in cm['spam'].items()}
            for k, v in cm['ham'].items():
                plot_y_ham[k].append(v)
            for k, v in cm['spam'].items():
                plot_y_spam[k].append(v)

        plt.grid()
        plt.xticks(fraq_to_remove)
        # plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
        plt.xlabel("Fraction of most frequent words removed")
        # plt.ylabel("F1")
        labels = ['Accuracy', 'Precision', 'Recall', 'F1']
        legends = []
        for k in labels:
            legend = "ham {}".format(k)
            plt.plot(plot_x, plot_y_ham[k], marker=markers_ham[k], label=legend)
            legends.append(legend)
        for k in labels:
            legend = "spam {}".format(k)
            plt.plot(plot_x, plot_y_spam[k], '--', marker=markers_spam[k], label=legend)
            legends.append(legend)
        plt.legend(legends, bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()

        ##############################################################

        plot_x = []
        plot_y_ham = {}
        plot_y_spam = {}
        smoothing = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for s in smoothing:
            plot_x.append(s)
            print('experiment 5 smoothed_{}'.format(s))
            model = NaiveBayesModel(smoothing=s)
            model.create_inverted_index(DataProvider(datadir, source='train'))
            model.calc_probability()
            results, cm = model.inference(DataProvider(datadir, source='test'))
            self.print_cm(cm)
            if len(plot_y_ham) == 0:
                plot_y_ham = {k: [] for k, v in cm['ham'].items()}
                plot_y_spam = {k: [] for k, v in cm['spam'].items()}
            for k, v in cm['ham'].items():
                plot_y_ham[k].append(v)
            for k, v in cm['spam'].items():
                plot_y_spam[k].append(v)

        plt.grid()
        plt.xticks(smoothing)
        # plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
        plt.xlabel("Smoothing coefficient")
        # plt.ylabel("F1")
        labels = ['Accuracy', 'Precision', 'Recall', 'F1']
        legends = []
        for k in labels:
            legend = "ham {}".format(k)
            plt.plot(plot_x, plot_y_ham[k], marker=markers_ham[k], label=legend)
            legends.append(legend)
        for k in labels:
            legend = "spam {}".format(k)
            plt.plot(plot_x, plot_y_spam[k], '--', marker=markers_spam[k], label=legend)
            legends.append(legend)
        plt.legend(legends, bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()

        plt.grid()
        plt.xticks(smoothing)
        # plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
        plt.xlabel("Smoothing coefficient")
        # plt.ylabel("F1")
        labels = ['Accuracy', 'Precision', 'Recall', 'F1']
        legends = []
        for k in labels:
            legend = "ham {}".format(k)
            plt.plot(plot_x[1:], plot_y_ham[k][1:], marker=markers_ham[k], label=legend)
            legends.append(legend)
        for k in labels:
            legend = "spam {}".format(k)
            plt.plot(plot_x[1:], plot_y_spam[k][1:], '--', marker=markers_spam[k], label=legend)
            legends.append(legend)
        plt.legend(legends, bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()

        ##############################################################

    def print_cm(self, cm):
        row_labels = list(cm.keys())
        col_labels = list(cm[row_labels[0]].keys())
        print('\t'.join(['label'] + col_labels))
        for r in row_labels:
            print('\t'.join(["{0:.4f}".format(s) if type(s) == float else str(s) for s in ["{0: <4}".format(r)] + list(cm[r].values())]))
        # print()


if __name__ == "__main__":
    unittest.main()
