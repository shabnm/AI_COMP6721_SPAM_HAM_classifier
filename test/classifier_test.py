import os

from src.data_provider import DataProvider
import unittest
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

        print('Data', datadir)
        ##############################################################

        print('experiment 1 baseline')
        model = NaiveBayesModel(smoothing=0.5)
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability()
        model.save_model_to_file(datadir + '/out/baseline-model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        print(cm)
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
        print(cm)
        model.save_results_to_file(results, datadir + '/out/stopword-result.txt')

        ##############################################################

        print('experiment 3 length')
        model = NaiveBayesModel(smoothing=0.5, min_len_filter=2, max_len_filter=9)
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability()
        model.save_model_to_file(datadir + '/out/wordlength-model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        print(cm)
        model.save_results_to_file(results, datadir + '/out/wordlength-result.txt')

        ##############################################################

        for f in [1, 5, 10, 15, 20]:
            print('experiment 4 low frequency_{}'.format(f))
            model = NaiveBayesModel(smoothing=0.5, cutoff_low_count=f)
            model.create_inverted_index(DataProvider(datadir, source='train'))
            model.calc_probability()
            results, cm = model.inference(DataProvider(datadir, source='test'))
            print(cm)

        for f in [0.05, 0.1, 0.15, 0.2, 0.25]:
            print('experiment 4 high frequency_{}'.format(f))
            model = NaiveBayesModel(smoothing=0.5, cutoff_top_frequent_words_fraction=f)
            model.create_inverted_index(DataProvider(datadir, source='train'))
            model.calc_probability()
            results, cm = model.inference(DataProvider(datadir, source='test'))
            print(cm)

        ##############################################################

        for s in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            print('experiment 5 smoothed_{}'.format(s))
            model = NaiveBayesModel(smoothing=s)
            model.create_inverted_index(DataProvider(datadir, source='train'))
            model.calc_probability()
            results, cm = model.inference(DataProvider(datadir, source='test'))
            print(cm)

        ##############################################################


if __name__ == "__main__":
    unittest.main()
