import os

from src.data_provider import DataProvider
import unittest
from src.naive_bayes_model import NaiveBayesModel


class ClassifierTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.full_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/"
        self.short_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data_tiny/"

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_pipeline_on_full_data(self):
        self.run_pipeline(self.full_data_dir)

    def test_pipeline_on_short_data(self):
        self.run_pipeline(self.short_data_dir)

    def run_pipeline(self, datadir):
        ##############################################################

        model = NaiveBayesModel(smoothing=0.5)
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability()
        model.save_model_to_file(datadir + '/out/model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        print('test_1_simple')
        print(cm)
        model.save_results_to_file(results, datadir + '/out/baseline-result.txt')

        ##############################################################

        with open('../English_stop_word.txt', 'r', ) as f:
            stop_words = [l.strip() for l in f.readlines()]
        model = NaiveBayesModel(smoothing=0.5, stop_words=stop_words)
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability()
        model.save_model_to_file(datadir + '/out/stopword-model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        print('test_2_stop_words')
        print(cm)
        model.save_results_to_file(results, datadir + '/out/stopword-result.txt')

        ##############################################################

        model = NaiveBayesModel(smoothing=0.5, min_len_filter=2, max_len_filter=9)
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability()
        model.save_model_to_file(datadir + '/out/wordlength-model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        print('test_3_length')
        print(cm)
        model.save_results_to_file(results, datadir + '/out/wordlength-result.txt')

        ##############################################################

        for f in [1, 5, 10, 15, 20]:
            model = NaiveBayesModel(smoothing=0.5, cutoff_frequency=f)
            model.create_inverted_index(DataProvider(datadir, source='train'))
            model.calc_probability()
            model.save_model_to_file(datadir + '/out/wordlowfreq_{}-model.txt'.format(f))
            results, cm = model.inference(DataProvider(datadir, source='test'))
            print('test_4_frequency_{}'.format(f))
            print(cm)
            model.save_results_to_file(results, datadir + '/out/wordlowfreq_{}-result.txt'.format(f))

        ##############################################################

        for s in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            model = NaiveBayesModel(smoothing=s)
            model.create_inverted_index(DataProvider(datadir, source='train'))
            model.calc_probability()
            model.save_model_to_file(datadir + '/out/smoothed_{}-model.txt'.format(s))
            results, cm = model.inference(DataProvider(datadir, source='test'))
            print('test_5_smoothed_{}'.format(s))
            print(cm)
            model.save_results_to_file(results, datadir + '/out/smoothed_{}-result.txt'.format(s))

        ##############################################################
