import os

from src.data_provider import DataProvider
import unittest
from src.naive_bayes_model import NaiveBayesModel


class ClassifierTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.full_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/"
        # self.short_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data_tiny/"

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_pipeline_on_full_data(self):
        self.run_pipeline(self.full_data_dir)

    # def test_pipeline_on_short_data(self):
    #     self.run_pipeline(self.short_data_dir)

    def run_pipeline(self, datadir):
        model = NaiveBayesModel()
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability(True)
        model.save_model_to_file(datadir + '/out/model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        print('test_1_simple')
        print(cm)
        model.save_results_to_file(results, datadir + '/out/baseline-result.txt')

        ##############################################################

        with open('../English_stop_word.txt', 'r', ) as f:
            stop_words = [l.strip() for l in f.readlines()]
        model = NaiveBayesModel(stop_words=stop_words)
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability(True)
        model.save_model_to_file(datadir + '/out/stopword-model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        print('test_2_stop_words')
        print(cm)
        model.save_results_to_file(results, datadir + '/out/stopword-result.txt')

        ##############################################################

        model = NaiveBayesModel(remove_this_or_shorter=2, remove_this_or_longer=9)
        model.create_inverted_index(DataProvider(datadir, source='train'))
        model.calc_probability(True)
        model.save_model_to_file(datadir + '/out/wordlength-model.txt')
        results, cm = model.inference(DataProvider(datadir, source='test'))
        print('test_3_length')
        print(cm)
        model.save_results_to_file(results, datadir + '/out/wordlength-result.txt')
