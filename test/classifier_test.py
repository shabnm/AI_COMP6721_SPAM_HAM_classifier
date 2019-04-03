import re

from src.classifier import Classifier
from src.conditional_probability import ConditionalProbability
from src.data_provider import DataProvider
from src.inverted_index import InvertedIndex
import unittest

from src.model_dumper import ModelDumper


class ClassifierTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_inverted_index(self):
        inverted_index = InvertedIndex.create_inverted_index(DataProvider())
        k_prob = ConditionalProbability.calc_probability(inverted_index, True)
        ModelDumper.dump(inverted_index, k_prob)

        provider = DataProvider()
        files = provider.get_files('test')
        for label in files:
            for file in files[label]:
                word_count = {}
                with open(file, encoding='latin-1', mode='r') as f:
                    clean_text = re.split("[^a-zA-Z]+", f.read().lower())
                    for word in clean_text:
                        if word not in word_count:
                            word_count[word] = 0
                        word_count[word] += 1
                    result_label, probs = Classifier.classify(k_prob, word_count, False)
                    print(label,result_label, probs)
