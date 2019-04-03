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
