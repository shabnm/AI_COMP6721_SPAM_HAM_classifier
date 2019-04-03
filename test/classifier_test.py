from src.conditional_probability import ConditionalProbability
from src.data_provider import DataProvider
from src.inverted_index import InvertedIndex
import unittest


class ClassifierTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_inverted_index(self):
        inverted_index = InvertedIndex.create_inverted_index(DataProvider())
        model = ConditionalProbability.calc_probability(inverted_index, True)
        with open('../model.txt', "w") as f:
            print("\n".join(['  '.join(row) for row in model]), file=f)
