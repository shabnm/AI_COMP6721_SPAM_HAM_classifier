from inverted_index import InvertedIndex
import unittest


class ClassifierTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_inverted_index(self):
        InvertedIndex.create_inverted_index(self, 2, True, True)