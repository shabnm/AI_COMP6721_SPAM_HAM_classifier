from src.conditional_probability import ConditionalProbability
from src.data_provider import DataProvider
from src.inverted_index import InvertedIndex


class classifier():
    inverted_index = InvertedIndex.create_inverted_index(DataProvider())
    model = ConditionalProbability.calc_probability(inverted_index, True)
