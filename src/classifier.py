import math

from src.conditional_probability import ConditionalProbability
from src.data_provider import DataProvider
from src.inverted_index import InvertedIndex


class Classifier:

    @staticmethod
    def classify(k_prob, word_count, smoothing):
        labels=list(k_prob.keys())
        results={}

        for label in labels:
            result = 0
            for word in word_count.keys():
                if word in k_prob[label]:
                    result+=math.log(word_count[word] * k_prob[label][word])
            results[label] = result
        result_label = labels[0]
        for label in labels:
            if results[label]> results[result_label]:
                result_label = label
        return result_label , results