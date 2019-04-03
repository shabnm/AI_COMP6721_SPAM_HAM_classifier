import re
import collections
import traceback

from src.conditional_probability import ConditionalProbability


class InvertedIndex:

    @staticmethod
    def create_inverted_index(number, stop_word_flag, smoothning_flag):
        inverted_index = {
            'ham': collections.OrderedDict(),
            'spam': collections.OrderedDict(),
        }
        types = ["ham", "spam"]
        for msg_type in types:

            address = '../data/train/train-{}-'.format(msg_type)
            total_word_count = 0
            vocab_size = 0
            for n in range(1, number):
                try:
                    filename = ("{}{:05}.txt".format(address, n))
                    with open(filename, encoding='latin-1') as f:
                        clean_text = re.split("[^a-zA-Z]+", f.read().lower())
                        for word in clean_text:
                            if word not in inverted_index['ham']:
                                inverted_index['spam'][word] = 0
                                inverted_index['ham'][word] = 0
                                vocab_size += 1
                            inverted_index[msg_type][word] += 1
                            total_word_count += 1
                except Exception as ex:
                    print("Exception ", ex)
                    traceback.print_exc()
                    print(f)

        sorted_inverted_index = {
            'ham': collections.OrderedDict(sorted(inverted_index['ham'].items())),
            'spam': collections.OrderedDict(sorted(inverted_index['spam'].items())),
        }


        ConditionalProbability.calc_probability(sorted_inverted_index, smoothning_flag)
