import re
import collections
import traceback

from src.conditional_probability import ConditionalProbability


class InvertedIndex:

    @staticmethod
    def create_inverted_index(number, stop_word_flag, smoothning_flag):
        ham_vocab = 0
        spam_vocab = 0
        ham_word_count = 0
        spam_word_count = 0
        inverted_index = {
            'ham': collections.defaultdict(list),
            'spam': collections.defaultdict(list),
        }
        sorted_inverted_index = {
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

            print(inverted_index)

            if msg_type is "ham":
                ham_vocab = vocab_size
                ham_word_count = total_word_count
            else:
                spam_vocab = vocab_size
                spam_word_count = total_word_count

            sorted_inverted_index['ham'] = sorted(inverted_index['ham'])
            sorted_inverted_index['spam'] = sorted(inverted_index['spam'])

            # for term in sorted_inverted_index:
            #     sorted_inverted_index[term] = inverted_index[term]

        ConditionalProbability.calc_probability(sorted_inverted_index,
                                                len(sorted_inverted_index['ham']), sum(sorted_inverted_index['ham'].values()),
                                                len(sorted_inverted_index['spam']), sum(sorted_inverted_index['spam'].values()), smoothning_flag)
