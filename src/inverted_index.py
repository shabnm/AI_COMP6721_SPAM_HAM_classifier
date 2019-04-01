import re
import collections
from conditional_probability import ConditionalProbability
from file_path import File_Path


class InvertedIndex:

    def create_inverted_index(self, number, stop_word_flag, smoothning_flag):
        ham_vocab = 0
        spam_vocab = 0
        ham_word_count = 0
        spam_word_count = 0
        inverted_index = collections.defaultdict(list)
        sorted_inverted_index = collections.OrderedDict()
        type = ["ham", "spam"]
        for msg_type in type:
            if msg_type is "ham": i = 0
            else: i = 1

            address = File_Path.TRAIN + msg_type + "-"
            total_word_count = 0
            vocab_size = 0
            for n in range(1, number):
                try:
                    filename = ("{}{:05}.txt".format(address, n))
                    with open(filename) as f:
                        clean_text = f.read()
                        clean_text = clean_text.lower()
                        clean_text = re.split("[^a-zA-Z]+", clean_text)
                        for vocab in clean_text:
                            if vocab not in inverted_index.keys():
                                if msg_type=="ham":
                                    inverted_index[vocab].append(1)
                                    inverted_index[vocab].append(0)
                                else:
                                    inverted_index[vocab].append(0)
                                    inverted_index[vocab].append(1)
                                vocab_size += 1
                            else:
                                if inverted_index[vocab][i]==0:
                                    vocab_size += 1
                                inverted_index[vocab][i] += 1
                            total_word_count += 1
                    f.close()
                except Exception as ex:
                    print("Exception ",ex)

            if msg_type is "ham":
                ham_vocab = vocab_size
                ham_word_count = total_word_count
            else:
                spam_vocab = vocab_size
                spam_word_count = total_word_count

            sort_inverted_index = sorted(inverted_index)
            for term in sort_inverted_index:
                sorted_inverted_index[term] = inverted_index[term]

        ConditionalProbability.calc_probability(self, sorted_inverted_index, ham_vocab,
                                                      ham_word_count, spam_vocab,
                                                      spam_word_count, smoothning_flag)

