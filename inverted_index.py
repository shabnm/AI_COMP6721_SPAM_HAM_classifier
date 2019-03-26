import re
import collections
from conditional_probability import ConditionalProbability


class InvertedIndex:


    def create_inverted_index(self, number, msg_type, stop_word_flag, smoothning_flag):
        address = "C:\\Users\\admin\\PycharmProjects\\untitled\\Project2-Train\\train\\train-"+msg_type+"-"
        inverted_index = {}
        sorted_inverted_index = collections.OrderedDict()
        total_word_count = 0
        for n in range(1, number):
            try:
                filename = ("{}{:05}.txt".format(address, n))
                with open(filename) as f:
                    clean_text = f.read()
                    clean_text = clean_text.lower()
                    clean_text = re.split("[^a-zA-Z]+", clean_text)
                    for vocab in clean_text:
                        if vocab not in inverted_index.keys():
                            inverted_index[vocab] = 1
                        else:
                            inverted_index[vocab] = inverted_index[vocab]+1
                        total_word_count += 1
                f.close()
            except Exception as ex:
                print(ex)
        sort_inverted_index = sorted(inverted_index)
        for term in sort_inverted_index:
            sorted_inverted_index[term] = inverted_index[term]
        print(ConditionalProbability.calc_probability(self, sorted_inverted_index, total_word_count,
                                                      smoothning_flag, msg_type))

