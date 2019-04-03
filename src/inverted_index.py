import re


class InvertedIndex:

    @staticmethod
    def create_inverted_index(data_provider):

        inverted_index = {}
        for k in data_provider.labels:
            inverted_index[k] = {}

        files = data_provider.get_files()
        word_list = []

        for msg_type in inverted_index.keys():
            for filename in files[msg_type]:
                with open(filename, encoding='latin-1') as f:
                    clean_text = re.split("[^a-zA-Z]+", f.read().lower())
                    for word in clean_text:
                        if word not in word_list:
                            word_list.append(word)
                            for k in data_provider.labels:
                                inverted_index[k][word] = 0
                        inverted_index[msg_type][word] += 1

        sorted_inverted_index = {}
        for k in data_provider.labels:
            sorted_inverted_index[k] = dict(sorted(inverted_index[k].items()))

        return sorted_inverted_index
