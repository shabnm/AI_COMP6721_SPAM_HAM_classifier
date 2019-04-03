class ConditionalProbability:

    @staticmethod
    def calc_probability(sorted_inverted_index, smoothning_flag):

        ham_vocab = len(sorted_inverted_index['ham'])
        ham_count = sum(sorted_inverted_index['ham'].values())
        spam_vocab = len(sorted_inverted_index['spam'])
        spam_count = sum(sorted_inverted_index['spam'].values())

        delta = 0
        if smoothning_flag is True:
            delta = 0.5

        line_num = 1
        with open('../model.txt', "w") as f:
            for word in sorted_inverted_index['ham'].keys():
                ham_num = sorted_inverted_index['ham'][word] + delta
                ham_den = ham_count + (delta * ham_vocab)

                spam_num = sorted_inverted_index['spam'][word] + delta
                spam_den = spam_count + (delta * spam_vocab)

                ham_prob = ham_num / ham_den
                spam_prob = spam_num / spam_den

                # def write_to_model(self, i, word, ham_word_count, ham_probability, spam_word_count, spam_probability):
                print("{}  {}  {}  {}  {}  {}".format(line_num, word, sorted_inverted_index['ham'][word], ham_prob, sorted_inverted_index['spam'][word], spam_prob), file=f)

                line_num += 1
