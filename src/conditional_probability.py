class ConditionalProbability:

    @staticmethod
    def calc_probability(dic, ham_vocab, ham_count, spam_vocab, spam_count, smoothning_flag):
        line_num = 1
        delta = 0
        if smoothning_flag is True:
            delta = 0.5

        with open('../model.txt', "w") as f:
            for word in dic.keys():
                ham_num = dic[word][0] + delta
                ham_den = ham_count + (delta * ham_vocab)

                spam_num = dic[word][1] + delta
                spam_den = spam_count + (delta * spam_vocab)

                ham_prob = ham_num / ham_den
                spam_prob = spam_num / spam_den

                # def write_to_model(self, i, word, ham_word_count, ham_probability, spam_word_count, spam_probability):
                print("{}  {}  {}  {}  {}  {}".format(line_num, word, dic[word][0], ham_prob, dic[word][1], spam_prob), file=f)

                line_num += 1
