from src.writer import WriteToExternal


class ConditionalProbability:

    def calc_probability(self, dic, ham_vocab, ham_count, spam_vocab, spam_count, smoothning_flag):
        line_num = 1
        delta = 0
        if smoothning_flag is True:
            delta = 0.5

        for vocab in dic.keys():
            ham_num = dic[vocab][0] + delta
            ham_den = ham_count + (delta * ham_vocab)

            spam_num = dic[vocab][1] + delta
            spam_den = spam_count + (delta * spam_vocab)

            ham_prob = ham_num/ham_den
            spam_prob = spam_num/spam_den

            WriteToExternal.write_to_model(self, line_num, vocab, dic[vocab][0], ham_prob,
                                           dic[vocab][1], spam_prob)
            line_num += 1


