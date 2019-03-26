from writer import WriteToExternal

class ConditionalProbability:

    # smoothed using the add δ with δ = 0.5
    #smoothed conditional probability of wi in the class ham - P(wi|ham)

    def calc_probability(self, dic, count, smoothning_flag, msg_type):
        delta = 0
        if smoothning_flag is True:
            delta = 0.5
        if msg_type is "ham":
            i = 2
        else: i = 4
        vocab_size = len(dic)
        for vocab in dic.keys():
            num = dic[vocab] + delta
            den = count + (delta*vocab_size)
            result = num/den
            WriteToExternal.write_to_model(self, i, vocab, dic[vocab], result)


