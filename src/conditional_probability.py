class ConditionalProbability:

    @staticmethod
    def calc_probability(inverted_index, smoothing_flag):
        labels = list(inverted_index.keys())
        vocabulary = inverted_index[labels[0]].keys()

        k_vocab = {}
        k_count = {}
        k_prob = {}
        for k in labels:
            k_vocab[k] = len(inverted_index[k])
            k_count[k] = sum(inverted_index[k].values())
            k_prob[k] = {w: 0 for w in vocabulary}

        delta = 0.5 if smoothing_flag else 0

        for word in vocabulary:
            for k in labels:
                num = inverted_index[k][word] + delta
                den = k_count[k] + (delta * k_vocab[k])
                k_prob[k][word] = num / den

        return k_prob
