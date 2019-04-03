class ConditionalProbability:

    @staticmethod
    def calc_probability(sorted_inverted_index, smoothing_flag):
        k_vocab = {}
        k_count = {}
        for k in sorted_inverted_index.keys():
            k_vocab[k] = len(sorted_inverted_index[k])
            k_count[k] = sum(sorted_inverted_index[k].values())

        delta = 0.5 if smoothing_flag else 0

        line_num = 0
        outputs = []
        for word in sorted_inverted_index[list(sorted_inverted_index.keys())[0]].keys():
            line_num += 1
            output = [str(line_num), word]
            for k in sorted_inverted_index.keys():
                num = sorted_inverted_index[k][word] + delta
                den = k_count[k] + (delta * k_vocab[k])
                prob = num / den
                output.append(str(sorted_inverted_index[k][word]))
                output.append(str(prob))
            outputs.append(output)
        return outputs
