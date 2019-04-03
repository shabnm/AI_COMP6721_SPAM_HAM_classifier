class ModelDumper:

    @staticmethod
    def dump(inverted_index, k_prob):
        labels = list(inverted_index.keys())
        vocabulary = inverted_index[labels[0]].keys()

        line_num = 0
        outputs = []
        for word in vocabulary:
            line_num += 1
            output = [str(line_num), word]
            for k in labels:
                output.append(str(inverted_index[k][word]))
                output.append(str(k_prob[k][word]))
            outputs.append(output)

        with open('../model.txt', "w") as f:
            print("\n".join(['  '.join(row) for row in outputs]), file=f)
