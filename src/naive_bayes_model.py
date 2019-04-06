import os
import re
import math


class NaiveBayesModel:

    def __init__(self,
                 stop_words=None,
                 smoothing=0,
                 min_len_filter=0,
                 max_len_filter=math.inf,
                 cutoff_low_count=0,
                 cutoff_top_frequent_words_fraction=0.0
                 ) -> None:
        self.inverted_index = {}
        self.frequencies = {}
        self.cutoff_low_count = cutoff_low_count
        self.cutoff_top_frequent_words_fraction = cutoff_top_frequent_words_fraction
        self.labels = []
        self.vocabulary = []
        self.k_prob = {}
        self.prior_prob = {}
        self.min_len_filter = min_len_filter
        self.max_len_filter = max_len_filter
        self.smoothing = smoothing

        self.stop_words = stop_words if stop_words is not None else []

    def text_processor(self, text):
        return [word for word in re.split("[^a-zA-Z]+", text)
                if self.min_len_filter < len(word) < self.max_len_filter]

    def create_inverted_index(self, data_provider):
        self.labels = data_provider.labels[:]
        for k in self.labels:
            self.inverted_index[k] = {}
            self.prior_prob[k] = 0

        files = data_provider.get_files()

        for msg_type in self.labels:
            clean_words = []
            texts = []
            for filename in files[msg_type]:
                self.prior_prob[msg_type] += 1
                with open(filename, 'r', encoding='latin-1') as f:
                    texts.append(f.read().lower())
            clean_words += self.text_processor(''.join(texts))

            unique = set(clean_words)
            doc_words_count = {w: 0 for w in unique}
            for w in clean_words:
                doc_words_count[w] += 1

            self.vocabulary += list(unique)
            self.vocabulary = list(set(self.vocabulary) - set(self.stop_words))

            for word, freq in doc_words_count.items():
                if word not in self.inverted_index[msg_type]:
                    self.frequencies[word] = 0
                    for k in self.labels:
                        self.inverted_index[k][word] = 0
                self.inverted_index[msg_type][word] += freq
                self.frequencies[word] += freq

        words_to_remove = [w for w, f in self.frequencies.items() if f <= self.cutoff_low_count]

        if self.cutoff_top_frequent_words_fraction < 100 or True:
            sorted_freq = sorted(self.frequencies.items(), key=lambda kv: -kv[1])
            fraction_to_remove = [k for k, v in sorted_freq[0:int(len(self.frequencies) * self.cutoff_top_frequent_words_fraction)]]
            words_to_remove += fraction_to_remove

        self.vocabulary = list(set(self.vocabulary) - set(words_to_remove))
        for w in words_to_remove:
            del self.frequencies[w]
        self.vocabulary = sorted(self.vocabulary)

        sorted_inverted_index = {}
        for k in self.labels:
            sorted_inverted_index[k] = {w: self.inverted_index[k][w] for w in self.vocabulary}
        self.inverted_index = sorted_inverted_index

    def calc_probability(self):
        k_vocab = {}
        k_count = {}
        for k in self.labels:
            k_vocab[k] = len(self.inverted_index[k])
            k_count[k] = sum(self.inverted_index[k].values())
            self.k_prob[k] = {w: 0 for w in self.vocabulary}

        for k in self.labels:
            divide_by = k_count[k] + self.smoothing * k_vocab[k]
            for word in self.vocabulary:
                self.k_prob[k][word] = (self.inverted_index[k][word] + self.smoothing) / divide_by

    def save_model_to_file(self, file_name):
        line_num = 0
        outputs = []
        for word in self.vocabulary:
            line_num += 1
            output = [line_num, word]
            for k in self.labels:
                output.append(self.inverted_index[k][word])
                output.append(self.k_prob[k][word])
            outputs.append(output)

        with open(file_name, "w") as f:
            print("\n".join(['  '.join([str(item) for item in row]) for row in outputs]), file=f)

    def inference(self, provider):
        line_num = 0
        results = []
        files = provider.get_files()

        true_classes = []
        predicted_classes = []

        for label in files:
            for file in files[label]:
                line_num += 1
                with open(file, encoding='latin-1', mode='r') as f:
                    clean_words = self.text_processor(f.read().lower())

                unique = set(clean_words)
                doc_words_count = {w: 0 for w in unique}
                for w in clean_words:
                    doc_words_count[w] += 1

                result_label, probs = self.classify(doc_words_count)
                results.append([
                    line_num,
                    os.path.split(file)[1],
                    result_label,
                    probs[self.labels[0]],
                    probs[self.labels[1]],
                    label,
                    'right' if label == result_label else 'wrong'
                ])

                true_classes.append(label)
                predicted_classes.append(result_label)

        cm = self.confusion_matrix(true_classes, predicted_classes, self.labels)
        return results, cm

    def classify(self, word_count):
        resulting_probabilities = {}
        for label in self.labels:
            resulting_probabilities[label] = math.log(self.prior_prob[label] / sum(self.prior_prob.values()))
            for word in word_count.keys():
                if word in self.k_prob[label] and self.k_prob[label][word] > 0:
                    resulting_probabilities[label] += math.log10(self.k_prob[label][word])
        argmax_label = self.labels[0]
        for label in self.labels:
            if resulting_probabilities[label] > resulting_probabilities[argmax_label]:
                argmax_label = label
        return argmax_label, resulting_probabilities

    @staticmethod
    def save_results_to_file(results, file_name):
        with open(file_name, "w") as f:
            print("\n".join(['  '.join([str(item) for item in row]) for row in results]), file=f)

    @staticmethod
    def confusion_matrix(true_classes, predicted_classes, labels):
        cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for yt, yp in zip(true_classes, predicted_classes):
            if labels.index(yt) == labels.index(yp):
                if labels.index(yp) == 0:
                    cm['TP'] += 1
                else:
                    cm['TN'] += 1
            else:
                if labels.index(yp) == 0:
                    cm['FP'] += 1
                else:
                    cm['FN'] += 1
        result = dict(cm)
        print(cm)
        result['Accuracy'] = (cm['TP'] + cm['TN']) / sum(cm.values())
        result['Precision'] = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) != 0 else 1
        result['Recall'] = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) != 0 else 1
        result['F1'] = 2 * (result['Recall'] * result['Precision']) / (result['Recall'] + result['Precision']) if (result['Recall'] + result['Precision']) != 0 else 0
        return result
