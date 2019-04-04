import os
import re
import math
import numpy
import numpy as np


class NaiveBayesModel:

    def __init__(self, stop_words=None, min_len_filter=0, max_len_filter=math.inf, cutoff_frequency=0) -> None:
        self.inverted_index = {}
        self.frequencies = {}
        self.cutoff_frequency = cutoff_frequency
        self.labels = []
        self.vocabulary = []
        self.k_prob = {}
        self.prior_prob = {}
        self.min_len_filter = min_len_filter
        self.max_len_filter = max_len_filter

        self.stop_words = stop_words if stop_words is not None else []

    def text_processor(self, text):
        clean_text = [word for word in re.split("[^a-zA-Z]+", text.lower()) if word is not None and
                      word != '' and
                      word not in self.stop_words and
                      self.min_len_filter < len(word) < self.max_len_filter
                      ]
        return clean_text

    def create_inverted_index(self, data_provider):
        self.labels = data_provider.labels[:]
        for k in self.labels:
            self.inverted_index[k] = {}
            self.prior_prob[k] = 0

        files = data_provider.get_files()

        for msg_type in self.labels:
            clean_words = []
            for filename in files[msg_type]:
                self.prior_prob[msg_type] += 1
                with open(filename, 'r', encoding='latin-1') as f:
                    clean_words += self.text_processor(f.read())

            unique, counts = numpy.unique(clean_words, return_counts=True)
            doc_words_count = dict(zip(unique, counts))

            self.vocabulary += list(unique)
            self.vocabulary = list(set(self.vocabulary))

            for word, freq in doc_words_count.items():
                if word not in self.inverted_index[msg_type]:
                    self.frequencies[word] = 0
                    for k in self.labels:
                        self.inverted_index[k][word] = 0
                self.inverted_index[msg_type][word] += freq
                self.frequencies[word] += freq

        words_to_remove = [w for w, f in self.frequencies.items() if f <= self.cutoff_frequency]

        self.vocabulary = list(set(self.vocabulary) - set(words_to_remove))
        for w in words_to_remove:
            del self.frequencies[w]
        self.vocabulary = sorted(self.vocabulary)

        sorted_inverted_index = {}
        for k in self.labels:
            sorted_inverted_index[k] = {w: self.inverted_index[k][w] for w in self.vocabulary}
        self.inverted_index = sorted_inverted_index

    def calc_probability(self, smoothing=0):
        k_vocab = {}
        k_count = {}
        for k in self.labels:
            k_vocab[k] = len(self.inverted_index[k])
            k_count[k] = sum(self.inverted_index[k].values())
            self.k_prob[k] = {w: 0 for w in self.vocabulary}

        for word in self.vocabulary:
            for k in self.labels:
                self.k_prob[k][word] = (self.inverted_index[k][word] + smoothing) / (k_count[k] + (smoothing * k_vocab[k]))

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

    def inference(self, provider, smoothing=0):
        line_num = 0
        results = []
        files = provider.get_files()

        true_classes = []
        predicted_classes = []

        for label in files:
            for file in files[label]:
                line_num += 1
                with open(file, encoding='latin-1', mode='r') as f:
                    clean_words = self.text_processor(f.read())

                unique, counts = numpy.unique(clean_words, return_counts=True)
                doc_words_count = dict(zip(unique, counts))

                result_label, probs = self.classify(doc_words_count, smoothing)
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

    def classify(self, word_count, smoothing=0):
        resulting_probabilities = {}
        for label in self.labels:
            resulting_probabilities[label] = math.log(self.prior_prob[label] / sum(self.prior_prob.values()))
            for word in word_count.keys():
                if word in self.k_prob[label] and self.k_prob[label][word] > 0:
                    resulting_probabilities[label] += math.log(self.k_prob[label][word])
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
        cm = np.zeros((len(labels), len(labels)))
        for yt, yp in zip(true_classes, predicted_classes):
            cm[labels.index(yt), labels.index(yp)] += 1
        return cm
