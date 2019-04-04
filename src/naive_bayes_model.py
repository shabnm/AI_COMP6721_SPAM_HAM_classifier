import os
import re
import math
import numpy as np


class NaiveBayesModel:

    def __init__(self, stop_words=None, remove_this_or_shorter=0, remove_this_or_longer=math.inf) -> None:
        self.inverted_index = {}
        self.labels = []
        self.vocabulary = []
        self.k_prob = {}
        self.prior_prob = {}
        self.remove_this_or_shorter = remove_this_or_shorter
        self.remove_this_or_longer = remove_this_or_longer

        self.stop_words = stop_words if stop_words is not None else []

    def text_processor(self, text):
        clean_text = [word for word in re.split("[^a-zA-Z]+", text.lower()) if word is not None and
                      word != '' and
                      word not in self.stop_words and
                      self.remove_this_or_shorter < len(word) < self.remove_this_or_longer
                      ]
        return clean_text

    def create_inverted_index(self, data_provider):
        self.labels = data_provider.labels[:]

        for k in self.labels:
            self.inverted_index[k] = {}
            self.prior_prob[k] = 0

        files = data_provider.get_files()

        for msg_type in self.inverted_index.keys():
            for filename in files[msg_type]:
                self.prior_prob[msg_type] += 1
                with open(filename, 'r', encoding='latin-1') as f:
                    clean_words = self.text_processor(f.read())
                    for word in clean_words:
                        if word not in self.vocabulary:
                            self.vocabulary.append(word)
                            for k in self.labels:
                                self.inverted_index[k][word] = 0
                        self.inverted_index[msg_type][word] += 1

        sorted_inverted_index = {}
        for k in self.labels:
            sorted_inverted_index[k] = dict(sorted(self.inverted_index[k].items()))

        self.vocabulary = sorted(self.vocabulary)

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
                num = self.inverted_index[k][word] + smoothing
                den = k_count[k] + (smoothing * k_vocab[k])
                self.k_prob[k][word] = num / den

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
                word_count = {}
                with open(file, encoding='latin-1', mode='r') as f:
                    clean_text = self.text_processor(f.read())
                    for word in clean_text:
                        if word not in word_count:
                            word_count[word] = 0
                        word_count[word] += 1
                    result_label, probs = self.classify(word_count, smoothing)
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
                if word in self.k_prob[label]:
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
