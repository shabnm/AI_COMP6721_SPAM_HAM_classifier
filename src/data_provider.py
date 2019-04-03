import os


class DataProvider:
    def __init__(self, labels=None) -> None:
        if labels is None:
            labels = ['ham', 'spam']
        self.labels = labels

    def get_files(self, source='train'):
        data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/"

        train_files = os.listdir(data_dir + "{}/".format(source))
        files = {}
        for k in self.labels:
            files[k] = [data_dir + '{}/{}'.format(source, f) for f in train_files if '-{}-'.format(k) in f]
        return files
