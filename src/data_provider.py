import os


class DataProvider:
    def __init__(self, dir, source='train', labels=None) -> None:
        if labels is None:
            labels = ['ham', 'spam']
        self.labels = labels
        self.source = source
        self.dir = dir

    def get_files(self):
        files_in_dir = os.listdir(self.dir + "{}/".format(self.source))
        files_in_dir = sorted(files_in_dir)
        files = {}
        for k in self.labels:
            files[k] = [self.dir + '{}/{}'.format(self.source, f) for f in files_in_dir if '-{}-'.format(k) in f]
        return files
