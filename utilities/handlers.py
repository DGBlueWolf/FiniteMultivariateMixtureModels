import shelve

import numpy as np

from configs.file_locations import config
from utilities.base import extractor


class Reader(dict):
    def __init__(self, data_name: str, dict_label: str = 'data'):
        super().__init__()
        self.data_name = data_name
        self.dict_label = dict_label
        shelf = shelve.open(config['shelves'][data_name])
        if dict_label not in shelf:
            shelf.close()
            self.read()

    def read(self):
        dtype = config[self.data_name]['dataformat']
        for k, v in config[self.data_name]['files'].items():
            with open(v) as f:
                self[k] = np.array(list(map(
                    lambda s: extractor(s, dtype),
                    f.readlines()[1:]
                )), dtype=dtype)

    def write(self):
        header = config['printformat']['separator'].join(
            s for (s, _), _ in config[self.data_name]['dataformat']
        )
        printformat = config['printformat']['separator'].join(
            config['printformat'][tipe] for (_, _), tipe in config[self.data_name]['dataformat'])
        for k, fname in config[self.data_name]['files'].items():
            with open(fname, 'w') as f:
                for entry in self[k]: print(printformat.format(*entry))


    def save(self):
        shelf = shelve.open(config['shelves'][self.data_name])
        shelf.clear()
        shelf[self.dict_label] = dict(self)
        shelf.close()
