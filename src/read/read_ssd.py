import numpy as np
from configs.file_locations import config
from utilities.base import extractor
import shelve

#read files as defined in the config
def reader():
    data = {}
    shelf = shelve.open(config['shelves']['snow_size_distribution'])
    shelf.clear()
    #Record Structure
    dtype = config['snow_size_distribution']['dataformat']
    extra = lambda s: extractor(s,dtype)
    for k,v in config['snow_size_distribution']['files'].items():
        #read file and convert to numpy array with data structure as defined in dtype
        with open(v) as f:
            data[k] = np.array(list(map(extra, f.readlines()[1:])), dtype = dtype)
    globals()['data'] = data
    shelf['data'] = data
    shelf.close()

#read from shelf if on shelf or call reader
shelf = shelve.open(config['shelves']['snow_size_distribution'])
if not 'data' in shelf:
    shelf.close()
    reader()
else:
    data = shelf['data']
    shelf.close()
del(shelf)
