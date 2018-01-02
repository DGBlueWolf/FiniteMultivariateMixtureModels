from configs.file_locations import config
from utilities.base import extractor
import numpy as np
import pandas as pd
import shelve

#reads the files as specified in the config
def reader():
    data = {}
    #Record Structure
    dtype = config['rain_rate']['dataformat']
    extra = lambda s: extractor(s, dtype)
    for k,v in config['rain_rate']['files'].items():
        #read file and convert to numpy array with data structure as defined in dtype
        data[k] = pd.read_csv(v, dtype=dtype).to_records()
        #with open(v) as f:
         #   data[k] = np.array(list(map(extra, f.readlines()[1:])), dtype = dtype)
    globals()['data'] = data
    save(data)

def save(data_):
    shelf = shelve.open(config['shelves']['rain_rate'])
    shelf.clear()
    shelf['data'] = data_
    shelf.close()

#checks if data is on the shelf and calls reader or loads from shelf
shelf = shelve.open(config['shelves']['rain_rate'])
if not 'data' in shelf:
    shelf.close()
    reader()
else:
    data = shelf['data']
    shelf.close()
del(shelf)
