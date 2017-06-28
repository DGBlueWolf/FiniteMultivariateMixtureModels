from configs.file_locations import config
from utilities.base import extractor
import numpy as np
import shelve

#reads the files as specified in the config
def reader():
    data = {}
    shelf = shelve.open(config['shelves']['rain_rate'])
    shelf.clear()
    #Record Structure
    dtype = config['rain_rate']['dataformat']
    #rr,swer.fake,den.fake,rr1 are extra records that should not be extracted.
    extra = lambda s: extractor(s,dtype)
    for k,v in config['rain_rate']['files'].items():
        #read file and convert to numpy array with data structure as defined in dtype
        with open(v) as f:
            data[k] = np.array(list(map(extra, f.readlines()[1:])), dtype = dtype)
    globals()['data'] = data
    shelf['data'] = data
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
