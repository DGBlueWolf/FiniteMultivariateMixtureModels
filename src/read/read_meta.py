import numpy as np
import shelve
from configs.file_locations import config
from utilities.base import extractor

#get config info
metafields = config['meta']

#reads each item in metadata and saves it to the system
def reader():
    data = {}
    shelf = shelve.open(config['shelves']['meta'])
    shelf.clear()
    for fkey,field in metafields.items():
        dtype = field['dataformat']
        extra = lambda s: extractor(s,dtype)
        data[fkey] = {}
        for k,v in field['files'].items():
            with open(v) as f:
                data[fkey][k] = np.array(list(map(extra, f.readlines()[1:])), dtype = dtype)
    shelf['data'] = data
    globals()['data'] = data
    shelf.close()

#get config info and check if data exists on the system already otherwise read it
shelf = shelve.open(config['shelves']['meta'])
if not 'data' in shelf:
    shelf.close()
    reader()
else:
    data = shelf['data']
    shelf.close()
del(shelf)
