from configs.file_locations import config
from utilities.base import getarr as extra
import numpy as np
import shelve

#reads the particle_density data
def reader():
    data = {}
    shelf = shelve.open(config['shelves']['particle_density'])
    shelf.clear()
    #Data Structure
    dtype = config['particle_density']['dataformat']
    for k,v in config['particle_density']['files'].items():
        print("reading {}: {}".format(k,v) )
        #read file and convert to numpy array with data structure as defined in dtype
        with open(v) as f:
            linear = lambda s: tuple(s.strip().split(","))
            data[k] = np.array(list(map(extra, f.readlines()[1:])), dtype = dtype)

    #Save content
    globals()['data'] = data
    shelf['data'] = data
    shelf.close()

#get config info and check if data exists on the system already otherwise read it
shelf = shelve.open(config['shelves']['particle_density'])
if not 'data' in shelf:
    shelf.close()
    reader()
else:
    data = shelf['data']
    shelf.close()
del(shelf)
