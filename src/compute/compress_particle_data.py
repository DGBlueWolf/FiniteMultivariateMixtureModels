import numpy as np
import shelve
from src.compute.compute_dvd_from_particles import data as pd
from sklearn.cluster import Birch
from configs.file_locations import config
from utilities.base import inset
from configs.naming_conventions import config as names

fileinfo = config['compressed_particle_data']

def reader():
    data = {}
    shelf = shelve.open(config['shelves']['computed_snow_rate'])
    shelf.clear()

    key1 = 'compressed_pip_data'
    for e in events:
        data[e] = {}
        print("Computing dvd for {}.".format(e))

        data[e][key1] = np.zeros( len(partsr), dtype=fileinfo[key1]['dataformat'])
        data[e][key1]['t'] =
        data[e][key1]['d'] =
        data[e][key1]['v'] =
        data[e][key1]['sr'] =

    globals()['data'] = data
    shelf['data'] = data
    shelf.close()

def writer(key = 'compressed_pip_data'):
    for k,filename in fileinfo[key1]['files'].items():
        with open(filename,'w') as f:
            for t,d,v,sr in list(data[k][key1]):
                f.write( fileinfo[key1]['printformat'].format(t=t,d=d,v=v,sr=sr) )
