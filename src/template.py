import shelve
import numpy as np
from configs.file_locations import config
from src.compute.compute_dvd_from_particles import data as partdvsr
from src.read.read_rr import data as rr

def reader():
    '''
    Computes the average area ratio for each event as defined in the config
    '''
    data = {}
    shelf = shelve.open(config['shelves']['computed_area_ratio'])
    shelf.clear()
    globals()['data'] = data
    shelf['data'] = data
    shelf.close()

shelf = shelve.open(config['shelves']['computed_area_ratio'])
if not 'data' in shelf:
    shelf.close()
    reader()
else:
    data = shelf['data']
    shelf.close()
del(shelf)
