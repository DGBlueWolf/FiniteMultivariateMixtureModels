import shelve
import numpy as np
from utilities.base import inset
from configs.file_locations import config
from configs.naming_conventions import config as names
from src.compute.compute_dvd_from_particles import data as partdvsr
from src.read.read_rr import data as rr

feed = 'computed_area_ratio'
dkey1 = 'sliding_window_averages'
dkey2 = 'event_summary'

def get_sliding_window_indices(outtime,slwidth):
    ranges = list() #list of (start,stop)
    idx,start = 0,outtime[0]
    for i,stop in enumerate(list(outtime[1:])+[outtime[-1]+slwidth,],start=1):
        if not stop - start < slwidth:
            while True:
                ranges.append( [idx,i] )
                idx += 1
                if not idx < len(outtime): break
                start = outtime[idx]
                if stop - start < slwidth: break
        else:
            ranges.append( (idx,i) )
    return np.array(ranges)

def window_sums(arr,ranges_in,axis=None):
    '''Compute windowed sums over arbitrarily sized windows'''
    ranges = np.array(ranges_in)
    shape = list(arr.shape)
    shape[axis or 0] = 1
    ans = np.append(np.zeros(shape),arr.cumsum(axis),axis)
    return ans.take(ranges[:,1],axis) - ans.take(ranges[:,0],axis)

def reader():
    '''
    Computes the average area ratio for each event as defined in the config file.
    There is a moving window feed with a configurable width as well as event and global averages.
    Area ratio is computed as the ratio between bohm's density and the density as measured by snow_rate and snow water
    equivalent rate.
    Hopefully the computed area ratio is consistent across each event so that this method makes sense.
    '''
    data = {}
    shelf = shelve.open(config['shelves'][feed])
    shelf.clear()

    data[dkey1] = {}
    data[dkey2] = {}
    for e in names['events']:
        ssdsrkey = 'pip_ssd_snow_rate'
        psrkey = 'pip_particle_snowrate'

        ssdsrtime = partdvsr[e][ssdsrkey]['t']
        rrtime = rr[e]['t']-5 #pip records 5 minutes after disdrometer observation
        timeset = frozenset(ssdsrtime) & frozenset(rrtime)
        outtime = np.array(sorted(list(timeset)))

        ssdsr = partdvsr[e]['pip_ssd_snow_rate'][inset(ssdsrtime,timeset)]
        swer = rr[e][inset(rrtime,timeset)]

        slidx = get_sliding_window_indices( outtime, config[feed][dkey1]['sliding_window_width'] )
        win_swer = window_sums( swer['rr']*60, slidx )
        win_ssdsr = window_sums( ssdsr['sr'], slidx )
        win_swer_bd = window_sums( ssdsr['sr']*ssdsr['bd'], slidx )
        zdenom = win_swer_bd != 0
        ar = win_swer/win_swer_bd

        slwav = np.zeros( len(slidx), dtype=config[feed][dkey1]['dataformat'] )
        eva = np.zeros( None, dtype=config[feed][dkey2]['dataformat'] )

        slwav['tstart'] = outtime[ slidx[:,0] ]
        slwav['tstop'] = outtime[ slidx[:,1]-1 ]
        slwav['istart'] = slidx[:,0]
        slwav['istop'] = slidx[:,1]
        slwav['count'] = slidx[:,1] - slidx[:,0]
        slwav['swer'] = win_swer
        slwav['ssdsr'] = win_ssdsr
        slwav['swer_bd'] = win_swer_bd
        slwav['ar'] = ar

        eva['swer'] = (win_swer/slwav['count']).mean()
        eva['ssdsr'] = (win_ssdsr/slwav['count']).mean()
        eva['bd'] = win_swer_bd.sum()/win_ssdsr.sum()
        eva['arave'] = (win_ssdsr*ar)[ win_swer_bd != 0 ].sum()/win_ssdsr[ win_swer_bd != 0 ].sum()
        eva['arstd'] = np.sqrt(np.average( (ar - eva['arave'])[zdenom]**2, weights=win_ssdsr[zdenom] ))
        eva['minutes'] = len(outtime)

        data[dkey1][e] = slwav
        data[dkey2][e] = eva

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
