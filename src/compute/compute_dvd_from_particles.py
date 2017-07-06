import numpy as np
import shelve
from src.read.read_meta import data as meta
from configs.file_locations import config
from utilities.base import inset
from configs.naming_conventions import config as names

fileinfo = config['computed_snow_rate']
pipbins = meta['pip_bin_centers']['bins']['bins'][0]
pipwindow = meta['pip_info']['pipinfo']['window'][0]
events = names['events']
data = {}

def calc_snowrate_part( d , v ):
    x,y = pipwindow
    relative_sample_volume = 117*d*(x-d)*(y-d)*1e-9 #approximate sample volume in cubic meters
    return (np.pi*6e-4*v*d**3)/relative_sample_volume #mm/hr??

def calc_snowrate_vdssd( vd , ssd ):
    delta = np.append(pipbins[1:]-pipbins[:-1],[0,]) #Width of interval in millimeters
    return np.sum(np.pi*6e-4*delta*(vd*ssd)*pipbins**3,1)

def reader():
    from src.read.read_vd import data as vd
    from src.read.read_ssd import data as ssd
    from src.read.read_particle_density import data as pd
    data = globals()['data']

    #Register computed data in config file, specify format there.
    for e in events:
        data[e] = {}
        print("Computing dvd for {}.".format(e))
        ssdtime, uissd = np.unique(ssd[e]['t'],return_index=True)
        vdtime, uivd = np.unique(vd[e]['t'],return_index=True)
        parttime = pd[e]['t']
        #get indices of unique elements to account for duplicate times in ssd and vd
        ssd[e] = ssd[e][uissd]
        vd[e] = vd[e][uivd]

        #The time must be in the intersection of times for each data feed
        timeset = frozenset(ssdtime) & frozenset(vdtime) & frozenset(parttime)
        outtime = np.array(sorted(list(timeset)))
        #get the particle
        partdv = pd[e][['d','vy','t','rhoS']][inset(parttime,timeset)]
        #compute estimated snowrate from particles
        partsr = calc_snowrate_part( partdv['d'], partdv['vy'] )
        #compute actual observed snowrate
        ssdsr = calc_snowrate_vdssd( vd[e]['vd'][inset(vdtime,timeset)], ssd[e]['ssd'][inset(ssdtime,timeset)] )

        prevt = outtime[0]
        idx = 0
        partsrmintot = np.zeros(len(outtime),dtype=[('sr','f8'),('rhoS','f8'),('t','i4')])
        partsrmintot['t'] = outtime
        cnt = 0
        for i,(sr,t,bd) in enumerate(zip( list(partsr), list(partdv['t']), list(partdv['rhoS']) )):
            if t != prevt:
                prevt = t
                idx += 1
                if cnt != 0:
                    partsrmintot['rhoS'][idx-1] /= cnt
                cnt = 0
                partsrmintot['rhoS'][idx] = 0
                partsrmintot['sr'][idx] = 0

            if not np.isnan(bd):
                partsrmintot['rhoS'][idx] += sr*bd
                cnt += sr

            partsrmintot['sr'][idx] += sr

        if cnt != 0:
            partsrmintot['rhoS'][-1] /= cnt

        factor = ssdsr/(partsrmintot['sr'] + 1e-9)
        #match actual to estimate
        for i in range(len(partsr)):
            partsr[i] *= factor[partdv['t'][i]==outtime]

        #data keys from config file
        key1 = 'pip_particle_snowrate'
        key2 = 'pip_ssd_snow_rate'

        #particle snowrates
        data[e][key1] = np.zeros( len(partsr), dtype=fileinfo[key1]['dataformat'])
        data[e][key1]['t'] = partdv['t']
        data[e][key1]['d'] = partdv['d']
        data[e][key1]['v'] = partdv['vy']
        data[e][key1]['bd'] = partdv['rhoS']
        data[e][key1]['sr'] = partsr

        #Minute total snowrates
        data[e][key2] = np.zeros( len(outtime), dtype=fileinfo[key2]['dataformat'])
        data[e][key2]['t'] = outtime
        data[e][key2]['bd'] = partsrmintot['rhoS']
        data[e][key2]['sr'] = ssdsr

def save():
    shelf = shelve.open(config['shelves']['computed_snow_rate'])
    shelf.clear()
    shelf['data'] = globals()['data']
    shelf.close()

def write_pip_particle_snowrate():
    key1 = 'pip_particle_snowrate'
    for k,filename in fileinfo[key1]['files'].items():
        with open(filename,'w') as f:
            for t,d,v,sr in list(data[k][key1]):
                f.write( fileinfo[key1]['printformat'].format(t=t,d=d,v=v,sr=sr) )

def write_pip_ssd_snow_rate():
    key2 = 'pip_ssd_snow_rate'
    for k,filename in fileinfo[key1]['files'].items():
        with open(filename,'w') as f:
            for t,d,v,sr in list(data[k][key1]):
                f.write( fileinfo[key1]['printformat'].format(t=t,d=d,v=v,sr=sr) )

shelf = shelve.open(config['shelves']['computed_snow_rate'])
if not 'data' in shelf:
    shelf.close()
    reader()
    save()
else:
    data = shelf['data']
    shelf.close()
del(shelf)
