import numpy as np
from IPython.display import clear_output
import sys
import shelve
from sklearn.cluster import KMeans
from configs.file_locations import config
from utilities.base import inset
from configs.naming_conventions import config as names

events = names['events']
fileinfo = config['compressed_particle_data']
npoints = 100
data = {}

def reader():
    from src.compute.compute_dvd_from_particles import data as partdvsr
    data = globals()['data']
    psrkey = 'pip_particle_snowrate'
    ssdkey = 'pip_ssd_snow_rate'
    outkey = 'compressed_pip_data'

    for e in events:
        data[e] = {}
        print("Compressing data for {}.".format(e))
        outtime = partdvsr[e][ssdkey]['t']
        N = len(outtime)
        new = 0
        total = 0
        check = 0
        blank = np.zeros(0).reshape(-1)
        bytime = {'t': [blank]*N, 'd': [blank]*N, 'v': [blank]*N, 'sr': [blank]*N, 'var': [blank]*N }
        for i,ts in enumerate(outtime):
            clear_output(wait=True)
            print( "Compressing data for {}. Step {:04d}/{:04d}: ".format(e,i+1,len(outtime)) )
            old = new
            for t in partdvsr[e][psrkey]['t'][old:]:
                if t != ts:
                    break
                new += 1
            tmpt = partdvsr[e][psrkey][old:new]
            dv = np.concatenate( (tmpt['d'].reshape(-1,1),tmpt['v'].reshape(-1,1)), axis = 1)
            if new - old <= npoints:
                total += new-old
                bytime['t'][i] = np.repeat([ts],new-old).reshape(-1)
                bytime['d'][i] = np.array(dv[:,0]).reshape(-1)
                bytime['v'][i] = np.array(dv[:,1]).reshape(-1)
                bytime['sr'][i] = tmpt['sr'].reshape(-1)
                bytime['var'][i] = np.repeat([0],new-old).reshape(-1)
            else:
                total += npoints
                bytime['t'][i] = np.repeat([ts],npoints).reshape(-1)
                variance = np.zeros(npoints)
                sr = np.zeros(npoints)
                kmeans = KMeans(npoints, n_init=5).fit(dv)
                for j in range(npoints):
                    tmp = np.array(tmpt[ kmeans.labels_ == j ])
                    if len(tmp) > 1:
                        blah =  np.sum( (tmp['d'] - kmeans.cluster_centers_[j,0])**2 )
                        variance[j] = np.sum( (tmp['d'] - kmeans.cluster_centers_[j,0])**2 + (tmp['v'] - kmeans.cluster_centers_[j,1])**2 )/(npoints-1)
                    sr[j] = np.sum( tmp['sr'] )

                bytime['d'][i] = np.array(kmeans.cluster_centers_[:,0]).reshape(-1)
                bytime['v'][i] = np.array(kmeans.cluster_centers_[:,1]).reshape(-1)
                bytime['sr'][i] = np.array(sr).reshape(-1)
                bytime['var'][i] = np.array(variance).reshape(-1)

        data[e][outkey] = np.zeros( total , dtype=fileinfo[outkey]['dataformat'] )
        data[e][outkey]['t'] = np.concatenate( bytime['t'] )
        data[e][outkey]['d'] = np.concatenate( bytime['d'] )
        data[e][outkey]['v'] = np.concatenate( bytime['v'] )
        data[e][outkey]['sr'] = np.concatenate( bytime['sr'] )
        data[e][outkey]['var'] = np.concatenate( bytime['var'] )

def save():
    shelf = shelve.open(config['shelves']['compressed_particle_data'])
    shelf.clear()
    shelf['data'] = globals()['data']
    shelf.close()

def writer(key = 'compressed_pip_data'):
    printformat = ''
    for (_,_),tipe in fileinfo[key]['dataformat']:
        printformat += config['printformat'][tipe] + config['printformat']['seperator']
    for k,filename in fileinfo[key]['files'].items():
        with open(filename,'w') as f:
            for entry in list(data[k][key]):
                f.write( printformat.format(*entry) )

shelf = shelve.open(config['shelves']['compressed_particle_data'])
if not 'data' in shelf:
    shelf.close()
    reader()
    save()
else:
    print("Reading {} from {}".format('compressed_particle_data','cpd.shelf'))
    data = shelf['data']
    shelf.close()
del(shelf)
