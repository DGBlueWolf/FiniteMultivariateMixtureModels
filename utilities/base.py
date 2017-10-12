import os,re,sys,collections
import numpy as np

def setattrs(_self, **kwargs):
    for k,v in kwargs.items():
        setattr(_self, k, v)
        
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

def inset(t , tset):
    return np.vectorize(lambda t: t in tset)(t)

def getarr(s):
    return tuple(s.strip().replace('NA','nan').split(','))

def extractor(s,dtype):
    #reshape flat array of comma separated strings to match dtype format
    arr = iter(s.strip().replace('NA','nan').split(','))
    nparr = np.zeros(None,dtype).item()
    def fillarr(nparr):
        #print(type(nparr))
        if isinstance(nparr,tuple):
            l = list()
            for x in nparr:
                l.append( fillarr(x) )
            return tuple(l)
        if isinstance(nparr,list):
            l = list()
            for x in nparr:
                l.append( fillarr(x) )
            return l
        if isinstance(nparr,np.ndarray):
            l = list()
            for x in list(nparr):
                l.append( fillarr(x) )
            return l
        if isinstance(nparr,np.void):
            return fillarr(nparr.item())
        return next(arr)
    return fillarr(nparr)


def specfiles(files,base="",ext=""):
    '''
    Generates a dictionary with the full paths given only partial file names with base and extension
    '''
    if type(files) is str: return base + files + ext
    if type(files) is list: return list(map(lambda s: base + s + ext,files))
    out = {}
    for lab,val in files.items():
        out[lab] = base + val + ext
    return out

def listmatch(dir,key=""):
    '''
    Gets list of files in 'dir' that contain the regular expression 'key'
    '''
    return list(filter( lambda s: re.search(key,s) , os.listdir(dir) ))
