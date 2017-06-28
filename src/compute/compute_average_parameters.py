import shelve
import numpy as np
from utilities.base import inset
from configs.file_locations import config
from configs.naming_conventions import config as names
from src.compute_dvd_from_particles import data as partdvsr
from src.compute.compute_mean_area_ratio_from_particles import data as mar
from src.read.read_rr import data as rr
from scipy.optimize import minimize

def plaw0( d , ar , k , a ):
    return ar * k * (d**a)

def plaw1( d , v , ar , k , a , b ):
    return ar * k * (d**a) * (v**b)

def plaw2( d , v , ar , k , a0 , b0, a1, b1):
    return ar * k * ( d**2 + a0*d**b0 ) * (v ** 2 + a1*v**b1) / (d**3)

def bohms_mass( dia , vel , rhoA=1.328 , etaA=1.618e-5 , Ar= 1):
    g = 9.8
    re = rhoA*dia*vel/etaA # (kg.m-3) . (m.s-1) . (m2)1/2 . (kg.m-1.s-1)-1  = Reynolds Number, Dimensionless
    x = (((np.sqrt(re/8.5)+1)**2-1)/0.1519)**2 # Davies Number, Dimensionless
    mass = Ar*np.pi*x*(etaA**2)/(8*g*rhoA) # particle mass in kg
    density = mass/(1000*(np.pi/6)*dia**3)
    return mass , density, re , x

def reader():
    '''
    Fits parameters of the powerlaws to bohm's method to find the constant of proportionality
    '''
    data = {}
    #shelf = shelve.open(config['shelves']['computed_plaw_params'])
    #shelf.clear()
    ar = 1
    d,v = [ x.flatten() for x in np.mgrid[0.1:10:100j,0.1:10:100j] ]
    bd = ar*bohms_mass(d,v)[1]

    init0 = [.002,-1.0]
    def loss0(p):
        k , a = p
        f = lambda d: plaw0(d , ar , k , a )
        return np.sum( ( f(d) - bd )**2 )
    res0 = minimize(loss0,init0)
    data['init0'] = res0.x
    print(res0.x,res0.success,res0.fun,res0.message)

    init1 = [6.3692e-05,-1.0,2.0]
    def loss1(p):
        k , a , b = p
        f = lambda d, v: plaw1(d, v, ar, k, a, b)
        return np.sum( ( f(d,v) - bd )**2 )
    res1 = minimize(loss1,init1)
    data['init1'] = res1.x
    print(res1.x,res1.success,res1.fun,res1.message)

    init2 = [6.3692e-05,1.0,1.0,1.0,1.0]
    def loss2(p):
        k, a0, b0, a1, b1 = p
        f = lambda d, v: plaw2(d, v, ar, k, a0, b0, a1, b1)
        return np.sum( ( f(d,v) - bd )**2 )
    res2 = minimize(loss2,init2)
    data['init2'] = res2.x
    print(res2.x,res2.success,res2.fun,res2.message)

    for e in names['events']:
        ar = mar['event_summary'][e]['arave']
        part = partdvsr[e]['pip_particle_snowrate']
        sr = part['sr']
        d = part['d']
        v = part['v']
        t = part['t']
        def compute_swer_plaw(d,v,sr,plaw,t):
            cnt = 0
            swermintot = np.zeros( len(t) , dtype = np.float64)
            for i,(den,sr,t) in enumerate(zip(density,sr,t)):
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
    '''globals()['data'] = data
    shelf['data'] = data
    shelf.close()'''

'''shelf = shelve.open(config['shelves']['computed_area_ratio'])
if not 'data' in shelf:
    shelf.close()
    reader()
else:
    data = shelf['data']
    shelf.close()
del(shelf)'''
