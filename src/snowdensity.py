import numpy as np
import matplotlib.pyplot as plt
from .compute_dvd_from_particles import data as partdvsr
from .read.read_rr import data as rr
from scipy.optimize import minimize

def get_density_func( a , b , c ):
    return lambda diameter,velocity: c*(diameter**a)*(velocity**b)

def calc_gauge( diameter, velocity, snowrate, den ):
    np.sum(den(diameter,velocity)*snowrate)
    return gauge

def calc_snowrate( diameter, velocity, dvd ):
    return (np.pi*1.2e-4)*dvd*velocity*diameter**3

def bohms_mass( dia , vel , rhoA=1.328 , etaA=1.618e-5 , Ar= 1):
    g = 9.8
    re = rhoA*dia*vel/etaA # (kg.m-3) . (m.s-1) . (m2)1/2 . (kg.m-1.s-1)-1  = Reynolds Number, Dimensionless
    x = (((np.sqrt(re/8.5)+1)**2-1)/0.1519)**2 # Davies Number, Dimensionless
    mass = Ar*np.pi*x*(etaA**2)/(8*g*rhoA) # particle mass in kg
    density = mass/(1000*(np.pi/6)*dia**3)
    return mass , density, re , x

def myplotf(  f1 , f2 , d , v):
    def cycle(col):
        while True:
            for c in col:
                yield c
    color = cycle('bgrcmy')
    for v0 in v:
        plt.semilogy(d,f1(d,v0),color=next(color))
        plt.semilogy(d,f2(d,v0),color=next(color))
        plt.ylabel('density')
        plt.xlabel('diameter')
        plt.axis([0.00,0.01,0.001,1])
    plt.show()

class PowMass:
    def __init__(self,**params):
        self.params = { 'k': 0.00012, 'a': -0.92 , 'b': 2.0 , 'r': 0.75 , 's': 0.8}
        self.params.update(params)
    def __call__(self,d,v,Ar = 1):
        k,a,b,r,s = tuple( self.params[k] for k in 'kabrs')
        return Ar*k*d**a*abs(v)**b*(s*np.exp(-r*v)+1)

def optimize_particle( diameter, velocity, data , gauge ,lr = 0.1,mr=0.1, threshold = 1e-8):
    calcrr = lambda snowrate,den: calc_gauge(diameter,velocity,snowrate,den)
    calcsr = lambda dvd: calc_snowrate(diameter,velocity,dvd)
    rhoA = data['particle'][:]['rhoA']
    etaA = data['particle'][:]['etaA']
    dvd = data['particle'][:]['dvd']

    initBohm = [1.0,]
    def lossBohm(arg):
        Ar, = arg
        return bohms_mass(diameter, velocity,
            data['particle'][:]['rhoA'],
            data['particle'], Ar = Ar)

    initEq = [0.00012,-0.92,2.0,0.0,0.0]
    def lossEq(arg):
        k,a,b,r,s = arg
        return (gauge - Ar*k*diameter**a*abs(velocity)**b*(s*np.exp(-r*velocity)+1))

def 
