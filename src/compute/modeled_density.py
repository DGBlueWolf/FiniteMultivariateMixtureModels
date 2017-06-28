import shelve
import numpy as np
from scipy.stats import gamma
from scipy.stats import norm
from src.compute.compute_dvd_from_particles import data
import matplotlib.pyplot as plt
# We will have 3 types of particles 'columns','dendrites', and 'clusters', which have descriptive parameters
# Columns will have higher density, smaller size, and the greatest variability in velocity
# Dendrites will have lower density, size larger than columns, and standard variable in velocity
# Clusters will have the lowest density, size much larger and more variable than dendrites and standard velocity
# The velocity can be modeled well as a function of Diameter via lines of constant density with respect to density model
# variability in velocity will be gaussian
# variability in diameter will be exponential (Gamma) with a constant shape parameter

# Column: mass proportional to D^1 --- velocity : k/sqrt(d)
# Dendrite: mass proportional to D^2  --- velocity : k
# Aggregate: mass proportional to D^3 --- velocity : k*sqrt(d)
def em_optimizer_dvd( d, v, dvd ):
    #filters f1, f2, f3
    for f in filters:
        dvdf = dvd*f

        #snow size gamma distribution parameters
        ssd = dvdf.sum(1) #rowSums
        dm = np.average( d, weights = ssd )
        dv = np.average( (d - dm)**2 , weights = ssd ) * dvd.shape[0] / (dvd.shape[0]-1)
        beta = dm/dv
        alpha = beta*dm
        print( 'Alpha: {:.6f}, Beta: {:.6f}'.format(alpha,beta) )
        # velocity distribution parameters. evolution of variance and mean diameter
        # biased power law fit for mean
        # biased exponential fit for variance
        val = dvdf.sum()/4
        idx1 = ssd.cumsum() < val
        idx2 = ssd.cumsum() < 2*val
        idx3 = ssd.cumsum() < 3*val
        idx3 = ~idx2 & idx3
        idx2 = ~idx1 & idx2

        d1 = np.average( d[idx1], weights = ssd[idx1] )
        d2 = np.average( d[idx2], weights = ssd[idx2] )
        d3 = np.average( d[idx3], weights = ssd[idx3] )
        D = np.array( [[1,log(d1)],[1,log(d2)],[1,log(d3)]] )

        vd1 = dvdf[idx1].sum(0)
        vd2 = dvdf[idx2].sum(0)
        vd3 = dvdf[idx3].sum(0)

        vm1 = np.average( v, weights = vd1)
        vm2 = np.average( v, weights = vd2)
        vm3 = np.average( v, weights = vd3)
        betavm0,betavm1 = np.linalg.solve(D,[vm1,vm2,vm3])

        vv1 = np.average( (v-vm1)**2 , weights = vd1 ) * dvd.shape[1] / (dvd.shape[1]-1)
        vv2 = np.average( (v-vm1)**2 , weights = vd1 ) * dvd.shape[1] / (dvd.shape[1]-1)
        vv3 = np.average( (v-vm1)**2 , weights = vd1 ) * dvd.shape[1] / (dvd.shape[1]-1)
        betavv0,betavv1 = np.linalg.solve(D,[vv1,vv2,vv3])

def pdfD(D,alpha,beta):
    return gamma.pdf(D,alpha,scale=beta)

def pdfV(D,V,alphaScale,alphaPower,betaScale,betaPower):
    return gamma.pdf(V, alphaScale*D**alphaPower, scale=betaScale*D**betaPower )
    #return gamma.pdf(V, alphaScale + D*alphaPower, scale=betaScale + D*betaPower )

def em_optimizer_particle( partdvsr , arg_params = [], ncuts = 6 , k = 3 , tol = 1e0 , max_iter = 100):
    eps = 1e-12
    #take only velocity greater than 0
    v = partdvsr['v']
    vgoo = (v > 0) & (v < 4)
    d = partdvsr['d'][vgoo]
    sr = partdvsr['sr'][vgoo]
    t = partdvsr['t'][vgoo]
    v = v[vgoo]
    isort = d.argsort()

    idx = list()
    pcut = 0
    for i in range(1,ncuts):
        cut = (i*len(t)) // ncuts
        idx.append( isort[pcut:cut] )
        pcut = cut

    filters = np.zeros( (k, len(v)) )
    params = list()

    for i in range(k):
        params.append({
            'D': {
                'alpha': i + 1,
                'beta': 1,
            },
            'V': {
                'alphaScale': i + 1,
                'alphaPower': 0.5,
                'betaScale': i + 1,
                'betaPower': 0.05,
            },
            'mix': 0,
            'density_params': None,
        })

    for i in range(len(arg_params)):
        params[i].update( arg_params )

    iters = 0
    plike = -1.0e32
    while True:
        #e-step, calculate p(x|y,theta)
        iters += 1
        for i in range(k):
            #print( params[i]['D'] )
            filters[i] = pdfD( d , **params[i]['D'] ) * pdfV(d, v, **params[i]['V'])

        likelihood = np.sum( np.log( filters.sum(0) + 1e-24 ))
        print( "Iter {}: Likelihood = {}".format(iters,likelihood) )

        if iters > max_iter: # or likelihood - plike < tol
            break
        plike = likelihood

        filters /= filters.sum(0) + 1e-12
        filters /= filters.sum()
        mix = filters.sum(1)
        #print( "Mixture =", mix )

        #m-step
        for i in range(k):
            #print( "Iter {}.{}: ".format(iters,i) )
            params[i]['mix'] = mix[i]
            #calculate probabilites from initial parameters
            #f = dvd*f

            #snow size gamma distribution parameters
            dm = np.average( d, weights = filters[i] )
            dv = np.average( (d-dm)**2, weights = filters[i] )*(len(d)-1)/len(d)
            #print( "  D.mean = {}, D.var = {}".format(dm,dv) )
            beta = dv/dm
            alpha = dm/beta
            params[i]['D'].update( alpha = alpha , beta = beta )
            #print( "  D.alpha = {:e}, D.beta = {:e}".format(alpha,beta))
            #print( 'N0: {:e}, Alpha: {:e}, Beta: {:e}'.format(len(t),alpha,beta) )

            # velocity distribution parameters. evolution of variance and mean diameter
            # biased power law fit for mean
            # biased exponential fit for variance
            D = list()
            valpha = list()
            vbeta = list()
            for j in idx:
                dm = np.average( d[j], weights = filters[i][j] )
                D.append( [1,np.log(dm)] )
                vm = np.average( v[j], weights = filters[i][j] )
                vv = np.average( (v[j]-vm)**2, weights = filters[i][j] )
        #        print( "  D: {}, V.mean = {}, V.var = {}".format(dm,vm,vv) )
                b = vv/( vm + 1e-24)
                a = vm/b
                #print(a,b)
                valpha.append( a )
                vbeta.append( b )

            #print( valpha, vbeta)
            (alphaScale,alphaPower),(res,*_) = np.linalg.lstsq( D, np.log(valpha) )[:2]
            (betaScale,betaPower),(res,*_) = np.linalg.lstsq( D, np.log(vbeta) )[:2]
            alphaScale = np.exp(alphaScale)
            betaScale = np.exp(betaScale)

            params[i]['V'].update( alphaScale = alphaScale , alphaPower = alphaPower ,
                betaScale = betaScale , betaPower = betaPower )

            #print( 'Var_Velocity(D), Beta0: {:e}, Beta1: {:e}, Err: {:e}'.format(np.exp(betavv0),betavv1,res) )
            #if input('ok? ') == 'no':
    return len(t),params

def get_ssd( d , n , k0, k1 ):
    return n*(d**k1)*np.exp(-k0*d)

def get_velocity( d, v, v0, v1, vsigma, mu, r0 ):
    # v0 should be the hypothetical velocity of a partical with d = 1mm
    r = np.exp(-r0*d)
    mur = 3*r + (1 - r)*mu
    std = vsigma*(0.3+0.7*r)
    return norm.pdf( v, v0*std + v1*np.sqrt(d)**(mur-2) , vsigma*r )

def get_modeled_dvd(e):
    partdvsr = data[e]['pip_particle_snowrate']
    vt,dt = np.mgrid[0.1:15.1:0.1,0.1:15.1:0.1]
    N, params = em_optimizer_particle( partdvsr )

    def dvd(d,v):
        p = 0
        plist = list()
        for i in range(len(params)):
            a = N * params[i]['mix'] * pdfD( d, **params[i]['D'] ) * pdfV( d, v, **params[i]['V'] )
            p += a
            plist.append(a)
        return p, plist

    return params, dvd, dvd(dt,vt)

'''
v,d = np.mgrid[0.01:1:100j,0.01:1:100j]
dvd1 = get_ssd( d, 10000, -np.log(1/5.0)/0.02, 1.0 )*\
    get_velocity( d, v, 0.5, 0.20, 0.3, 1.5, -np.log(0.1)/0.1 ) #needles
dvd2 = get_ssd( d, 400000, -np.log(1/3.0)/0.02 , 2.0 )*\
    get_velocity( d, v, 0.5, 0.15, 0.08, 2.01, -np.log(0.5)/0.1 ) #dendrites
dvd3 = get_ssd( d, 5000, -np.log(1/2.0)/0.05 , 1.0 )*\
    get_velocity( d, v, 0.5, 0.19, 0.05, 2.4, -np.log(0.5)/0.6 ) #aggregates
dvd = dvd1 + dvd2 + dvd3
'''
