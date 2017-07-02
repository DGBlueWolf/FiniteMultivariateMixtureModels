import shelve
import numpy as np
from pprint import pprint
from scipy.special import digamma
from IPython.display import clear_output
from scipy.stats import gamma
from scipy.stats import norm
import matplotlib.pyplot as plt
from configs.file_locations import config as fconfig
import configs.compute.model_dv_distribution as config
from src.compute.compress_particle_data import data as cpd
from configs.naming_conventions import config as names

v_alpha_params = ['slope','scale','intercept']
v_beta_params = ['slope','scale','intercept']
inkey = 'compressed_pip_data'
outkey = 'model_dv_distribution'
events = ['e3',]

def init_v_alpha():
    return {
        'slope': np.random.uniform(0.5,5),
        'scale': np.random.uniform(0.1,1),
        'intercept': np.random.uniform(0.5,2),
    }

def init_v_beta():
    return {
        'slope': np.random.uniform(0.5,5),
        'scale': np.random.uniform(0.01,1),
        'intercept': np.random.uniform(2,3.5),
    }

def alpha_v( d, slope, scale, intercept):
    return  intercept + slope*(1 - np.exp(-scale*d))

def der_alpha_v(d, slope, scale, intercept):
    return  {
        'slope':     1 - np.exp(-scale*d),
        'scale':     d*slope*np.exp(-scale*d),
        'intercept': 1,
    }

def beta_v( d, slope, scale, intercept):
    return intercept + slope*(1 - np.exp(-scale*d))

def der_beta_v(d, slope, scale, intercept):
    return  {
        'slope':     1 - np.exp(-scale*d),
        'scale':     d*slope*np.exp(-scale*d),
        'intercept': 1,
    }

def der_loggamma_alpha( d, alpha, beta ):
    return np.log(beta*d)-digamma(alpha)

def der_loggamma_beta( d, alpha, beta ):
    return alpha/beta - d

def der_V_alpha( d, v, alpha, beta):
    dave = der_alpha_v(d , **alpha)
    dlog = der_loggamma_alpha(v, alpha_v(d,**alpha), beta_v(d,**beta))
    for k in dave:
        dave[k] *= dlog
    return dave

def der_V_beta( d, v, alpha, beta):
    dave = der_beta_v(d , **alpha)
    dlog = der_loggamma_beta(v, alpha_v(d,**alpha), beta_v(d,**beta))
    for k in dave:
        dave[k] *= dlog
    return dave

def logmodel( d , v , D , V  ):
    av = alpha_v( d, **V['alpha'] )
    bv = beta_v( d, **V['beta'] )
    vd = np.log( gamma.pdf( v, av, 0, 1/bv ))
    return np.log(gamma(D['alpha'],0,1/D['beta']).pdf( d )) + vd

def em( parts , max_iter=1000 , thresh = 0.05 , lr = 0.05 ):
    parts = parts[ (parts['v'] > 0) & (parts['v'] < 4) ]
    done = False
    outtime = np.unique(parts['t'])
    deltas = [{
        'D': {
            'alpha': 0.0,
            'beta': 0.0,
        },
        'V': {
            'alpha': {},
            'beta': {},
        },
    } for _ in range(config.n_feats)]

    #parameters for the distribution model
    params = [{
        'D': { #gamma distribution parameters
            'alpha': 0.0,
            'beta': 0.0,
        },
        'V': {
            'alpha': {},
            'beta': {},
        }
    } for _ in range(config.n_feats)]

    #initialize parameters somehow
    for j in range( config.n_feats ):
        params[j]['D']['alpha'] = np.random.uniform(1,3)
        params[j]['D']['beta'] = np.random.uniform(2,12)
        params[j]['V']['alpha'].update( init_v_alpha() )
        params[j]['V']['beta'].update( init_v_beta() )

    #define error model parameters
    pprint( params )
    if input('Ok? ') == "no":
        return params

    #define a structure to hold the scores for each model in the mixture
    group_type = [
        (('Time','t'),'i4'),
        (('Feature scores','fs'),'{}f8'.format(config.n_feats)),
        (('Error scores','err'),'{}f8'.format(config.err_feats)),
    ]
    group_scores = np.zeros( len(outtime) , dtype = group_type )

    plike = -1.2e100
    for iterc in range( max_iter ):
        new = 0
        likelihood = 0
        #Set deltas to zero
        for j in range( config.n_feats ):
            deltas[j]['D']['alpha'] = 0.0
            deltas[j]['D']['beta'] = 0.0
            for k in v_alpha_params:
                deltas[j]['V']['alpha'][k] = 0.0
            for k in v_beta_params:
                deltas[j]['V']['beta'][k] = 0.0

        tsr = np.zeros( len(outtime) )
        #compute log scores and accumulate deltas
        for i,ts in enumerate(outtime):
            old = new
            log_score = np.zeros( config.n_feats , dtype = 'f8')
            for t in parts['t'][old:]:
                if t != ts:
                    break
                new += 1

            d,v,sr = parts['d'][old:new], parts['v'][old:new], parts['sr'][old:new]
            tsr[i] = sr.sum()

            for j in range( config.n_feats ):
                log_score[j] = np.sum( sr * logmodel( d, v, **params[j] ) )

            likelihood += log_score.sum()

            clear_output(wait=True)
            print( "Computing scores... Iteration {}/{}. Minute {}/{}. Likelihood: {:12.6g} Change: {:12.6g}".format(iterc + 1,max_iter,i+1,len(outtime),likelihood,likelihood-plike))

            log_score -= log_score.mean()
            group_scores[i]['t'] = ts
            group_scores[i]['fs'] = np.exp(log_score)/np.exp(log_score).sum()

            for j in range( config.n_feats ):
                factor = lr * group_scores[i]['fs'][j]
                partder_D_alpha = der_loggamma_alpha( d , **params[j]['D'] )
                partder_D_beta = der_loggamma_beta( d , **params[j]['D'] )
                partder_V_alpha = der_V_alpha( d, v, **params[j]['V'])
                partder_V_beta = der_V_beta( d, v, **params[j]['V'])
                deltas[j]['D']['alpha'] += factor * np.sum( sr * partder_D_alpha )
                deltas[j]['D']['beta'] += factor * np.sum( sr *  partder_D_beta )
                for k in v_alpha_params:
                    deltas[j]['V']['alpha'][k] += factor * np.sum( sr * partder_V_alpha[k] )
                for k in v_beta_params:
                    deltas[j]['V']['beta'][k] += factor * np.sum( sr * partder_V_beta[k] )

        for j in range( config.n_feats ):
            factor = np.sum( tsr * group_scores['fs'][:,j] )
            params[j]['D']['alpha'] += deltas[j]['D']['alpha'] / factor
            params[j]['D']['beta'] += deltas[j]['D']['beta'] / factor
            for k in v_alpha_params:
                params[j]['V']['alpha'][k] += deltas[j]['V']['alpha'][k] / factor
            for k in v_beta_params:
                params[j]['V']['beta'][k] += deltas[j]['V']['beta'][k] / factor

        pprint(params)
        if (input('Ok? ') == "no") and (likelihood < plike) and (input("Ok? " ) == 'no'):
            return params
        plike = likelihood
    return params

def reader( max_iter = 1000, lr = 0.05 ):
    data = {}
    shelf = shelve.open(fconfig['shelves'][outkey])
    shelf.clear()

    for e in events:
        data[e] = em( cpd[e][inkey] , max_iter = max_iter , lr = lr )

    globals()['data'] = data
    shelf['data'] = data
    shelf.close()

shelf = shelve.open(fconfig['shelves'][outkey])
if not 'data' in shelf:
    shelf.close()
    reader()
else:
    data = shelf['data']
    shelf.close()
del(shelf)
