import shelve
import numpy as np
from pprint import pprint
from scipy.special import digamma
from IPython.display import clear_output
from scipy.stats import gamma
from scipy.stats import norm
from configs.file_locations import config as fconfig
import configs.compute.model_dv_distribution as config
from configs.naming_conventions import config as names

v_alpha_params = ['slope','scale','intercept']
v_beta_params = ['slope','scale','intercept']
inkey = 'compressed_pip_data'
outkey = 'model_dv_distribution'
events = names['events']
minlog = np.exp(-40)
data = {}

def init_v_alpha():
    return {
        'slope': np.random.uniform(3.5,4.5),
        'scale': np.random.uniform(0.5,1.5),
        'intercept': np.random.uniform(1,2.5),
    }

def init_v_beta():
    return {
        'slope': np.random.uniform(1,3),
        'scale': np.random.uniform(0.5,1.5),
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
    tmp = beta*d
    tmp[ tmp < minlog ] = minlog
    return np.log( tmp )-digamma(alpha)

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

    tmp1 = gamma.pdf( v, av, 0, 1/bv )
    if len(tmp1.shape) > 0 :
        tmp1[ tmp1 < minlog ] = minlog
    elif tmp1 < minlog:
        tmp1 = minlog

    tmp2 = gamma(D['alpha'],0,1/D['beta']).pdf( d )
    if len(tmp2.shape) > 0 :
        tmp2[ tmp2 < minlog ] = minlog
    elif tmp2 < minlog:
        tmp2 = minlog

    vd = np.log( tmp1 )
    return np.log( tmp2 ) + vd

def em( parts , max_iter=500 , thresh = 0.05 , lr = 0.05 , init_ = None):
    #parts = parts[ (parts['v'] > 0) & (parts['v'] < 4) ]
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
    if init_ is None:
        for j in range( config.n_feats ):
            params[j]['D']['alpha'] = np.random.uniform(3,8)
            params[j]['D']['beta'] = np.random.uniform(1,6)
            params[j]['V']['alpha'].update( init_v_alpha() )
            params[j]['V']['beta'].update( init_v_beta() )
    else:
        for j in range( config.n_feats ):
            params[j]['D']['alpha'] = init_[j]['D']['alpha']
            params[j]['D']['beta'] = init_[j]['D']['beta']
            params[j]['V']['alpha'].update( init_[j]['V']['alpha'] )
            params[j]['V']['beta'].update( init_[j]['V']['beta'] )

    #define error model parameters
    #pprint( params )
    #if input('Ok? ') == "no":
    #    return params

    #define a structure to hold the scores for each model in the mixture
    group_type = [
        (('Time','t'),'i4'),
        (('Feature scores','fs'),'{}f8'.format(config.n_feats)),
        (('Error scores','err'),'{}f8'.format(config.err_feats)),
    ]
    group_scores = np.zeros( len(outtime) , dtype = group_type )
    tsr = 0
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

            clear_output(wait=True)
            print( "Computing scores... Iteration {}/{}. Minute {}/{}. Likelihood: {:10.6g} Change This Run: {:12.6g}".format(iterc + 1,max_iter,i+1,len(outtime),plike,likelihood-plike))
            #pprint(deltas)

        for j in range( config.n_feats ):
            factor = np.sum( tsr * group_scores['fs'][:,j] )
            factor = 1 if factor < 1 else factor
            dalpha = deltas[j]['D']['alpha'] / factor
            dbeta = deltas[j]['D']['beta'] / factor

            #DIAMETER PART
            if dalpha + 1e-6 < -params[j]['D']['alpha']:
                params[j]['D']['alpha'] *= 0.5
            else:
                params[j]['D']['alpha'] += dalpha

            if dbeta + 1e-6 < -params[j]['D']['beta']:
                params[j]['D']['alpha'] *= 0.5
            else:
                params[j]['D']['beta'] += dbeta

            #VELOCITY PART
            for k in v_alpha_params:
                valpha = deltas[j]['V']['alpha'][k] / factor
                if valpha + 1e-6 < -params[j]['V']['alpha'][k]:
                    params[j]['V']['alpha'][k] *= 0.5
                else:
                    params[j]['V']['alpha'][k] += valpha

            for k in v_beta_params:
                vbeta = deltas[j]['V']['beta'][k] / factor
                if vbeta + 1e-6 < -params[j]['V']['beta'][k]:
                    params[j]['V']['beta'][k] *= 0.5
                else:
                    params[j]['V']['beta'][k] += vbeta

        #pprint(params)
        if ( np.isnan(likelihood)  or (likelihood < plike) ) and (input("Ok? " ) == 'no'):
            pprint(params)
            break

        if likelihood - plike < thresh*abs(likelihood)/(max_iter-iterc):
            break

        plike = likelihood
    return params, group_scores, ( group_scores['fs'].transpose() * tsr ).transpose()

def reader( max_iter = 500, lr = 0.04 ):
    from src.compute.compress_particle_data import data as cpd
    data = globals()['data']
    for e in events:
        data[e] = {}
        data[e]['params'], data[e]['group_scores'], data[e]['group_sr'] = em( cpd[e][inkey] , max_iter = max_iter , lr = lr )

def save():
    shelf = shelve.open(fconfig['shelves'][outkey])
    shelf.clear()
    shelf['data'] = globals()['data']
    shelf.close()

shelf = shelve.open(fconfig['shelves'][outkey])
if not 'data' in shelf:
    shelf.close()
    reader()
    save()
else:
    data = shelf['data']
    shelf.close()
del(shelf)
