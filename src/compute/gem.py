import shelve
import numpy as np
from pprint import pprint
from scipy.special import digamma
from IPython.display import clear_output
from numpy import exp, log, sqrt
from numpy.random import permutation as permute
from configs.file_locations import config as fconfig
import configs.compute.model_dv_distribution as config
from configs.naming_conventions import config as names
from src.compute.mdd.lognormal_dvmodel import DVModel

inkey = 'compressed_pip_data'
outkey = 'model_dv_distribution'
events = names['events']
minlog = np.exp(-40)
data = {}

class Gem:
    def __init__(self, model_ ):
        self.model = model_  # a Mixture Model which can be called on an observation to produce a log probability score

    def train(self, data):
        scores =
        for obs in data:



def em(parts, max_iter=500, tol=1e-6, theta=0.1, lr=0.01, batches=16, init_=None):
    parts = parts[(parts['v'] > 0) & (parts['sr'] > 0)]
    outtime = np.unique(parts['t'])
    batch_counter = np.zeros(config.n_feats)
    batch_goal = np.zeros(config.n_feats) + 1.2e10
    batch_sum = np.zeros(config.n_feats)
    meanchange = 0

    # parameters for the distribution model
    mxmodel = [DVModel() for _ in range(config.n_feats)] \
        if init_ is None else \
        [DVModel(init_[i], ) for i in range(config.n_feats)]

    print(mxmodel)
    # define a structure to hold the scores for each model in the mixture
    group_type = [
        (('Time', 't'), 'i4'),
        (('Feature scores', 'fs'), '{}f8'.format(config.n_feats)),
        (('Error scores', 'err'), '{}f8'.format(config.err_feats)),
    ]
    group_scores = np.zeros(len(outtime), dtype=group_type)

    new = 0
    # total snow_rate for each minute and the particles for each minute
    tsr = np.zeros(len(outtime))
    bytime = list()
    for i, ts in enumerate(outtime):
        old = new
        for t in parts['t'][old:]:
            if t != ts:
                break
            new += 1
        bytime.append(parts[old:new])
        tsr[i] = parts['sr'][old:new].sum()

    # run the algorithm
    change = 0
    plike = 0
    for iterc in range(max_iter):
        likelihood = 0
        batch_sum *= 0.0

        # compute log scores and accumulate deltas
        for minute, (i, ts) in enumerate(permute(list(enumerate(outtime))), 1):
            log_score = np.zeros(config.n_feats, dtype='f8')
            d, v, sr = bytime[i]['d'], bytime[i]['v'], bytime[i]['sr']

            for j in range(config.n_feats):
                log_score[j] = np.sum(sr * mxmodel[j](d, v))

            likelihood += log(exp(log_score).sum())
            log_score -= log_score.mean()
            group_scores[i]['t'] = ts
            group_scores[i]['fs'] = np.exp(log_score) / np.exp(log_score).sum()

            batch_counter += tsr[i] * group_scores[i]['fs']
            batch_sum += tsr[i] * group_scores[i]['fs']

            for j in range(config.n_feats):
                mxmodel[j].prep(d, v, sr, group_scores[i]['fs'][j])

            for j in range(config.n_feats):
                if batch_counter[j] > batch_goal[j]:
                    clear_output(wait=True)
                    mxmodel[j].step()
                    print(
                    "Computing scares... Iteration {}/{}. Minute {:4d}/{:4d}. Likelihood: {:10.6g} Mean Change: {:12.6g}".format(
                        iterc + 1, max_iter, minute, len(outtime), plike, meanchange))
                    batch_counter[j] = 0.0

        change = plike and likelihood - plike
        meanchange *= (1 - theta)
        meanchange += theta * change
        if plike and exp(-(
            log(10) / (max_iter - iterc)) * meanchange / tsr.sum()) > 1 - tol:  # and (input("Ok? " ) == 'no'):
            # pprint(mxmodel)
            # plot( 'r1', mxmodel )
            # if( input("Continue? ") == 'no' ):
            break

        batch_goal = batch_sum / batches
        plike = likelihood
    return mxmodel, group_scores, tsr


def reader(**kwargs):
    from src.compute.compress_particle_data import data as cpd
    data = globals()['data']
    for e in events:
        data[e] = {}
        data[e]['params'], data[e]['group_scores'], data[e]['tsr'] = em(cpd[e][inkey], **kwargs)


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
del (shelf)
