import shelve
import numpy as np
from IPython.display import clear_output
from numpy import exp, log
from numpy.random import permutation as permute
from configs.file_locations import config as fconfig
import configs.compute.model_dv_distribution as config
from configs.naming_conventions import config as names
from src.compute.mdd.rbdvmodel_new import DVModel

v_alpha_params = ['slope', 'scale', 'intercept']
v_beta_params = ['slope', 'scale', 'intercept']
inkey = 'compressed_pip_data'
outkey = 'model_dv_distribution'
events = names['events']
minlog = np.exp(-40)
data = {}


def em(parts, max_iter=500, tol=1e-6, theta=0.1, init_=None):
    parts = parts[(parts['v'] > 0)]  # & (parts['v'] < 4) ]
    outtime = np.unique(parts['t'])
    meanchange = 0

    # parameters for the distribution model
    mxmodel = [DVModel() for _ in range(config.n_feats)] \
        if init_ is None else \
        [DVModel(init_[i]) for i in range(config.n_feats)]

    # define a structure to hold the scores for each model in the mixture

    group_scores = np.zeros((len(outtime), config.n_feats))

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
    plike = 0
    for iterc in range(max_iter):
        likelihood = 0

        # compute log scores and accumulate deltas
        for minute, (i, ts) in enumerate(permute(list(enumerate(outtime))), 1):
            log_score = np.zeros(config.n_feats, dtype='f8')
            d, v, sr = bytime[i]['d'], bytime[i]['v'], bytime[i]['sr']

            for j in range(config.n_feats):
                log_score[j] = np.sum(sr * mxmodel[j](d, v))

            likelihood += log(exp(log_score).mean())
            log_score -= log_score.mean()
            group_scores[i, :] = np.exp(log_score) / np.exp(log_score).sum()
            clear_output(wait=True)
            print(
                "Computing scares... Iteration {}/{}. Minute {:4d}/{:4d}. Likelihood: {:10.6g} Mean Change: {:12.6g}".format(
                    iterc + 1, max_iter, minute, len(outtime), plike, meanchange))

        for j in range(config.n_feats):
            mxmodel[j].optimize(list(zip(bytime, group_scores[:, j])))

        change = plike and likelihood - plike
        meanchange *= (1 - theta)
        meanchange += theta * change
        if plike and exp(-(
                    log(10) / (max_iter - iterc)) * meanchange / tsr.sum()) > 1 - tol:
            break

        plike = likelihood
    return mxmodel, group_scores, tsr, outtime


def reader(**kwargs):
    from src.compute.compress_particle_data import data as cpd
    data = globals()['data']
    r_events = [x.strip() for x in input().split(',')]
    for e in r_events:
        data[e] = {}
        data[e]['params'], data[e]['group_scores'], data[e]['tsr'], data[e]['outtime'] = em(cpd[e][inkey], **kwargs)


def save():
    shelf = shelve.open(fconfig['shelves'][outkey])
    shelf.clear()
    shelf['data'] = globals()['data']
    shelf.close()


shelf = shelve.open(fconfig['shelves'][outkey])
if 'data' not in shelf:
    shelf.close()
    reader()
    save()
else:
    data = shelf['data']
    shelf.close()
del shelf
