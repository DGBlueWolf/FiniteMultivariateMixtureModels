import shelve

import numpy as np
from IPython.display import clear_output
from numpy import exp, log

from configs.file_locations import config as fconfig
from configs.naming_conventions import config as names
from src.compute.mdd.rain_snow_models import RainDV, SnowDV

np.set_printoptions(precision=3, formatter={'float_kind': lambda x: "%.2f" % x})

n_rain, n_snow = 3, 3
v_alpha_params = ['slope', 'scale', 'intercept']
v_beta_params = ['slope', 'scale', 'intercept']
inkey = 'compressed_pip_data'
outkey = 'gamma_model_mixing_ratio'
events = names['events']
rain_events = ['r1', 'r2', 'e1', 'e4', 'e5']
snow_events = ['e2', 'e3', 'e6', 'e7', 'e8']
minlog = np.exp(-40)
data = {}


def em(parts, max_iter=500, tol=1e-6, theta=0.8, all_snow=False):
    parts = parts[(parts['v'] > 0)]
    if all_snow:
        parts = parts[(parts['v'] < 4)]
    outtime = np.unique(parts['t'])
    meanchange = 0

    # parameters for the distribution model
    mxmodel = [RainDV() for _ in range(n_rain)] + [SnowDV() for _ in range(n_snow)]

    # define a structure to hold the scores for each model in the mixture
    group_scores = np.zeros((len(outtime), n_rain + n_snow ))
    part_scores = np.zeros((len(parts), n_rain + n_snow))

    new = 0
    # total snow_rate for each minute and the particles for each minute
    count_per_minute = np.zeros(len(outtime))
    bytime = list()
    for i, ts in enumerate(outtime):
        old = new
        for t in parts['t'][old:]:
            if t != ts:
                break
            new += 1
        bytime.append(parts[old:new])
        count_per_minute[i] = parts['count'][old:new].sum()

    # run the algorithm
    plike = 0
    obias = 0.1
    bias = obias
    for iterc in range(max_iter):
        likelihood = 0
        idx = 0
        # compute log scores and accumulate deltas
        for minute, (i, ts) in enumerate(sorted(list(enumerate(outtime)),
                                                key=lambda x: tuple(group_scores[x[0], :])), 1):
            total_log_score = np.zeros((n_rain + n_snow,))
            log_score = np.zeros((len(bytime[i]), n_snow + n_rain))
            d, v, count = bytime[i]['d'], bytime[i]['v'], bytime[i]['count']

            for j in range(n_rain + n_snow):
                log_score[:, j] = count * mxmodel[j](d, v)
                total_log_score[j] = log_score[:, j].sum()

            likelihood += log(max(exp(total_log_score).mean(), 1e-250))
            total_log_score -= total_log_score.max()
            log_score -= log_score.max(axis=0)
            s = np.exp(total_log_score).sum()
            ps = np.exp(log_score).sum(axis=1)
            for j in range(n_rain + n_snow):
                for k in range(len(bytime[i])):
                    part_scores[idx + k, j] = 0.25 if np.isnan(ps[k]) or s < 1e-20 else 0.25*bias + (1-bias)*np.exp(log_score[k, j]) / ps[k]

            group_scores[i, :] = 0.25 if np.isnan(s) or s < 1e-20 else 0.25*bias + (1-bias)*np.exp(total_log_score) / s
            clear_output(wait=True)
            idx += len(bytime[i])
            print("Computing scores... Iteration {}/{}. Minute {:4d}/{:4d}. "
                  "Likelihood: {:10.6g} Mean Change: {:12.6g}\nGroupScores: {}".format(
                    iterc + 1, max_iter, minute, len(outtime),
                    plike, meanchange, group_scores.sum(axis=0)))

        for j in range(n_rain + n_snow):
            mxmodel[j].optimize((parts['d'], parts['v'], parts['count'], part_scores[:, j]))

        bias *= obias
        change = plike and likelihood - plike
        meanchange *= (1 - theta)
        meanchange += theta * change
        if plike and exp(-(
                    log(10) / (max_iter - iterc)) * meanchange / count_per_minute.sum()) > 1 - tol:
            break
        plike = likelihood
    return mxmodel, group_scores, count_per_minute, outtime


def reader(r_events = None, **kwargs):
    from src.compute.compress_particle_data import data as cpd
    r_events = r_events or events
    data = globals()['data']
    for e in r_events:
        data[e] = {}
        data[e]['params'], data[e]['group_scores'], \
        data[e]['tsr'], data[e]['outtime'] = em(cpd[e][inkey], **kwargs)


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
