from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from configs.naming_conventions import config as names

jan1epoch = 1388534400


def get_cmap(alpha):
    cmap = plt.get_cmap("hot")
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, [0, 2]] = my_cmap[:, [2, 0]]
    my_cmap[:, -1] = alpha
    return LinearSegmentedColormap.from_list("my_cmap", my_cmap, cmap.N)


def fill_zero(gps, tsr, outtime):
    L = len(outtime)
    count = 0
    for i in range(L - 1):
        offset = outtime[i + 1] - outtime[i]
        if offset > 2:
            count += 2

    gps_ = np.zeros((L + count, gps.shape[1]))
    tsr_ = np.zeros(L + count)
    outtime_ = np.zeros(L + count)

    count = 0
    for i in range(L - 1):
        offset = outtime[i + 1] - outtime[i]
        outtime_[i + count] = outtime[i]
        tsr_[i + count] = tsr[i]
        gps_[i + count, :] = gps[i, :]
        # If the distance between datapoints is greater than two minutes, add zeros in between them
        if offset > 2:
            count += 1
            outtime_[i + count] = outtime[i] + 1
            tsr_[i + count] = 0
            gps_[i + count, :] = 0
            count += 1
            outtime_[i + count] = outtime[i + 1] - 1
            tsr_[i + count] = 0
            gps_[i + count, :] = 0

    gps_[-1,:] = gps[-1]
    tsr_[-1] = tsr[-1]
    outtime_[-1] = outtime[-1]
    return gps_, tsr_, outtime_


def get_tsr(parts):
    parts = parts[(parts['v'] > 0)]  # & (parts['v'] < 4 ]
    outtime = np.unique(parts['t'])
    tsr = np.zeros(len(outtime))
    new = 0
    for i, ts in enumerate(outtime):
        old = new
        for t in parts['t'][old:]:
            if t != ts:
                break
            new += 1
        tsr[i] = parts['sr'][old:new].sum()
    return tsr


def get_weights(parts, dvmodel):
    parts = parts[(parts['v'] > 0)]  # & (parts['v'] < 4) ]
    outtime = np.unique(parts['t'])

    tsr = np.zeros(len(outtime))
    bytime = list()
    new = 0
    for i, ts in enumerate(outtime):
        old = new
        for t in parts['t'][old:]:
            if t != ts:
                break
            new += 1
        bytime.append(parts[old:new])
        tsr[i] = parts['sr'][old:new].sum()

    particle_weights = []
    for i, minute in enumerate(outtime):
        d, v, sr = bytime[i]['d'], bytime[i]['v'], bytime[i]['sr']
        mpart_weights = []
        minute_weights = []
        for j in range(len(dvmodel)):
            weights = sr * dvmodel[j](d, v)
            minute_weights.append(sum(weights))
            mpart_weights.append(weights)
        minute_weights = np.array(minute_weights) - max(minute_weights)
        for j in range(len(dvmodel)):
            mpart_weights[j] += minute_weights[j]
        mpart_weights = np.array(mpart_weights)
        mpart_weights -= np.max(mpart_weights)
        particle_weights.append(np.exp(mpart_weights))

    return np.concatenate(particle_weights, axis=1).transpose()


def plot(event, dvmodel, parts):
    parts = parts[(parts['v'] > 0)]  # & (parts['v'] < 4) ]
    pweights = get_weights(parts, dvmodel)
    v, d = np.mgrid[0.1:10:50j, 0.1:10:50j]

    # Make the fitted distribution plots
    fig = plt.figure(figsize=(20, 16), dpi=100)
    fig.suptitle("Distribution of Each Subpopulation for " + names['event_descriptions'][event], fontsize=32)
    im = None

    # Make the subplots
    for j in range(len(dvmodel)):
        max_d = 4 * np.average(parts['d'], weights=pweights[:, j])
        a, b, c = dvmodel[j]['V']['ratio'].flatten()
        d0 = d[0][d[0] < max_d]

        def line(x):
            return a + b * (1 - np.exp(-c * x))

        ax = fig.add_subplot(2, 2, j + 1)
        ans = dvmodel[j](d, v) / np.log(10)
        ax.set_title('Sub {}'.format(j + 1), fontsize=24)
        if j in {2, 3}:
            ax.set_xlabel('Diameter (mm)', fontsize=24)
        if j in {0, 2}:
            ax.set_ylabel('Velocity (m/s)', fontsize=24)

        rgba = np.zeros((len(parts), 4))
        rgba[:, 0] = 1
        rgba[:, 3] = 0.1 * pweights[:, j]

        ax.scatter(parts['d'], parts['v'], color=rgba)
        im = ax.pcolormesh(d, v, ans, vmin=-4, vmax=0, cmap=get_cmap(0.6))
        im.set_edgecolor(np.array([0, 0, 0, 0]))
        ax.plot(d0, line(d0), 'r', linewidth=2,
                label=r"${:.2f} + {:.2f}\cdot ( 1 - e^{}{:.3f}{})$".format(a, b, "{", c, "}"))

        if a + b * (1 - np.exp(-c * max_d)) > 4:
            ax.plot(d0, 9.65 - 10.3 * np.exp(-0.6 * d0), 'b', linewidth=2,
                    label=r"$9.65 - 10.3\cdot e^{-0.6 D}$")
        ax.legend(loc=1, fontsize=20)
        ax.tick_params(labelsize=16)
        ax.axis([0, 10, 0, 10])

    fig.tight_layout()
    fig.subplots_adjust(left=0.1, top=0.9, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(label='Log10 Likelihood', fontsize=20)
    cb.ax.tick_params(labelsize=16)
    fig.savefig('charts/mdd_test_gamma_{}'.format(event))
    plt.show()


def subpop_plot(event, parts, group_scores, outtime, rain_snow_labels, rnr=None, alpha=0.5):
    tsr = get_tsr(parts)
    gps = np.array(group_scores)
    gps, tsr, outtime = fill_zero(gps, tsr, outtime)
    tsr /= max(tsr)

    rscols = {
        'R': '#ee0000{:02x}',
        'S': '#0000ee{:02x}',
        'M': '#aa42f4{:02x}',
    }

    n = gps.shape[1]
    t = np.array([datetime.utcfromtimestamp(jan1epoch + 60 * minute) for minute in outtime])
    fig, ax = plt.subplots(n, figsize=(13, 8), sharex='all', sharey='all')
    fig.suptitle('Subpopulation Likelihoods for {}'.format(names['event_descriptions'][event]), fontsize=16)
    plt.minorticks_on()
    plt.gcf().autofmt_xdate()

    for i in range(n):
        z = gps[:, i]
        ax[i].grid(which='both')
        ax[i].set_axisbelow(True)
        ax[i].fill_between(t, tsr, -tsr, color='#0000003f')
        ax[i].fill_between(t, z, -z, color=[rscols[rain_snow_labels[i]].format(int(255*a)) for a in tsr*(1-alpha) + alpha])
        ax[i].set_ylabel("Sub {} ({})".format(i + 1, rain_snow_labels[i]), fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    for i in range(n):
        ax[i].set_yticklabels(abs(ax[i].yaxis.get_ticklocs()))

    fig.savefig('charts/mdd_test_gamma_subpop_{}'.format(event))
    plt.show()
