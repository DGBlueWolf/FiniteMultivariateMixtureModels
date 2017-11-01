import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from configs.naming_conventions import config as names

jan1epoch = 1388534400


def plot(event, dvmodel, parts):
    v, d = np.mgrid[0.1:10:100j, 0.1:10:100j]
    max_d = max(parts['d'])

    # Make the fitted distribution plots
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Distribution of Each Subpopulation", fontsize=16)
    im = None
    for j in range(len(dvmodel)):
        a, b, c = dvmodel[j]['V']['ratio'].flatten()
        d0 = d[0][d[0] < 5]

        def line(x):
            return a + b * (1 - np.exp(-c * x))

        ax = fig.add_subplot(2, 2, j + 1)
        ans = dvmodel[j](d, v) / np.log(10)
        ax.set_title('Sub {}'.format(j + 1), fontsize=12)
        if j in {2, 3}:
            ax.set_xlabel('Diameter (mm)', fontsize=12)
        if j in {0, 2}:
            ax.set_ylabel('Velocity (m/s)')

        #ax.scatter()
        im = ax.pcolor(d, v, ans, vmin=-4, vmax=0)
        ax.plot(d0, line(d0), 'r', linewidth=2,
                label=r"${:.2f} + {:.2f}\cdot ( 1 - e^{}{:.3f}{})$".format(a, b, "{", c, "}"))

        if a + b*(1-np.exp(-c*max_d)) > 4:
            ax.plot(d0, 9.65 - 10.3 * np.exp(-0.6 * d0), 'b', linewidth=2,
                    label=r"$9.65 - 10.3\cdot e^{-0.6 D}$")
        ax.legend(loc=1)
        ax.axis([0, 10, 0, 10])

    fig.tight_layout()
    fig.subplots_adjust(left=0.1, top=0.9, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig('charts/mdd_test_gamma_{}'.format(event))
    plt.show()


def subpop_plot(event, group_scores, outtime):
    gps = np.array(group_scores)
    n = gps.shape[1]
    t = np.array([datetime.utcfromtimestamp(jan1epoch + 60 * minute) for minute in outtime])
    fig, ax = plt.subplots(n, figsize=(13, 8), sharex=True, sharey=True)
    fig.suptitle('Subpopulation Likelihoods for {}'.format(names['event_descriptions'][event]), fontsize=16)
    plt.minorticks_on()
    plt.gcf().autofmt_xdate()

    for i in range(n):
        z = group_scores[:, i]
        ax[i].grid(which='both')
        ax[i].set_axisbelow(True)
        ax[i].fill_between(t, z, -z, color="C{}".format(i))

        ax[i].set_ylabel("Sub {}".format(i + 1), fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    for i in range(n):
        ax[i].set_yticklabels(abs(ax[i].yaxis.get_ticklocs()))

    fig.savefig('charts/mdd_test_gamma_subpop_{}'.format(event))
    plt.show()
