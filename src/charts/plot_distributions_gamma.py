import numpy as np
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from configs.naming_conventions import config as names
jan1epoch = 1388534400

def plot( event, DVModel ):
    v,d = np.mgrid[0.1:10:100j,0.1:10:100j]

    #Make the fitted distribution plots
    fig = plt.figure( figsize = (10,8) )
    fig.suptitle( "Distribution of Each Subpopulation", fontsize = 16)
    for j in range(len(DVModel)):
        ax = fig.add_subplot(2,2,j+1)
        ans = DVModel[j]( d, v )/np.log(10)
        ax.set_title( 'Sub {}'.format(j+1) , fontsize = 12)
        if j in {2,3}:
            ax.set_xlabel('Diameter (mm)' , fontsize = 12)
        if j in {0,2}:
            ax.set_ylabel('Velocity (m/s)')
        im = ax.pcolor(d,v,ans,vmin=-4,vmax=0)
        '''
        ax.plot( d[0] , DVModel[j].line(d[0]) , 'r', linewidth = 2,
            label = r"${:.2f} + {:.2f}\cdot ( 1 - e^{}{:.3f}{})$".format( DVModel[j].a, DVModel[j].b, "{", DVModel[j].c, "}" ) )
        if( DVModel[j].a + DVModel[j].b > 4 ):
            ax.plot( d[0] , 9.65 - 10.3*np.exp(-0.6*d[0]) , 'b', linewidth = 2 ,
                label = r"$9.65 - 10.3\cdot e^{-0.6 D}$" )
        '''
        ax.legend(loc=1)
        ax.axis([0,10,0,10])

    fig.tight_layout()
    fig.subplots_adjust(left = 0.1, top = 0.9, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig( 'charts/mdd_test_gamma_{}'.format(event))
    plt.show()

def subpop_plot( event, group_scores ):
    gps = np.array(group_scores)
    N = gps['fs'].shape[1]
    t = np.array([ datetime.utcfromtimestamp( jan1epoch + 60*minute ) for minute in gps['t']])
    fig, ax = plt.subplots(N, figsize = (13,8), sharex = True, sharey = True )
    fig.suptitle('Subpopulation Likelihoods for {}'.format(names['event_descriptions'][event]), fontsize = 16)
    plt.minorticks_on()
    plt.gcf().autofmt_xdate()

    for i in range(N):
        z = group_scores['fs'][:,i]
        ax[i].grid( which = 'both')
        ax[i].set_axisbelow(True)
        ax[i].fill_between( t, z , -z , color = "C{}".format(i) )

        ax[i].set_ylabel( "Sub {}".format(i+1) , fontsize = 12)

    fig.tight_layout()
    fig.subplots_adjust( top = 0.9 )

    for i in range(N):
        ax[i].set_yticklabels( abs(ax[i].yaxis.get_ticklocs()) )

    fig.savefig( 'charts/mdd_test_gamma_subpop_{}'.format(event) )
    plt.show()
