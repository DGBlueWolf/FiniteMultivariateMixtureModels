import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from configs.naming_conventions import config as names
import src.compute.modeled_density as scmd
from src.compute.compute_dvd_from_particles import data
from src.compute.compute_dvd_from_particles import calc_snowrate_part as get_snowrate

def plotterlog(e):
    eps = 1e-12
    log = lambda x: np.log(x)/np.log(10)
    SR = data[e]['pip_particle_snowrate']['sr']
    bad = np.isnan(SR)
    D = data[e]['pip_particle_snowrate']['d'][~bad]
    V = data[e]['pip_particle_snowrate']['v'][~bad]
    SR = SR[~bad]

    x = y =  np.exp( np.arange( np.log(0.2), np.log(15), 0.1 ) )
    xt = yt = [2e-1,3e-1,5e-1,1e0,2e0,3e0,5,10,15]

    HN,yedges,xedges = np.histogram2d(V,D,bins=[y,x],normed=True)
    HSR,yedges,xedges = np.histogram2d(V,D,bins=[y,x],weights=SR,normed=True)
    HN *= len(D)
    HSR *= SR.sum()

    fig = plt.figure(figsize=(12,6.5),dpi=100)
    ax = fig.add_subplot(111)    # The big subplot
    ax.set_xlabel(r'Diamter (mm)')
    ax.set_ylabel(r'Velocity (m/s)')
    ax.set_title( 'Dia-Vel Log-Histogram for {}'.format(names['event_descriptions'][e]))

    ax1 = fig.add_subplot(1,2,1,adjustable='box', aspect=1)
    ax2 = fig.add_subplot(1,2,2,adjustable='box', aspect=1)

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    im1 = ax1.pcolor(log(xedges),log(yedges),10*log(HN+eps) , vmin=1 )
    ax1.set_xticks(log(xt))
    ax1.set_xticklabels(xt)
    ax1.set_yticks(log(yt))
    ax1.set_yticklabels(yt)
    ax1.set_title('Counts')
    cax1 = make_axes_locatable(ax1).append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im1,cax=cax1)

    im2 = ax2.pcolor(log(xedges),log(yedges),10*log(HSR+eps), vmin=-25 )
    ax2.set_xticks(log(xt))
    ax2.set_xticklabels(xt)
    ax2.set_yticks(log(yt))
    ax2.set_yticklabels(yt)
    ax2.set_title('Snowrate')
    cax2 = make_axes_locatable(ax2).append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2,cax=cax2)

def plotter(e):
    eps = 1e-12
    log = lambda x: np.log(x)/np.log(10)
    SR = data[e]['pip_particle_snowrate']['sr']
    bad = np.isnan(SR)
    D = data[e]['pip_particle_snowrate']['d'][~bad]
    V = data[e]['pip_particle_snowrate']['v'][~bad]
    SR = SR[~bad]

    x = y = (np.ogrid[0.1:15:0.1])
    xt = yt = list(range(1,15))

    HN,yedges,xedges = np.histogram2d(V,D,bins=[y,x],normed=True)
    HSR,yedges,xedges = np.histogram2d(V,D,bins=[y,x],weights=SR,normed = True)

    fig = plt.figure(figsize=(12,6.1),dpi=100)
    ax = fig.add_subplot(111)    # The big subplot
    ax.set_xlabel(r'Diamter (mm)')
    ax.set_ylabel(r'Velocity (m/s)')
    ax.set_title( 'Dia-Vel Log-Histogram for {}'.format(names['event_descriptions'][e]))

    ax1 = fig.add_subplot(1,2,1,adjustable='box', aspect=1)
    ax2 = fig.add_subplot(1,2,2,adjustable='box', aspect=1)

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    im1 = ax1.pcolor(xedges,yedges,10*log(HN+eps) , vmin=1 )
    ax1.set_xticks(xt)
    ax1.set_xticklabels(xt)
    ax1.set_yticks(yt)
    ax1.set_yticklabels(yt)
    ax1.set_title('Counts')
    cax1 = make_axes_locatable(ax1).append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im1,cax=cax1)

    im2 = ax2.pcolor(xedges,yedges,10*log(HSR+eps), vmin=-25 )
    ax2.set_xticks(xt)
    ax2.set_xticklabels(xt)
    ax2.set_yticks(yt)
    ax2.set_yticklabels(yt)
    ax2.set_title('Snowrate')
    cax2 = make_axes_locatable(ax2).append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2,cax=cax2)

def plotter_scmd(name):
    V, D = scmd.v, scmd.d
    HN = getattr(scmd,name)
    HSR = HN*get_snowrate(D,V)
    fig = plt.figure(figsize=(12,6.5),dpi=100)
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax.set_xlabel('Diamter (mm)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title( 'Dia-Vel Log-Histogram for Modelled Parameters')

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    im1 = ax1.imshow(log(HN), vmin=0, extent=[0,10,0,10], origin='lower')
    ax1.xaxis.set_ticks(np.ogrid[0:10.01:1])
    ax1.yaxis.set_ticks(np.ogrid[0:10.01:1])
    ax1.set_title('Counts')
    cax1 = make_axes_locatable(ax1).append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im1,cax=cax1)

    im2 = ax2.imshow(log(HSR), vmin=-4, extent=[0,10,0,10], origin='lower')
    ax2.xaxis.set_ticks(np.ogrid[0:10.01:1])
    ax2.yaxis.set_ticks(np.ogrid[0:10.01:1])
    ax2.set_title('Snowrate')
    cax2 = make_axes_locatable(ax2).append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2,cax=cax2)
    return fig

def plotter_dvd( dvd , e ):
    eps = 1e-12
    log = lambda x: np.log(x)/np.log(10)
    SR = data[e]['pip_particle_snowrate']['sr']
    bad = np.isnan(SR)
    #D = data[e]['pip_particle_snowrate']['d'][~bad]
    #V = data[e]['pip_particle_snowrate']['v'][~bad]
    SR = SR[~bad]

    y,x = np.mgrid[0.1:15.1:0.1,0.1:15.1:0.1]
    xt = yt = list(range(1,15))
    xedges = yedges = np.arange(0,15.1,0.1)
    HN = dvd(x,y)
    HSR = y*x**3*HN

    fig = plt.figure(figsize=(12,6.1),dpi=100)
    ax = fig.add_subplot(111)    # The big subplot
    ax.set_xlabel(r'Diamter (mm)')
    ax.set_ylabel(r'Velocity (m/s)')
    ax.set_title( 'Dia-Vel Log-Histogram from Model {}'.format(names['event_descriptions'][e]))

    ax1 = fig.add_subplot(1,2,1,adjustable='box', aspect=1)
    ax2 = fig.add_subplot(1,2,2,adjustable='box', aspect=1)

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    im1 = ax1.pcolor(xedges,yedges,10*log(HN+eps) , vmin=1 )
    ax1.set_xticks(xt)
    ax1.set_xticklabels(xt)
    ax1.set_yticks(yt)
    ax1.set_yticklabels(yt)
    ax1.set_title('Counts')
    cax1 = make_axes_locatable(ax1).append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im1,cax=cax1)

    im2 = ax2.pcolor(xedges,yedges,10*log(HSR+eps), vmin=1 )
    ax2.set_xticks(xt)
    ax2.set_xticklabels(xt)
    ax2.set_yticks(yt)
    ax2.set_yticklabels(yt)
    ax2.set_title('Snowrate')
    cax2 = make_axes_locatable(ax2).append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2,cax=cax2)
