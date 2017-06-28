import matplotlib.pyplot as plt
import src.compute.compute_mean_area_ratio_from_particles as mar
from configs.naming_conventions import config as names
from configs.event_details import config as events

def plot_event_ar(e):
    source = mar.data[mar.dkey1][e]
    time = source['tstart']
    beg,end = events['event_ranges'][e]
    idx = ( time < end ) & (time > beg)
    time = time[idx]
    ar = source['ar'][idx]
    bd = source['swer_bd'][idx]/source['ssdsr'][idx]
    plt.plot(time,ar,color="blue")
    plt.plot(time,bd,color="red")
    plt.xlabel('Time')
    plt.ylabel('Computed Area Ratio')
    plt.ylim([0,2])
    plt.title(names['event_descriptions'][e])
