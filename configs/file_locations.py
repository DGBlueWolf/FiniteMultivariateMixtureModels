from utilities import base
home = 'C:/Users/7110008/Documents/Employment/NASA Fall Internship/SnowDensityProject/'
config = {
    #readme locations
    'readme': base.specfiles({
        'particle_density': 'data/particle_density/',
    }, base = home, ext = "README.txt"),

    #print format
    'printformat': {
        'f8': '{:12.6g}',
        'i4': '{:8d}',
        'separator': ',',
    },

    #source files locations and such, a view of the src directory
    'sources': {
        'readers': base.specfiles({
            'meta':                    'read_meta',
            'particle_density':        'read_particle_density',
            'rain_rate':               'read_rr',
            'snow_size_distribution':  'read_ssd',
            'velocity_distribution':   'read_vd',
        }, base = home + 'src/read', ext = ".py"),

        'computers': base.specfiles({
            'compressed_particle_data': 'compress_particle_data',
            'computed_snow_rate':  'compute_dvd_from_particles',
            'computed_area_ratio': 'compute_mean_area_ratio_from_particles',
        }, base = home + 'src/compute', ext = '.py'),

        'charts': base.specfiles({
            'plot_bohms_vs_powerlaw': 'plot_bohms',
        }, base = home + 'src/charts', ext = ".py")
    },

    #shelf files for efficiently reloading data
    'shelves': base.specfiles({
        'meta':                     'meta',
        'snow_size_distribution':   'ssd',
        'velocity_distribution':    'vd',
        'rain_rate':                'rr',
        'particle_density':         'pd',
        'computed_snow_rate':       'csr',
        'computed_area_ratio':      'car',
        'compressed_particle_data': 'cpd',
    }, base = home + 'data/shelves/' , ext = '.shelf'),

    #MetaData
    'meta': {
        'pip_bin_centers': {
            'dataformat': [
                (("Bin Centers","bins"),'f8',131),
            ],
            'files': base.specfiles({
                'bins': 'pip02MMbins',
            }, base = home + "data/meta/", ext = ".csv"),
        },
        'pip_info': {
            'dataformat': [
                (('Bin Spacing (mm)','binwidth'),'f8'),
                (('Measurement Window (mm)','window'),[('x','f8'),('y','f8')]),
            ],
            'files': base.specfiles({
                'pipinfo': 'pipinfo'
            }, base = home + "data/meta/", ext = ".csv"),
        },
    },

    #Snow Size Distribution
    'snow_size_distribution':{
        'shelf': home + 'data/shelves/velocity_distribution.shelf',
        'dataformat':[
            (('Day','day'),'i4'),
            (('Minute','t'),'i4'),
            (('Snow-size Distribution','ssd'),'f8',131)
        ],
        'files': base.specfiles({
            'r1': 'ssd20131209',
            'e1': 'ssd20140102-03',
            'e2': 'ssd20140121-22',
            'e3': 'ssd20140128-29',
            'r2': 'ssd20140203-04',
            'e4': 'ssd20140213-14',
            'e5': 'ssd20140215',
            'e6': 'ssd20140303',
            'e7': 'ssd20140317',
            'e8': 'ssd20140325-26',
        }, base = home + "data/ssd/", ext = ".csv")
    },

    #Velocity Distribution
    'velocity_distribution':{
        'shelf': home + 'data/shelves/velocity_distribution.shelf',
        'dataformat':[
            (('Day','day'),'i4'),
            (('Minute','t'),'i4'),
            (('Velocity Distribution','vd'),'f8',131)
        ],
        'files': base.specfiles({
            'r1': 'vd20131209',
            'e1': 'vd20140102-03',
            'e2': 'vd20140121-22',
            'e3': 'vd20140128-29',
            'r2': 'vd20140203-04',
            'e4': 'vd20140213-14',
            'e5': 'vd20140215',
            'e6': 'vd20140303',
            'e7': 'vd20140317',
            'e8': 'vd20140325-26',
        }, base = home + "data/vd/", ext = ".csv")
    },

    #pluvio snow water equivalent rate
    'rain_rate':{
        'shelf': home + 'data/shelves/rain_rate.shelf',
        'dataformat':[
            (('Day','day'),'i4'),
            (('Time','t'),'i4'),
            (('Rain rate','rr'),'f8'),
            (('Totals','tot'),'f8'),
        ],
        'files': base.specfiles({
            'r1': 'R1',
            'e1': 'E1',
            'e2': 'E2',
            'e3': 'E3',
            'r2': 'R2',
            'e4': 'E4',
            'e5': 'E5',
            'e6': 'E6',
            'e7': 'E7',
            'e8': 'E8',
        }, base = home + "data/rr/", ext = ".csv")
    },

    #particle density file locations
    'particle_density':{
        'shelf': home + 'data/shelves/particle_density.shelf',
        'dataformat':[
            (('Record Number','rec'),'f8'),
            (('Particle Number','part'),'f8'),
            (('Diameter','d'),'f8'),
            (('Horizontal Velocity','vx'),'f8'),
            (('Vertical Velocity','vy'),'f8'),
            (('Time Index (minutes)','t'),'i4'),
            (('Air Density','rhoA'),'f8'),
            (('Air Viscosity','etaA'),'f8'),
            (('Particle Mass','m'),'f8'),
            (('Particle Density','rhoS'),'f8'),
            (('Reynolds Number','rn'),'f8'),
        ],
        'files': base.specfiles({
            'r1': 'pd20131209.csv',
            'e1': 'pd20140102-03.csv',
            'e2': 'pd20140121-22.csv',
            'e3': 'pd20140128-29.csv',
            'r2': 'pd20140203-04.csv',
            'e4': 'pd20140213-14.csv',
            'e5': 'pd20140215.csv',
            'e6': 'pd20140303.csv',
            'e7': 'pd20140317.csv',
            'e8': 'pd20140325-26.csv',
        }, base = home + "data/particle_density/", ext = "")
    },

    #Compressed particle information
    'compressed_particle_data': {
        'compressed_pip_data':{
            'dataformat': [
                (('Time','t'),'i4'),
                (('Cluster Diameter','d'),'f8'),
                (('Cluster Velocity','v'),'f8'),
                (('Cluster SnowRate','sr'),'f8'),
                (('Cluster Varaince','var'),'f8'),
            ],
            'files': base.specfiles({
                'r1': 'clustersr20131209',
                'e1': 'clustersr20140102-03',
                'e2': 'clustersr20140121-22',
                'e3': 'clustersr20140128-29',
                'r2': 'clustersr20140203-04',
                'e4': 'clustersr20140213-14',
                'e5': 'clustersr20140215',
                'e6': 'clustersr20140303',
                'e7': 'clustersr20140317',
                'e8': 'clustersr20140325-26',
            }, base = home + "data/computed/snow_rate/", ext = ".csv")
        },
    },
    #Computed snowrates
    'computed_snow_rate': {
        'pip_particle_snowrate': {
            'dataformat': [
                (('Time','t'),'i4'),
                (('Particle Diameter','d'),'f8'),
                (('Particle Velocity','v'),'f8'),
                (('Bohms Density','bd'),'f8'),
                (('Particle SnowRate','sr'),'f8'),
            ],
            'files': base.specfiles({
                'r1': 'partsr20131209',
                'e1': 'partsr20140102-03',
                'e2': 'partsr20140121-22',
                'e3': 'partsr20140128-29',
                'r2': 'partsr20140203-04',
                'e4': 'partsr20140213-14',
                'e5': 'partsr20140215',
                'e6': 'partsr20140303',
                'e7': 'partsr20140317',
                'e8': 'partsr20140325-26',
            }, base = home + "data/computed/snow_rate/", ext=".csv"),
        },

        'pip_ssd_snow_rate': {
            'dataformat': [
                (('Time','t'),'i4'),
                (('Bohms Density','bd'),'f8'),
                (('Snow Rate','sr'),'f8'),
            ],
            'files': base.specfiles({
                'r1': 'ssdsr20131209',
                'e1': 'ssdsr20140102-03',
                'e2': 'ssdsr20140121-22',
                'e3': 'ssdsr20140128-29',
                'r2': 'ssdsr20140203-04',
                'e4': 'ssdsr20140213-14',
                'e5': 'ssdsr20140215',
                'e6': 'ssdsr20140303',
                'e7': 'ssdsr20140317',
                'e8': 'ssdsr20140325-26',
            }, base = home + "data/computed/snow_rate/", ext=".csv"),
        },
    },

    #Computed area ratios
    'computed_area_ratio': {
        'sliding_window_averages': {
            'sliding_window_width': 60, #minutes
            'dataformat': [
                (('Window Start Time','tstart'),'i4'),
                (('Window Stop Time','tstop'),'i4'),
                (('Window Count','count'),'i4'),
                (('Pluvio SWER','swer'),'f8'),
                (('Bohms SWER','swer_bd'),'f8'),
                (('Area Ratio','ar'),'f8'),
                (('Snow Rate','ssdsr'),'f8'),
                (('Window Start Index','istart'),'i4'),
                (('Window Stop Index','istop'),'i4'),
            ],
            'files': base.specfiles({
                'r1': '20131209',
                'e1': '20140102-03',
                'e2': '20140121-22',
                'e3': '20140128-29',
                'r2': '20140203-04',
                'e4': '20140213-14',
                'e5': '20140215',
                'e6': '20140303',
                'e7': '20140317',
                'e8': '20140325-26',
            }, base = home + "data/computed/area_ratio/arts", ext=".csv"),
        },
        'event_summary': {
            'dataformat': [
                (('Pluvio Mean SWER','swer'),'f8'),
                (('Mean Snow Rate','ssdsr'),'f8'),
                (('Mean Bohms Density','bd'),'f8'),
                (('Mean Area Ratio','arave'),'f8'),
                (('Standard Deviation of Area Ratio','arstd'),'f8'),
                (('Total Minutes','minutes'),'i4'),
            ],
            'files': base.specfiles({
                'summary': 'summary',
            }, base = home + "data/computed/area_ratio/", ext=".csv"),
        }
    },
}
