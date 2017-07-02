from utilities import base
home = 'C:/Users/7110008/Documents/Employment/NASA Fall Internship/SnowDensityProject/'

n_feats = 4
err_feats = 1
dataformat = [
    (('Time','t'),'i4'),
    (('Cluster Diameter','d'),'f8'),
    (('Cluster Velocity','v'),'f8'),
    (('Cluster SnowRate','sr'),'f8'),
    (('Cluster Scores','scores'),'{}f8'.format(n_feats + err_feats)),
    (('Cluster Label','label'),'i4')
]
