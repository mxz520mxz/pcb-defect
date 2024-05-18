import pyximport
from collections import defaultdict
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

from skimage import measure
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from skimage import measure
import random
from .cydefect import calc_edge_distmap

def cluster_defects(objs, shape, cluster_dist=32, nn_k=10, max_edge_points=0):
    nn_k = min(nn_k, len(objs))
    centers = [i['centroid'] for i in objs.values()]
    edge_list = []
    for k, o in objs.items():
        off = o['bbox'][:2]
        if 'contour' in o:
            edge_points = o['contour'].tolist()

        elif 'mask' in o:
            m = o['mask']
            m = np.pad(m, 1)
            edge_points = []
            for i in measure.find_contours(m):
                edge_points += (i[:,::-1] - 0.5 + off).astype('i4').tolist()
        else:
            raise

        if max_edge_points > 0:
            edge_points = random.sample(edge_points, min(max_edge_points, len(edge_points)))
        edge_list.append(np.asarray(edge_points, dtype=int))
    print('centers',type(centers))
    tree = NearestNeighbors(n_neighbors=nn_k, algorithm='ball_tree').fit(centers)
    dists, nn_idxs = tree.kneighbors(centers)
    N = len(centers)
    
    distmap = calc_edge_distmap(edge_list, nn_idxs)
    
    cluster = DBSCAN(cluster_dist, min_samples=1, metric='precomputed').fit(distmap)
    return cluster.labels_

def build_groups(objs):
    groups = {
        '-1': {
            'children': [],
        }
    }

    for k, o in objs.items():
        if 'group' not in o:
            groups[str(-1)]['children'].append(o['id'])
            continue

        gid = o['group']
        if gid not in groups:
            groups[gid] = {
                'children': []
            }
        groups[gid]['children'].append(o['id'])

    for gid, v in groups.items():
        if int(gid) < 0:
            continue

        children =[objs[oid] for oid in v['children']]
        xs = np.array([i['bbox'] for i in children])
        x0, y0 = xs[:,:2].min(axis=0)
        x1, y1 = xs[:,2:].max(axis=0)

        v['bbox'] = [x0, y0, x1, y1]
        v['area'] = sum(i['area'] for i in children)
        v['level_stats'] = defaultdict(int)
        for oid in v['children']:
            o = objs[oid]
            v['level_stats'][o['level']] += 1

        if 'black' in v['level_stats']:
            v['level'] = 'black'
        elif 'gray' in v['level_stats']:
            v['level'] = 'gray'
        elif 'white' in v['level_stats']:
            v['level'] = 'white'

    return groups
