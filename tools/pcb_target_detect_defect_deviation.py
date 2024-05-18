#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import toml
import numpy as np
from functools import partial
from easydict import EasyDict as edict
import cv2

import json
import pickle
from copy import deepcopy

from deeppcb.base import file_name, imread, imsave, Image
from deeppcb.target import read_list
from deeppcb.utils import run_parallel, get_zoom, update_zoomed_len
from deeppcb.deviation import detect_deviations
from deeppcb.draw import draw_defects

import click

def process_one(name, segmaps, defects_dir, cls_info, templates, verify, no_valid_mask, zoom, C):
    print('-------------', name)

    zoom = get_zoom(zoom, segmaps)

    ci = cls_info[name]
    cam_id = ci['cam_id']
    tpl_dir = templates.format(board=ci['board'])
    fname = osp.join(tpl_dir, f'distmaps/{cam_id}.jpg')
    tpl_distmap = imread(fname, to_gray=True)

    fname = osp.join(tpl_dir, f'segmaps/{cam_id}.png')
    tpl_segmap = imread(fname, to_gray=True)

    fname = osp.join(segmaps, name+'.png')
    segmap = imread(fname, to_gray=True)
    imh, imw = segmap.shape
    mask = None

    # save patches
    img_fname = osp.join('images_1x', name+'.jpg')
    img = imread(img_fname)

    if no_valid_mask == False:
        tile_valid_mask_f = osp.join(tpl_dir, f'valid_masks/{cam_id}.png')
        print('tile_valid_mask_f',tile_valid_mask_f)
        tpl_valid_mask = imread(tile_valid_mask_f, to_gray=True)
        # print('tpl_valid_mask',tpl_valid_mask)
        mask = np.zeros_like(segmap, dtype=bool)
        mask[tpl_valid_mask == 255] = True
    else:
        align_region = json.load(open(osp.join(tpl_dir, f'align_region.json')))
        print('no valid mask')
        x0, y0, x1, y1 = align_region[str(cam_id)]['align_bbox']
        mask = np.zeros_like(segmap, dtype=bool)
        mask[y0:y1, x0:x1] = True

    C_1x = deepcopy(C)
    for k in [
            'border_gap',
            'align_contour_margin',
            'connect_len',
            'coarse_far_dist_th',
            # 'coarse_far_ratio',
            'coarse_near_dist_th',
            'strict_dist_th',
            # 'strict_ratio',
            'refine_margin',
            'refine_dist_th',
            
            'cluster.cluster_dist',
            # 'cluster.max_edge_points',
    ]:
        update_zoomed_len(C.deviation, k, zoom)
    # print(C)

    defects, ctx = detect_deviations(name, segmap, tpl_segmap, tpl_distmap, C, mask=mask)
    fmt = '.png'
    d = cv2.imencode(fmt, defects['segmap'])[1].tobytes()
    defects['segmap'] = {
        'format': fmt,
        'data': d,
    }

    convex_nr = 0
    concave_nr = 0
    for k, obj in defects['objects'].items():
        if obj['type'] == 'convex':
            convex_nr += 1
        elif obj['type'] == 'concave':
            concave_nr += 1
    
    print (f'{name}: convex_nr {convex_nr}, concave_nr {concave_nr}')
    os.makedirs(defects_dir, exist_ok=True)
    pickle.dump(defects, open(osp.join(defects_dir, name+'.pkl'), 'wb'))

    if verify:
        def draw_box_fn(o):
            if 'type' not in o or o['type'] == 'concave':
                return {
                    'color': [255, 0, 0],
                    'thickness': 10,
                }
            elif o['type'] == 'convex':
                return {
                    'color': [255, 255, 0],
                    'thickness': 5,
                }

        os.makedirs(verify, exist_ok=True)
        canvas = np.zeros((imh, imw, 4), dtype='u1')
        # draw_defects(canvas, defects, box_fn=draw_box_fn, cfg=C.deviation)
        detect_type = 'deviations'
        draw_defects(detect_type, name, img, canvas, defects, box_fn=draw_box_fn, cfg=C.deviation) # save patches
        Image.fromarray(canvas).save(osp.join(verify, name+'.png'))



@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--debug', is_flag=True)
@click.option('--jobs', default=4)
@click.option('--lst', default='list.txt')
@click.option('--templates', default='')
@click.option('--verify', default='')
@click.option('--zoom', default=0)
@click.option('--no_valid_mask', is_flag=True)
@click.argument('segmaps')
@click.argument('defects')
def main(cfg, debug, jobs, lst, templates, verify, zoom, no_valid_mask, segmaps, defects):
    C = edict(toml.load(cfg))

    cls_info = {}
    if lst and osp.exists(lst):
        cls_info = read_list(lst, C.target.cam_mapping, return_dict=True)

    tsks = sorted(set(file_name(i) for i in os.listdir(segmaps)))
    tsks = [i for i in tsks if i in cls_info]

    worker = partial(
        process_one,
        segmaps=segmaps,
        defects_dir=defects,
        cls_info=cls_info,
        templates=templates,
        verify=verify,
        zoom=zoom,
        no_valid_mask=no_valid_mask,
        C=C,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
