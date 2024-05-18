#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import pickle
from copy import deepcopy
import numpy as np
import numpy.random as npr
import toml
from easydict import EasyDict as edict
import cv2
import skimage as sk
from skimage.draw import polygon
from functools import partial
import json

from deeppcb.base import imsave, imread, resize, Image
from deeppcb.target import read_list
from deeppcb.utils import run_parallel, get_zoom, update_zoomed_len, update_zoomed_area
from deeppcb.foreign import detect_foreigns, draw_defects

import pylab as plt
import click

def file_name(f):
    return osp.basename(osp.splitext(f)[0])

# def process_one(name, images, segmaps, defects_dir, verify, zoom, C):
def process_one(name, images, segmaps, defects_dir, cls_info, templates, verify, no_valid_mask, zoom, C):
    print('-------------', name)
    zoom = get_zoom(zoom, images)

    ci = cls_info[name]
    cam_id = ci['cam_id']
    tpl_dir = templates.format(board=ci['board'])

    img = imread(osp.join(images, name+'.jpg'))
    imh, imw = img.shape[:2]

    segmap = imread(osp.join(segmaps, name+'.png'), to_gray=True)

    mask = None
    if no_valid_mask == False:
        tile_valid_mask_f = osp.join(tpl_dir, f'valid_masks/{cam_id}.png')
        print('tile_valid_mask_f',tile_valid_mask_f)
        tpl_valid_mask = imread(tile_valid_mask_f, to_gray=True)
        # print('tpl_valid_mask',tpl_valid_mask)
        mask = np.zeros_like(segmap, dtype=bool)
        mask[tpl_valid_mask == 255] = True
    else:
        print('no valid mask')
        mask = True

    cfg = deepcopy(C.foreign)
    for k in [
            'crop_border',
            'copper_margin',
            'bg_margin',
            'cluster.cluster_dist',
            # 'cluster.max_edge_points',

            'inland_sea.surr_radius',
            'inland_sea.floodfill_tol',

            'insea_land.surr_radius',
            'insea_land.floodfill_tol',

            'deep_water.surr_radius',
            'deep_water.min_intensity_var',
            'deep_water.min_rb_var',

            'high_sand.surr_radius',
            'high_sand.min_intensity_var',
            'high_sand.min_rb_var',

            'shallow_water.surr_radius',
            'shallow_water.floodfill_tol',
            'shallow_water.max_intensity_range',

            'shallow_sand.surr_radius',
            'shallow_sand.floodfill_tol',
            'shallow_sand.max_intensity_range',

    ]:
        update_zoomed_len(cfg, k, zoom)

    for k in [
            'inland_sea.max_area',
            'inland_sea.min_area',

            'insea_land.max_area',
            'insea_land.min_area',

            'small_pond.max_area',

            'small_reef.max_area',
            
    ]:
        update_zoomed_area(cfg, k, zoom)
    # print('zoom', zoom)
    # print('update config', cfg)
    # raise
    # defects, ctx = detect_foreigns(segmap, img, C.classes, cfg)
    defects, ctx = detect_foreigns(segmap, img, C.classes, cfg, mask=mask)

    segmap = defects['segmap']
    fmt = '.png'
    d = cv2.imencode(fmt, defects['segmap'])[1].tobytes()
    defects['segmap'] = {
        'format': fmt,
        'data': d,
    }

    black_nr = 0
    gray_nr = 0
    light_nr = 0
    # print(defects['objects'])
    for k, obj in defects['objects'].items():
        if obj['level'] == 'black':
            black_nr += 1
        elif obj['level'] == 'gray':
            gray_nr += 1
        if obj['type'] == 'shallow_water':
            light_nr += 1

    print (f'{name}: black_nr {black_nr}, gray_nr {gray_nr}, light_nr {light_nr}, group_nr {len(defects["groups"])}')

    os.makedirs(defects_dir, exist_ok=True)
    pickle.dump(defects, open(osp.join(defects_dir, name+'.pkl'), 'wb'))

    def draw_box_fn(o):
        if o['level'] == 'black':
            return {
                'color': [255, 0, 0],
                'thickness': 10,
            }
        elif o['level'] == 'gray':
            return {
                'color': [255, 255, 0],
                'thickness': 5,
            }
        elif o['area'] > 10:
            return {
                'color': [0, 255, 0],
                'thickness': 5,
            }

    if verify:
        os.makedirs(verify, exist_ok=True)
        canvas = np.zeros((imh, imw, 4), dtype='u1')
        # draw_defects(canvas, defects, box_fn=draw_box_fn, cfg=C.foreign)
        detect_type = 'foreigns'
        draw_defects(detect_type, name, img, canvas, defects, box_fn=draw_box_fn, cfg=C.foreign) # save patches
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
@click.argument('images')
@click.argument('segmaps')
@click.argument('defects')
def main(cfg, debug, jobs, lst, templates, verify, zoom, no_valid_mask, images, segmaps, defects):
    C = edict(toml.load(cfg))

    cls_info = {}
    if lst and osp.exists(lst):
        cls_info = read_list(lst, C.target.cam_mapping, return_dict=True)

    tsks = sorted(set(file_name(i) for i in os.listdir(segmaps)))
    tsks = [i for i in tsks if i in cls_info]

    worker = partial(
        process_one,
        images=images,
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

