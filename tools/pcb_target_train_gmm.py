#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import pickle
import numpy as np
import numpy.random as npr
import json
import cv2
import toml
from copy import deepcopy
from easydict import EasyDict as edict
from PIL import Image
from functools import partial

from tqdm import tqdm
from deeppcb.base import imsave, rescale, imread, resize
from deeppcb.utils import run_parallel, get_zoom
from deeppcb.target import read_list, process_train_gmm, verify_train_gmm, get_blank_mask

import pylab as plt
import click

def process_one(tsk, images, dst, templates, verify, C, zoom=None):
    img_name, board, face, cam_id = tsk
    name, ext = osp.splitext(img_name)
    dst_f = osp.join(dst, name+'.pkl')
    if osp.exists(dst_f):
        return

    tpl_dir = templates.format(board=board)
    img = imread(osp.join(images, img_name))

    tpl_segmap = imread(osp.join(tpl_dir, f'segmaps/{cam_id}.png'), to_gray=True)

    zoom = get_zoom(zoom, images)
    cfg = deepcopy(C.target.train_gmm)
    for k in ['blank_gap', 'segmap_shrink', 'edge_area_expand']:
        cfg.gmm_params[k] /= zoom

    out, info = process_train_gmm(img, tpl_segmap, img_blank_mask=get_blank_mask(img.mean(axis=2)), shadow_label=C.classes.bg_shadow, **cfg)

    out['zoom'] = zoom
    os.makedirs(dst, exist_ok=True)
    pickle.dump(out, open(dst_f, 'wb'))

    if verify:
        os.makedirs(verify, exist_ok=True)
        canvas = verify_train_gmm(**info)
        imsave(osp.join(verify, name+'_gmm_sample.jpg'), canvas)

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--templates', default='')
@click.option('--debug', is_flag=True)
@click.option('--jobs', default=4)
@click.option('--verify', default='')
@click.argument('lst')
@click.argument('images')
@click.argument('dst')
def main(cfg, templates, debug, jobs, verify, lst, images, dst):
    C = edict(toml.load(cfg))

    worker = partial(
        process_one,
        images=images,
        dst=dst,
        templates=templates,
        verify=verify,
        C=C,
    )

    tsks = read_list(lst, C.target.cam_mapping)
    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
