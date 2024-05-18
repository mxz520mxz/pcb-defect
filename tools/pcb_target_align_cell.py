#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from glob import glob
import numpy as np
import toml
import json
from functools import partial
from easydict import EasyDict as edict
from skimage.exposure import match_histograms
from skimage.morphology import binary_dilation, disk
import pickle
from deeppcb.base import imsave, imread, get_gray, transform_img, parse_name, file_name
from deeppcb.utils import run_parallel
from deeppcb.cell import align_cell, extract_cell_contours
from deeppcb.align_edge import align_edge
import click
import pylab as plt
import numpy.linalg as npl

def process_one(img_name, tform_dir, src, dst, templates, verify, C):
    name, ext = osp.splitext(img_name)
    img_f = osp.join(src, img_name)

    img = imread(img_f, to_gray=True)
    imh, imw = img.shape[:2]

    cfg = C.target.align_cell

    blank_mask = img == 0
    fg_mask = img & C.classes.copper.label > 0
    fg_mask[blank_mask] = False
    blank_mask = binary_dilation(blank_mask, disk(cfg.valid_margin))

    info = edict(parse_name(name))
    tpl_distmap = imread(osp.join(templates, info.board, 'golden_cell/distmaps', info.pat+'.png'), to_gray=True)

    conts = extract_cell_contours(fg_mask, blank_mask, min_len=cfg.min_len)
    ret = align_cell(conts, tpl_distmap, **cfg)
    assert ret['tgt_shape'] == ret['tpl_shape']
    print('ret',ret)
    # raise
    # T_12 = ret['H']
    T_12 = npl.inv(ret['H_21']) #need to inv?

    tform = pickle.load(open(osp.join(tform_dir, name+'.pkl'), 'rb'))
    print('tform',tform)
    # raise
    tform['T_12'] = T_12 
    # T_02 = np.asarray(tform['T_01']).dot(T_12)
    T_02 = np.asarray(npl.inv(tform['T_10'])).dot(T_12) #need to inv to 01?
    tform['T_02'] = T_02
    tform['T'] = tform['T_02'] #?

    dst_f = osp.join(dst, name+'.pkl')
    os.makedirs(osp.dirname(dst_f), exist_ok=True)
    pickle.dump(tform, open(dst_f, 'wb'))

    if verify:
        os.makedirs(verify, exist_ok=True)

        fg_mask = np.dstack([fg_mask]*3) * [255, 127, 0]
        # warp_fg_mask = transform_img(fg_mask, T_12, (imw, imh), order='nearest')
        warp_fg_mask = transform_img(fg_mask, npl.inv(T_12), (imw, imh), order='nearest')

        tpl_img = imread(osp.join(templates, info['board'], 'golden_cell/segmaps', info['pat']+'.png'), to_gray=True)
        tpl_mask = tpl_img & C.classes.copper.label > 0
        tpl_mask = np.dstack([tpl_mask]*3) * [0, 127, 255]

        dst_f = osp.join(verify, f'{name}_init.png')
        imsave(dst_f, (fg_mask + tpl_mask).astype('u1'))

        dst_f = osp.join(verify, f'{name}_warp.png')
        imsave(dst_f, (warp_fg_mask + tpl_mask).astype('u1'))

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--jobs', default=2)
@click.option('--debug', is_flag=True)
@click.option('--templates', default='templates')
@click.option('--verify', default='')
@click.argument('tform_dir')
@click.argument('src')
@click.argument('dst')
def main(cfg, jobs, debug, templates, verify, tform_dir, src, dst):
    C = edict(toml.load(open(cfg)))

    tsks = sorted(os.listdir(osp.join(src)))

    worker = partial(
        process_one,
        tform_dir=tform_dir,
        src=src,
        dst=dst,
        templates=templates,
        verify=verify,
        C=C,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
