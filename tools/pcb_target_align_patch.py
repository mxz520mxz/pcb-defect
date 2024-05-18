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
from multiprocessing import Pool
from skimage.exposure import match_histograms
from deeppcb.base import imsave, imread, get_gray, transform_img, parse_name
from deeppcb.align_edge import align_edge
from deeppcb.align_ocv import ocv_find_ecc
import click
from tqdm import tqdm
import pylab as plt

def process_one(img_name, src, dst, templates, C):
    name, ext = osp.splitext(img_name)
    img = imread(osp.join(src, img_name))

    info = parse_name(name)
    tpl_name = name[img_name.index('-')+1:]
    patch_dir = osp.join(templates, info['board'], 'patches')
    tpl_distmap = imread(osp.join(patch_dir, 'distmaps', tpl_name+'.jpg'), to_gray=True)

    h, w = img.shape[:2]

    ag = C.align_gap
    x0, y0, x1, y1 = ag, ag, w-ag, h-ag
    init_bbox = [x0, y0, x1, y1]
    moving_img = get_gray(img[y0:y1, x0:x1])

    ret = align_edge(
        moving_img,
        {
            'distmap': tpl_distmap,
        },
        init_bbox=init_bbox,
        tform_tp=C.tform,
        sim_no_scale=C.sim_no_scale,
        lr=C.lr,
        max_patience=C.max_patience,
        max_iters=C.max_iters,
        msg_prefix=f'{img_name.split("-")[0]}: ',
        verbose=C.debug,
    )

    H = ret['H']

    warp_img = transform_img(img, H, (tpl_distmap.shape[1], tpl_distmap.shape[0]))

    dst_f = osp.join(dst, name + ext)
    imsave(dst_f, warp_img)

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--jobs', default=4)
@click.option('--debug', is_flag=True)
@click.option('--align_gap', default=8)
@click.option('--tform', default='similarity')
@click.option('--sim_no_scale', is_flag=True)
@click.option('--lr', default=3e-3)
@click.option('--max_patience', default=10)
@click.option('--max_iters', default=100)
@click.option('--templates', default='templates')
@click.argument('src')
@click.argument('dst')
def main(cfg, jobs, debug, align_gap, tform, sim_no_scale, lr, max_patience, max_iters, templates, src, dst):
    C = edict(toml.load(open(cfg)))

    tsks = sorted(os.listdir(src))
    os.makedirs(dst, exist_ok=True)

    C.update({
        'tform': tform,
        'sim_no_scale': sim_no_scale,
        'lr': lr,
        'max_patience': max_patience,
        'max_iters': max_iters,
        'align_gap': align_gap,
        'debug': debug,
    })

    worker = partial(
        process_one,
        src=src,
        dst=dst,
        templates=templates,
        C=C,
    )
    if debug:
        for tsk in tsks:
            worker(tsk)
    else:
        with Pool(min(len(tsks), jobs)) as p:
            p.map(worker, tsks)

if __name__ == "__main__":
    main()
