#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from glob import glob
from time import time
import numpy as np
import toml
import json
import cv2
from functools import partial
from easydict import EasyDict as edict
from deeppcb.base import imsave, imread
from deeppcb.cell import extract_cell_contours
from deeppcb.utils import run_parallel
from skimage import measure
from skimage.morphology import disk, binary_dilation
import pickle
import click
import pylab as plt

def process_one(img_name, src, dst, C, verify):
    name, ext = osp.splitext(img_name)
    img = imread(osp.join(src, img_name), to_gray=True)
    imh, imw = img.shape[:2]

    # cfg = C.target.contour
    cfg = C.target

    blank_mask = img == 0
    copper = img & C.classes.copper.label > 0
    copper[blank_mask] = False
    # blank_mask = binary_dilation(blank_mask, disk(cfg.valid_margin))
    blank_mask = binary_dilation(blank_mask, disk(cfg.align_cell.valid_margin))

    conts = extract_cell_contours(copper, blank_mask, min_len=cfg.align_cell.min_len)

    if verify:
        canvas = np.zeros((imh, imw, 3), dtype='u1')
        canvas[copper] = 255
        for c in conts:
            c = c['contour']
            canvas[c[:,1], c[:,0]] = [255, 0, 0]

        imsave(osp.join(verify, img_name), canvas)

    pickle.dump({
        'contours': conts,
    }, open(osp.join(dst, name+'.pkl'), 'wb'))

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--jobs', default=4)
@click.option('--debug', is_flag=True)
@click.option('--verify', default='')
@click.argument('src')
@click.argument('dst')
def main(cfg, jobs, debug, verify, src, dst):
    C = edict(toml.load(open(cfg)))

    if osp.isfile(src):
        tsks = osp.basename(src)
        src = osp.dirname(src)
    else:
        tsks = sorted(os.listdir(src))

    os.makedirs(dst, exist_ok=True)
    if verify:
        os.makedirs(verify, exist_ok=True)

    worker = partial(
        process_one,
        src=src,
        dst=dst,
        C=C,
        verify=verify,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
