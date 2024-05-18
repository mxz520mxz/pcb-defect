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
from PIL import Image
from tqdm import tqdm
import pickle
from multiprocessing import Pool
from deeppcb.base import imsave, parse_name, imread
from deeppcb.cell import cut_cells
from deeppcb.target import read_list
from deeppcb.utils import run_parallel
import cv2
import click
import pylab as plt
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def process_one(img_name, images, src, dst, templates):
    patch_cfgs = {}
    name, ext = osp.splitext(img_name)
    info = edict(parse_name(name))
    tpl_dir = osp.join(templates, info.board)

    patch_cfg = edict(pickle.load(open(osp.join(tpl_dir, 'patches', 'patches.pkl'), 'rb')))
    cell_w, cell_h = patch_cfg['cell_size']

    img = imread(osp.join(src, img_name))
    if src == images:
        ref_img = img
    else:
        ref_img = imread(osp.join(images, name+'.jpg'))

    ignore_mask = ref_img.max(2) > 230
    ignore_mask = cv2.dilate(ignore_mask.astype('u1'), np.ones((5, 5))) > 0

    img[ignore_mask, :] = 0

    for (r, c), (x0, y0, x1, y1) in patch_cfg['patch_bboxes']:
        cur_ignore_mask = ignore_mask[y0:y1, x0:x1]
        if cur_ignore_mask.sum() / cur_ignore_mask.size > 0.05:
            continue
        if cur_ignore_mask.shape[0] < y1 - y0 or cur_ignore_mask.shape[1] < x1 - x0:
            continue

        pimg = img[y0:y1, x0:x1]
        dst_name = f'{name}-ppos:{r}_{c}'
        dst_f = osp.join(dst, dst_name+ext)
        imsave(dst_f, pimg)

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--templates', default='templates')
@click.option('--jobs', default=4)
@click.option('--debug', is_flag=True)
@click.argument('images')
@click.argument('src')
@click.argument('dst')
def main(cfg, templates, jobs, debug, images, src, dst):
    C = edict(toml.load(open(cfg)))
    tsks = os.listdir(src)

    os.makedirs(dst, exist_ok=True)

    worker = partial(
        process_one,
        images=images,
        src=src,
        dst=dst,
        templates=templates,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
