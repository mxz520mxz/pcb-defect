#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import cv2
import numpy as np
import toml
from easydict import EasyDict as edict
from functools import partial
from PIL import Image
from glob import glob
import json
from time import time
from multiprocessing import Pool
from deeppcb.base import resize, imsave, get_gray, get_edge, imread, rescale, Image
from deeppcb.target import read_list, process_filter
from deeppcb.utils import run_parallel, get_zoomed_len
import pylab as plt
import click

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def process_one(img_name, src, dst, cfg, zoom):
    name = osp.splitext(img_name)[0]

    dst_f = osp.join(dst, img_name)
    if osp.exists(dst_f):
        return

    img = Image.open(osp.join(src, img_name))
    img = process_filter(img, cfg.d, cfg.sigma_color, cfg.sigma_space)

    os.makedirs(dst, exist_ok=True)
    imsave(dst_f, img)

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--zoom', default=1)
@click.option('--jobs', default=4)
@click.option('--debug', is_flag=True)
@click.argument('src')
@click.argument('dst')
def main(cfg, zoom, jobs, debug, src, dst):
    cfg = edict(toml.load(cfg)).target.filter
    cfg.d = get_zoomed_len(cfg.d, zoom)
    cfg.sigma_space = get_zoomed_len(cfg.sigma_space, zoom)
    
    tsks = sorted(os.listdir(src))

    worker = partial(
        process_one,
        src=src,
        dst=dst,
        cfg=cfg,
        zoom=zoom,
    )
    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
