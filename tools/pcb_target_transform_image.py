#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import json
from glob import glob
from easydict import EasyDict as edict
import toml
from functools import partial
from multiprocessing import Pool
from deeppcb.base import imread, transform_img, imsave, scale_H
from deeppcb.target import file_name
from deeppcb.utils import run_parallel
import cv2
import numpy as np
import numpy.linalg as npl
import click

def process_one(align_f, src, dst):
    name = file_name(align_f)
    align = edict(json.load(open(align_f)))
    src_img = imread(osp.join(src, name+'.jpg'))

    s = src_img.shape[1] / align.tgt_shape[1]
    H = scale_H(align.H_21, s)

    tpl_h, tpl_w = align.tpl_shape
    tpl_h = int(tpl_h * s)
    tpl_w = int(tpl_w * s)

    dst_img = transform_img(src_img, H, (tpl_w, tpl_h))
    dst_f = osp.join(dst, name+'.jpg')
    os.makedirs(dst, exist_ok=True)
    imsave(dst_f, dst_img)

@click.command()
@click.option('--debug', is_flag=True)
@click.option('--jobs', default=4)
@click.argument('transforms')
@click.argument('src')
@click.argument('dst')
def main(transforms, debug, jobs, src, dst):
    tsks = sorted(glob(osp.join(transforms, '*.json')))

    worker = partial(
        process_one,
        src=src,
        dst=dst,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
