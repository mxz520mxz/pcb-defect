#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import cv2
import pickle
import numpy as np
import numpy.random as npr
import toml
from easydict import EasyDict as edict
from functools import partial

from deeppcb.base import imsave, imread
from deeppcb.target import process_gmm_seg, get_blank_mask, get_gmm_img, get_feature_img
from deeppcb.utils import run_parallel

import pylab as plt
import click

def file_name(f):
    return osp.basename(osp.splitext(f)[0])

def process_one(name, src, dst, C):
    dst_f = osp.join(dst, name+'.png')
    if osp.exists(dst_f):
        return

    cfg = C.target.gmm_seg

    img_name = name+'.jpg'
    print('--------------', img_name)
    img = imread(osp.join(src, img_name))
    imdic = get_gmm_img(img, cfg.feature)
    img = get_feature_img(imdic, cfg.feature)

    segmap = process_gmm_seg(
        img,
        C.classes,
        blank_mask=get_blank_mask(img[...,0]),
        sample_nr=cfg.sample_nr,
        random_seed=cfg.random_seed,
    )

    bw = cfg.blank_border_width
    if bw:
        segmap[:bw, :] = 0
        segmap[-bw:, :] = 0
        segmap[:, :bw] = 0
        segmap[:, -bw:] = 0

    os.makedirs(osp.dirname(dst_f), exist_ok=True)
    imsave(dst_f, segmap)

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--debug', is_flag=True)
@click.option('--jobs', default=4)
@click.argument('src')
@click.argument('dst')
def main(cfg, debug, jobs, src, dst):
    C = edict(toml.load(cfg))

    tsks = sorted(set(file_name(i) for i in os.listdir(src)))

    worker = partial(
        process_one,
        src=src,
        dst=dst,
        C=C,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main() 
