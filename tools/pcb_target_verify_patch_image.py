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
from deeppcb.base import imsave, imread, parse_name
import click
from tqdm import tqdm
import pylab as plt

def calc_diff(x, y):
    return (x.astype('i4') - y.astype('i4') + 128).astype('u1')

def process_one(img_name, src1, src2, dst, templates):
    name, ext = osp.splitext(img_name)
    img1 = imread(osp.join(src1, img_name))
    img2 = imread(osp.join(src2, img_name))

    info = parse_name(name)
    match_name = name[img_name.index('-')+1:]

    match_img = imread(osp.join(templates, info['board'], 'patches', 'images', match_name+'.jpg'))

    dst_f = osp.join(dst, name + ext)
    canvas = np.vstack([
        np.hstack([img1, match_img, calc_diff(img1, match_img)]),
        np.hstack([img2, match_img, calc_diff(img2, match_img)]),
    ])
    imsave(dst_f, canvas)

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--jobs', default=4)
@click.option('--debug', is_flag=True)
@click.option('--templates', default='templates')
@click.argument('src1')
@click.argument('src2')
@click.argument('dst')
def main(cfg, jobs, debug, templates, src1, src2, dst):
    C = edict(toml.load(open(cfg)))

    tsks = sorted(os.listdir(src1))
    os.makedirs(dst, exist_ok=True)

    worker = partial(
        process_one,
        src1=src1,
        src2=src2,
        dst=dst,
        templates=templates,
    )
    if debug:
        for tsk in tsks:
            worker(tsk)
    else:
        with Pool(min(len(tsks), jobs)) as p:
            p.map(worker, tsks)

if __name__ == "__main__":
    main()
