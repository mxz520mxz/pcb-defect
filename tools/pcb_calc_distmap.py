#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from glob import glob
import toml
from easydict import EasyDict as edict
import cv2
from deeppcb.base import imread, file_name, Image
from deeppcb.align_edge import get_edge_img
from deeppcb.utils import run_parallel
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import click
import pylab as plt

def process_one(img_name, src, dst_distmap, dst_edge, is_segmap, max_dist, C):
    img_f = osp.join(src, img_name)
    if is_segmap:
        img = imread(img_f, to_gray=True)
        img = ((img & C.classes.copper.label > 0)*255).astype('u1')
        # TODO there shoud be more simple way for binary image than canny
    else:
        img = imread(img_f)

    imdic = get_edge_img(
        {'img': img},
        with_edge=bool(dst_edge),
        with_distmap=bool(dst_distmap),
        distmap_kws={'max_dist': max_dist}
    )

    name = file_name(img_name)
    if dst_distmap:
        if not dst_distmap.endswith('.jpg'):
            os.makedirs(dst_distmap, exist_ok=True)
            dst_f = osp.join(dst_distmap, name+'.jpg')
        else:
            dst_f = dst_distmap

        m = imdic['distmap'].astype('u1')
        cv2.imwrite(dst_f, m)

    if dst_edge:
        if not dst_edge.endswith('.png'):
            os.makedirs(dst_edge, exist_ok=True)
            dst_f = osp.join(dst_edge, name+'.png')
        else:
            dst_f = dst_edge

        m = imdic['edge'].astype('u1')
        cv2.imwrite(dst_f, m)

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--max_dist', default=100)
@click.option('--is_segmap', is_flag=True)
@click.option('--debug', is_flag=True)
@click.option('--jobs', default=1)
@click.argument('src')
@click.argument('dst_distmap')
@click.argument('dst_edge')
def main(cfg, max_dist, debug, jobs, is_segmap, src, dst_distmap, dst_edge):
    C = edict(toml.load(open(cfg)))

    assert max_dist < 255
    if osp.isdir(src):
        tsks = sorted(os.listdir(src))
    elif osp.isfile(src):
        tsks = [osp.basename(src)]
        src = osp.dirname(src)
    else:
        raise

    worker = partial(process_one, src=src, dst_distmap=dst_distmap, dst_edge=dst_edge, is_segmap=is_segmap, max_dist=max_dist, C=C)

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
