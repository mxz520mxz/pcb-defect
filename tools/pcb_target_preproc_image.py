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
from deeppcb.base import resize, imsave, get_gray, get_edge, imread, rescale, pil_rescale
from deeppcb.utils import run_parallel
from deeppcb.target import read_list, process_crop
import pylab as plt
import click

Image.MAX_IMAGE_PIXELS = None

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

# def process_one(tsk, images, dst, templates, C):
#     img_name, board, face, cam_id = tsk

#     dst_f = osp.join(dst, img_name)
#     if osp.exists(dst_f):
#         return

#     tpl_edge = Image.open(osp.join(templates.format(board=board), f'edges/{cam_id}.png'))
#     ys, xs = np.where(tpl_edge)
#     tpl_cx, tpl_cy = xs.mean(), ys.mean()
#     w, h = tpl_edge.size

#     img = Image.open(osp.join(images, img_name))
#     img = process_crop(img, (tpl_cx, tpl_cy), (w, h), scale=C.target.crop.edge_scale)

#     os.makedirs(dst, exist_ok=True)
#     imsave(dst_f, img)

def process_one(tsk, images, dst, templates, C):
    img_name, board, face, cam_id = tsk

    dst_f = osp.join(dst, img_name)
    if osp.exists(dst_f):
        return

    tpl_edge_path = osp.join(templates.format(board=board), f'edges/{cam_id}.png')
    if osp.exists(tpl_edge_path):
        tpl_edge = Image.open(tpl_edge_path)
        ys, xs = np.where(tpl_edge)
        tpl_cx, tpl_cy = xs.mean(), ys.mean()
        w, h = tpl_edge.size

        img = Image.open(osp.join(images, img_name))
        img = process_crop(img, (tpl_cx, tpl_cy), (w, h), scale=C.target.crop.edge_scale)

        os.makedirs(dst, exist_ok=True)
        imsave(dst_f, img)
    else:
        pass

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--templates', default='')
@click.option('--jobs', default=4)
@click.option('--debug', is_flag=True)
@click.argument('lst')
@click.argument('images')
@click.argument('dst')
def main(cfg, templates, jobs, debug, lst, images, dst):
    C = edict(toml.load(cfg))

    tsks = read_list(lst, C.target.cam_mapping)

    worker = partial(
        process_one,
        images=images,
        dst=dst,
        templates=templates,
        C=C,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
