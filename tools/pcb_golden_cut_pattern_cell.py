#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from glob import glob
import numpy as np
import json
from functools import partial
import toml
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from deeppcb.base import imsave, imread, rescale
from deeppcb.cell import cut_cells
from deeppcb.utils import run_parallel
import click
import pylab as plt
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def process_one(img_name, src, dst, scale, cells, C):
    pat = osp.splitext(img_name)[0]
    img = imread(osp.join(src, img_name), to_gray=True)
    ref_cell = cells.ref_cell
    x0, y0, x1, y1 = ref_cell[pat].box

    img = img[y0:y1, x0:x1]
    segmap = np.ones_like(img, dtype='u1') * C.classes.bg.label
    segmap[img & C.pattern.copper.label > 0] = C.classes.copper.label

    if scale != 1:
        segmap = rescale(segmap, scale, order='nearest')

    dst_f = osp.join(dst, img_name)
    imsave(dst_f, segmap)

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--scale', default=1.0)
@click.option('--jobs', default=4)
@click.option('--debug', is_flag=True)
@click.argument('cells')
@click.argument('src')
@click.argument('dst')
def main(cfg, scale,  jobs, debug, cells, src, dst):
    C = edict(toml.load(cfg))
    cells = edict(json.load(open(cells)))
    os.makedirs(dst, exist_ok=True)

    tsks = sorted(os.listdir(src))

    worker = partial(
        process_one,
        src=src,
        dst=dst,
        scale=scale,
        cells=cells,
        C=C,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
