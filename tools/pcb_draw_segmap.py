#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from easydict import EasyDict as edict
import numpy as np
import toml
import cv2
from glob import glob
from PIL import Image
import pylab as plt
from functools import partial
from deeppcb.base import file_name
from deeppcb.utils import run_parallel

import click

Image.MAX_IMAGE_PIXELS = None

def process_one(f, debug, out, C):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    imh, imw = img.shape
    canvas = np.zeros((imh, imw, 3), dtype='u1')

    for cls_name, v in C.classes.items():
        canvas[img & v.label > 0] = v.color

    if debug:
        plt.imshow(canvas)
        plt.show()

    if out:
        if not out.endswith('.jpg'):
            os.makedirs(out, exist_ok=True)
            out_f = osp.join(out, file_name(f)+'.jpg')
        else:
            out_f = out
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_f, canvas)

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--debug', is_flag=True)
@click.option('--jobs', default=4)
@click.option('-o', '--out', default='')
@click.argument('src')
def main(cfg, debug, jobs, out, src):
    C = edict(toml.load(cfg))

    if osp.isdir(src):
        tsks = sorted(glob(osp.join(src, '*.png')))
    else:
        assert osp.exists(src)
        tsks = [src]

    worker = partial(process_one, debug=debug, out=out, C=C)

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
