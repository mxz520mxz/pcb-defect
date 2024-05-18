#!/usr/bin/env python
import os
import os.path as osp

base_dir = osp.join(osp.dirname(__file__), '..')

from easydict import EasyDict as edict
import numpy as np
import toml
from glob import glob
from PIL import Image
import pylab as plt
import cv2
import click

Image.MAX_IMAGE_PIXELS = None

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--debug', is_flag=True)
@click.argument('src')
@click.argument('dst')
def main(cfg, debug, src, dst):
    C = edict(toml.load(cfg))
    img_files = list(set((osp.basename(i), Image.open(i).size) for i in glob(osp.join(src, '*', '*.png'))))

    img = np.array(Image.open(src))

    imh, imw = img.shape

    canvases = {}

    tot = np.zeros((imh, imw, 3), dtype='u1')
    for k, v in C.pattern.items():
        canvas = np.zeros((imh, imw, 3), dtype='u1')
        mask = np.bitwise_and(img, v.label) > 0
        if not mask.sum():
            continue

        if k == 'border_points':
            mask = cv2.dilate(mask.astype('u1'), np.ones((32, 32)))
            mask = mask > 0

        canvas[mask] = v.color
        canvases[k] = canvas
        tot[mask] = v.color

    if debug:
        plt.figure('total')
        plt.imshow(tot)
        for k, v in canvases.items():
            plt.figure(k)
            plt.imshow(v)
        plt.show()

    Image.fromarray(tot).save(dst)

if __name__ == "__main__":
    main()
