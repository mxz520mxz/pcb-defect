#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from easydict import EasyDict as edict
from time import time
from glob import glob
from PIL import Image
import numpy as np
from skimage.util import img_as_float
from skimage.transform import ProjectiveTransform, warp
from deeppcb.align_edge import align_edge, get_edge_img
import pylab as plt
import click

Image.MAX_IMAGE_PIXELS = None

def rescale(img, ratio):
    w = int(img.width * ratio)
    h = int(img.height * ratio)
    return img.resize((w, h))

@click.command()
@click.option('--box', default='0,0,1,1')
@click.option('--zoom', default=1)
@click.option('--apply_full', is_flag=True)
@click.argument('src')
@click.argument('dst')
def main(box, zoom, apply_full, src, dst):
    x0, y0, x1, y1 = [float(i) for i in box.split(',')]

    src_img = Image.open(src)
    if zoom:
        src_img = rescale(src_img, 1 / zoom)
    src_img = np.asarray(src_img)

    fixed = edict({
        'img': src_img,
    })

    dst_img = Image.open(dst)
    if zoom:
        dst_img = rescale(dst_img, 1 / zoom)

    dst_img = np.asarray(dst_img)
    dst_h, dst_w = dst_img.shape[:2]

    if x1 > 1 and y1 > 1:
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    else:
        x0, y0, x1, y1 = int(dst_w * x0), int(dst_h * y0), int(dst_w * x1), int(dst_h * y1)

    print ("box", x0, y0, x1, y1)

    moving = edict({
        'img': dst_img[y0:y1,x0:x1]
    })

    fixed = get_edge_img(fixed, with_distmap=True)
    moving = get_edge_img(moving, with_points=True)

    ret = align_edge(moving, fixed, init_bbox=[
        x0, y0, x1, y1,
    ], tform_tp='projective', max_iters=1000, max_patience=50, lr=1e-3)

    H = ret['H']

    tform = ProjectiveTransform(H)

    warp_dst_img = warp(dst_img, tform.inverse, output_shape=src_img.shape)

    canvas = (img_as_float(warp_dst_img) + img_as_float(src_img))/2
    if not apply_full:
        canvas = canvas[y0:y1, x0:x1]

    plt.imshow(canvas)
    plt.show()

if __name__ == "__main__":
    main()
