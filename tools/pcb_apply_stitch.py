#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from PIL import Image
import numpy as np
from deeppcb.base import imsave
from deeppcb.stitch import stitch_images
import click
import json

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

@click.command()
@click.option('--tform', default='stitched/Ts.json')
@click.option('--order', type=click.Choice(['linear', 'nearest']), default='linear')
@click.option('--border_value', default=0)
@click.option('--fusion', type=click.Choice(['avg', 'overlap']), default='avg')
@click.argument('src')
@click.argument('dst')
def main(tform, order, border_value, fusion, src, dst):
    Ts = json.load(open(tform))
    img_f = osp.join(src, os.listdir(src)[0])
    img = Image.open(img_f)
    chn = 3
    if img.mode == 'L':
        chn = 1

    ext = osp.splitext(img_f)[-1]

    imgs = {file_name(i): Image.open(osp.join(src, i)) for i in os.listdir(src)}

    o_w, o_h = Ts['golden_size']

    Ts = Ts['golden_Ts_wc']

    ret = stitch_images(imgs, Ts, (o_w, o_h, chn), order=order, border_value=border_value, fusion=fusion)

    imsave(dst, ret['img'])

if __name__ == "__main__":
    main()
