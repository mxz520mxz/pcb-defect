#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from functools import partial
from PIL import Image
from glob import glob
from multiprocessing import Pool
from deeppcb.base import resize, imsave
import click

Image.MAX_IMAGE_PIXELS = None

def process_one(img_f, src, dst, s, order, scale_value):
    dst_f = osp.join(dst, osp.relpath(img_f, src))
    if osp.exists(dst_f):
        return

    img = Image.open(img_f)
    imw, imh = img.size
    # print(s)
    # raise
    ratio = 1
    if not s[1]:
        if s[0] <= 1:
            ratio = s[0]
            w = ratio * imw
        else:
            w = s[0]
            ratio = w / imw

        h = ratio * imh

    elif not s[0]:
        if s[1] <= 1:
            ratio = s[1]
            h = ratio * imh
        else:
            h = s[1]
            ratio = h / imh

        w = ratio * imw
    else:
        w, h = s
        if w <= 1:
            w *= imw
        if h <= 1:
            h *= imh

    w = int(w)
    h = int(h)
    img = resize(img, (w, h), order='linear')
    if scale_value and ratio != 1:
        img = (img.astype('f4') * ratio).astype('u1')

    dst_f = osp.join(dst, osp.relpath(img_f, src))
    os.makedirs(osp.dirname(dst_f), exist_ok=True)
    imsave(dst_f, img)

@click.command()
@click.option('--size', '-s', required=True)
@click.option('--order', type=click.Choice(['linear', 'nearest']), default='linear')
@click.option('--ext', default='')
@click.option('--jobs', default=5)
@click.option('--debug', is_flag=True)
@click.option('--scale_value', is_flag=True)
@click.argument('src')
@click.argument('dst')
def main(size, order, ext, jobs, debug, scale_value, src, dst):
    s = list(float(i) if i else None for i in size.strip().split('x'))
    assert len(s) == 2

    tsks = glob(osp.join(src, '**', '*'+ext), recursive=True)
    assert tsks
    worker = partial(process_one, src=src, dst=dst, s=s, order=order, scale_value=scale_value)
    if debug:
        for tsk in tsks:
            worker(tsk)
    else:
        with Pool(min(len(tsks), jobs)) as p:
            p.map(worker, tsks)

if __name__ == "__main__":
    main()
