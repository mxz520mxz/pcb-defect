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
from tqdm import tqdm
from multiprocessing import Pool
import pickle
from deeppcb.base import imsave, file_name, Image
from deeppcb.cell import apply_cell_cuttings
from deeppcb.utils import run_parallel
import click
import pylab as plt
from tqdm import tqdm

def process_one(img_name, tform_dir, src, dst, order, border_value):
    name, ext = osp.splitext(img_name)
    img = Image.open(osp.join(src, img_name))

    tform_fs = sorted(glob(osp.join(tform_dir, name+'-*')))
    tforms = [pickle.load(open(i, 'rb'))for i in tform_fs]
    imgs = apply_cell_cuttings(img, tforms, order=order, border_value=border_value)
    for tform_f, img in zip(tform_fs, imgs):
        dst_name = file_name(tform_f)
        dst_f = osp.join(dst, dst_name+ext)
        os.makedirs(dst, exist_ok=True)
        imsave(dst_f, img)

@click.command()
@click.option('--order', default='nearest')
@click.option('--border_value', default=0)
@click.option('--jobs', default=4)
@click.option('--debug', is_flag=True)
@click.argument('tform_dir')
@click.argument('src')
@click.argument('dst')
def main(order, border_value, jobs, debug, tform_dir, src, dst):
    tsks = sorted(os.listdir(src))

    worker = partial(
        process_one,
        tform_dir=tform_dir,
        src=src,
        dst=dst,
        order=order,
        border_value=border_value,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
