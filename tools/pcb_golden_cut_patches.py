#!/usr/bin/env python
import os
import os.path as osp

base_dir = osp.join(osp.dirname(__file__), '..')

import toml
from easydict import EasyDict as edict
from collections import defaultdict
import cv2
import numpy as np
import json
from tqdm import tqdm
from glob import glob
from PIL import Image
import pickle
from pprint import pprint
import pylab as plt
import click
from deeppcb.base import imsave

Image.MAX_IMAGE_PIXELS = None

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--patches', default='patches/patches.pkl')
@click.option('--debug', is_flag=True)
@click.argument('src')
@click.argument('dst')
def main(cfg, patches, debug, src, dst):
    C = edict(toml.load(cfg))

    os.makedirs(dst, exist_ok=True)
    img_names = sorted(os.listdir(src))

    patch_cfg = pickle.load(open(patches, 'rb'))['partial_cells']
    for img_name in tqdm(img_names):
        img = Image.open(osp.join(src, img_name))
        name, ext = osp.splitext(img_name)
        patch_items = patch_cfg[name]['patches']
        for item in patch_items:
            dst_name = item['name']
            x0, y0, x1, y1 = item['bbox']
            pimg = img.crop((x0, y0, x1, y1))
            imsave(osp.join(dst, dst_name+ext),pimg)

if __name__ == "__main__":
    main()
