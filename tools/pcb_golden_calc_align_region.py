#!/usr/bin/env python
import os
import os.path as osp

base_dir = osp.join(osp.dirname(__file__), '..')

import toml
from easydict import EasyDict as edict
import numpy as np
import json
from glob import glob
from tqdm import tqdm
import cv2
import click

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--dilate', default=16)
@click.argument('src')
@click.argument('dst')
def main(cfg, dilate, src, dst):
    os.makedirs(osp.dirname(dst), exist_ok=True)
    C = edict(toml.load(cfg))

    kernel = np.ones((dilate, dilate), np.uint8)

    out = {}
    for seg_f in tqdm(glob(osp.join(src, '*.png'))):
        name = osp.splitext(osp.basename(seg_f))[0]

        segmap = cv2.imread(seg_f, cv2.IMREAD_GRAYSCALE)

        copper_mask = segmap & C.classes.copper.label > 0

        m = cv2.dilate(copper_mask.astype('u1'), kernel, iterations=1)
        ys, xs = np.where(m)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        out[name] = {
            'align_bbox': [x0, y0, x1, y1],
        }

    json.dump(out, open(dst, 'w'), indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
