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

Image.MAX_IMAGE_PIXELS = None

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def get_range(start, end, size, overlap):
    r = [start]
    cur = start
    while True:
        next_one = cur + size - overlap
        if next_one >= end:
            break
        r.append(next_one)
        cur = next_one

    r.append(end)
    return r

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--size', '-s', default=512)
@click.option('--overlap', default=64)
@click.option('--debug', is_flag=True)
@click.argument('src')
@click.argument('dst')
def main(cfg, size, overlap, debug, src, dst):
    C = edict(toml.load(cfg))

    segmap_fs = sorted(glob(osp.join(src, '*.png')))
    w, h = Image.open(segmap_fs[0]).size

    ys = get_range(0, h-size, size, overlap)
    xs = get_range(0, w-size, size, overlap)

    patch_bboxes = [([row, col], [x, y, x + size, y + size]) for row, y in enumerate(ys) for col, x in enumerate(xs)]
    out = {
        'cell_size': [w, h],
        'patch_bboxes': patch_bboxes,
        'partial_cells': {},
    }

    for segmap_f in tqdm(segmap_fs):
        pcell_name = file_name(segmap_f)

        ppatch_items = []

        segmap = cv2.imread(segmap_f, cv2.IMREAD_GRAYSCALE)

        for (row, col), (x0, y0, x1, y1) in patch_bboxes:
            sm = segmap[y0:y1, x0:x1]
            blank_mask = np.asarray(sm) == 0
            # ignore_mask = sm == C.classes.ignore.label

            if blank_mask.sum() / blank_mask.size > 0.05:
                continue

            pname = f'{pcell_name}-ppos:{row}_{col}'

            ppatch_items.append({
                'bbox': [x0, y0, x1, y1],
                'pos': [row, col],
                'name': pname,
            })

        if debug:
            canvas = cv2.imread(segmap_f)
            cs = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
            thicks = [1, 3, 2, 4]
            for idx, (x0, y0, x1, y1) in enumerate(bboxes):
                cv2.rectangle(canvas, (x0, y0), (x1, y1), cs[idx%4], thickness=thicks[idx%4])

            plt.imshow(canvas)
            plt.show()

        out['partial_cells'][pcell_name] = {
            'patches': ppatch_items
        }

    pickle.dump(out, open(dst, 'wb'))

if __name__ == "__main__":
    main()
