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
import pickle
from functools import partial
from easydict import EasyDict as edict
from multiprocessing import Pool
import cv2
from deeppcb.base import imsave, imread, parse_name
import click
from tqdm import tqdm
import pylab as plt

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--templates', default='templates')
@click.argument('src')
@click.argument('dst')
def main(cfg, templates, src, dst):
    C = edict(toml.load(open(cfg)))
    os.makedirs(dst, exist_ok=True)

    cur_key = None
    for idx, img_name in enumerate(sorted(os.listdir(src))):
        info = edict(parse_name(img_name))
        key = info.id.split('_')[-1]
        if key != cur_key:
            if cur_key is not None:
                imsave(osp.join(dst, cur_key+'.jpg'), canvas)

            cur_key = key
            board = info.board
            tpl_dir = osp.join(templates, board)
            canvas = imread(osp.join(tpl_dir, 'stitched/stitched_image.jpg'))

            patch_cfg = pickle.load(open(osp.join(tpl_dir, 'patches/patches.pkl'), 'rb'))
            cw, ch = patch_cfg['cell_size']
            patch_box_hash = {tuple(rc): (x0/cw, y0/ch, x1/cw, y1/ch) for rc, (x0, y0, x1, y1) in patch_cfg['patch_bboxes']}

            cells_cfg = json.load(open(osp.join(tpl_dir, 'stitched/cells.json')))

            cell_hash = {tuple(c['pos']): c for c in cells_cfg['cells']}

        px0, py0, px1, py1 = patch_box_hash[tuple(info.ppos)]
        ci = cell_hash[tuple(info.pos)]
        cx0, cy0, cx1, cy1 = ci['box']
        cw = cx1 - cx0
        ch = cy1 - cy0

        x0, y0, x1, y1 = cx0 + px0 * cw, cy0 + py0 * ch, cx0 + px1 * cw, cy0 + py1 * ch
        cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (0,255,0), 2)

    if cur_key is not None:
        imsave(osp.join(dst, cur_key+'.jpg'), canvas)

if __name__ == "__main__":
    main()
