#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from glob import glob
import numpy as np
import json
from functools import partial
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from deeppcb.base import imsave
from deeppcb.cell import cut_cells
import click
import pylab as plt
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def process_one(tsk, src, dst, board, C):
    cam_id, cells = tsk
    img_fs = glob(osp.join(src, f'{cam_id}*'))
    assert len(img_fs) == 1
    img_f = img_fs[0]
    ext = osp.splitext(img_f)[-1]

    img = Image.open(img_f)
    rets = cut_cells(img, cells, C.pat_sizes, C.image_width, order=C.order, border_value=C.border_value)

    for ret in tqdm(rets):
        cell = ret.cell
        cell_img = ret.img
        row, col = cell.pos
        pat = cell.pattern

        dst_name = f'board:{board}-cam:{cam_id}-pat:{pat}-pos:{row:02d}_{col:02d}'

        pat_dst = dst.format(pattern=pat)
        os.makedirs(pat_dst, exist_ok=True)
        dst_f = osp.join(pat_dst, dst_name+ext)
        imsave(dst_f, cell_img)

@click.command()
@click.option('--cells', default='stitched/cells.json')
@click.option('--order', default='linear')
@click.option('--border_value', default=0)
@click.option('--board', default=osp.basename(osp.abspath('.')))
@click.option('--jobs', default=4)
@click.option('--debug', is_flag=True)
@click.argument('src')
@click.argument('dst')
def main(cells, order, border_value, board, jobs, debug, src, dst):
    C = edict(json.load(open(cells)))
    C.order = order
    C.border_value = border_value

    tsks = C.tile_cells.items()
    worker = partial(
        process_one,
        src=src,
        dst=dst,
        board=board,
        C=C
    )
    if debug:
        for tsk in tsks:
            worker(tsk)
    else:
        with Pool(min(len(tsks), jobs)) as p:
            p.map(worker, tsks)

if __name__ == "__main__":
    main()
