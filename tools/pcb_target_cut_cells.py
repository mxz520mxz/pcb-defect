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
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import pickle
from deeppcb.base import imsave
from deeppcb.cell import cut_cells
from deeppcb.target import read_list
from deeppcb.utils import run_parallel
import click
import pylab as plt
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def process_one(tsk, images, templates, dst, out_tform, order, border_value):
    img_name, board, face, cam_id = tsk
    name, ext = osp.splitext(img_name)

    tpl_dir = osp.join(templates, board)
    cells_f = osp.join(tpl_dir, 'stitched/cells.json')

    C = edict(json.load(open(cells_f)))

    img = Image.open(osp.join(images, img_name))
    rets = cut_cells(img, C.tile_cells[cam_id], C.pat_sizes, C.image_width, order=order, border_value=border_value)

    for ret in tqdm(rets):
        cell_img = ret.pop('img')

        ret['T'] = ret['T_10']
        row, col = ret.pos
        pat = ret.pattern

        dst_name = f'{name}-board:{board}-cam:{cam_id}-pat:{pat}-pos:{row:02d}_{col:02d}'
        dst_f = osp.join(dst, dst_name+ext)
        os.makedirs(dst, exist_ok=True)
        imsave(dst_f, cell_img)

        if out_tform:
            dst_f = osp.join(out_tform, dst_name+'.pkl')
            os.makedirs(out_tform, exist_ok=True)
            pickle.dump(ret, open(dst_f, 'wb'))

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--templates', default='templates')
@click.option('--order', default='linear')
@click.option('--border_value', default=0)
@click.option('--jobs', default=4)
@click.option('--debug', is_flag=True)
@click.option('--out_tform', default='')
@click.argument('lst')
@click.argument('images')
@click.argument('dst')
def main(cfg, templates, order, border_value, jobs, debug, out_tform, lst, images, dst):
    C = edict(toml.load(open(cfg)))

    tsks = read_list(lst, C.target.cam_mapping)
    ext = osp.splitext(os.listdir(images)[0])[-1]
    tsks = [(osp.splitext(name)[0]+ext, *rest) for name, *rest in read_list(lst, C.target.cam_mapping) if osp.exists(osp.join(images, osp.splitext(name)[0]+ext))]

    worker = partial(
        process_one,
        images=images,
        templates=templates,
        dst=dst,
        out_tform=out_tform,
        order=order,
        border_value=border_value,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
