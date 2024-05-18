#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import numpy as np
import json
import cv2
import toml
from easydict import EasyDict as edict
from functools import partial
from multiprocessing import Pool
import pylab as plt

from deeppcb.base import imsave, imread, rescale, transform_img, Image
from deeppcb.utils import run_parallel, get_zoom
from deeppcb.target import read_list, process_align_camera, file_name
from deeppcb.draw import verify_align_camera
import click

def process_one(tsk, images, dst_tforms, dst_images, templates, verify, cfg, debug=True, zoom=None):
    img_name, board, face, cam_id = tsk
    name = osp.splitext(img_name)[0]
    dst_f = osp.join(dst_tforms, f'{name}.json')
    if osp.exists(dst_f):
        return

    zoom = get_zoom(zoom, images)

    tgt_tpl_dir = templates.format(board=board)
    align_region = json.load(open(osp.join(tgt_tpl_dir, 'align_region.json')))[cam_id]

    img = imread(osp.join(images, img_name))

    tpl_distmap = imread(osp.join(tgt_tpl_dir, 'distmaps', cam_id+'.jpg'), to_gray=True)

    tpl_h, tpl_w = tpl_distmap.shape[:2]
    x0, y0, x1, y1 = align_region['align_bbox']
    padding = int(cfg.padding / zoom)
    x0 += padding
    y0 += padding
    x1 -= padding
    y1 -= padding

    init_bbox = [x0, y0, x1, y1]
    ret = process_align_camera(
        img, tpl_distmap, init_bbox,
        tform_tp=cfg.tform_tp,
        msg_prefix=f'{name}: ',
        **cfg.edge_align,
    )

    os.makedirs(osp.dirname(dst_f), exist_ok=True)
    ret['zoom'] = zoom
    json.dump(ret, open(dst_f, 'w'), indent=2)

    H = np.asarray(ret['H_21'])
    warped_img = transform_img(img, H, (tpl_w, tpl_h))
    os.makedirs(dst_images, exist_ok=True)
    imsave(osp.join(dst_images, f'{name}.jpg'), warped_img)

    if verify:
        os.makedirs(verify, exist_ok=True)

        tpl_img = imread(osp.join(tgt_tpl_dir, 'images', cam_id+'.jpg'))
        if tpl_img.ndim == 2:
            tpl_img = cv2.cvtColor(tpl_img, cv2.COLOR_GRAY2RGB)

        init_f = osp.join(verify, name + '_init.jpg')
        init_canvas, final_canvas = verify_align_camera(H, img, tpl_img)
        imsave(osp.join(verify, name + '_init.jpg'), init_canvas)
        imsave(osp.join(verify, name + '_warp.jpg'), final_canvas)

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--templates', default='')
@click.option('--debug', is_flag=True)
@click.option('--jobs', default=4)
@click.option('--verify', default='')
@click.argument('lst')
@click.argument('images')
@click.argument('dst_tforms')
@click.argument('dst_images')
def main(cfg, templates, debug, jobs, verify, lst, images, dst_tforms, dst_images):
    C = edict(toml.load(cfg))

    tsks = read_list(lst, C.target.cam_mapping)

    worker = partial(
        process_one,
        images=images,
        templates=templates,
        verify=verify,
        cfg=C.target.align_camera,
        dst_tforms=dst_tforms,
        dst_images=dst_images,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
