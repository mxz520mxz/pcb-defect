#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import numpy as np
import toml
from copy import deepcopy
from easydict import EasyDict as edict
from functools import partial

from deeppcb.base import imsave, imread
from deeppcb.target import read_list
from deeppcb.ood import process_ood_seg, verify_ood_seg
from deeppcb.utils import run_parallel, update_zoomed_len, get_zoom

import click

def file_name(f):
    return osp.basename(osp.splitext(f)[0])

def process_one(tsk, images, segmaps, templates, dst, verify, zoom, C):
    zoom = get_zoom(zoom, images)

    img_name, board, face, cam_id = tsk
    name, ext = osp.splitext(img_name)
    tpl_dir = osp.join(templates, board)

    tpl_segmap = imread(osp.join(tpl_dir, f'segmaps/{cam_id}.png'), to_gray=True)

    dst_f = osp.join(dst, name+'.png')
    if osp.exists(dst_f):
        return

    img_name = name+'.jpg'
    img = imread(osp.join(images, img_name))
    segmap_name = name+'.png'
    segmap = imread(osp.join(segmaps, segmap_name))

    cfg = deepcopy(C.target.ood_seg.copper)
    for k in ['segmap_shrink', 'edge_region_radius']:
        update_zoomed_len(cfg, k, zoom)

    copper_mask = segmap == C.classes.copper.label
    copper_ood_mask, copper_info = process_ood_seg(img, copper_mask, cfg)
    segmap[copper_ood_mask] |= C.classes.wl_copper.label

    cfg = deepcopy(C.target.ood_seg.bg)
    for k in ['segmap_shrink', 'edge_region_radius']:
        update_zoomed_len(cfg, k, zoom)

    bg_mask = segmap == C.classes.bg.label
    bg_shadow_mask = tpl_segmap == C.classes.bg_shadow

    bg_ood_mask, bg_info = process_ood_seg(img, bg_mask, cfg, shadow_mask=bg_shadow_mask)
    segmap[bg_ood_mask] |= C.classes.wl_bg.label

    os.makedirs(osp.dirname(dst_f), exist_ok=True)
    imsave(dst_f, segmap)

    if verify:
        segmaps = {
            'copper': copper_info['mask'],
            'bg': bg_info['mask']
        }
        sample_points = {
            'copper': copper_info['sample_points'],
            'bg': bg_info['sample_points']
        }

        canvas = verify_ood_seg(img, segmaps, sample_points)

        dst_f = osp.join(verify, name + '_ood_samples.jpg')
        os.makedirs(osp.dirname(dst_f), exist_ok=True)
        imsave(dst_f, canvas)


@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--templates', default='')
@click.option('--debug', is_flag=True)
@click.option('--jobs', default=4)
@click.option('--verify', default='')
@click.option('--zoom', default=0)
@click.argument('lst')
@click.argument('images')
@click.argument('segmaps')
@click.argument('dst')
def main(cfg, templates, debug, jobs, verify, zoom, lst, images, segmaps, dst):
    C = edict(toml.load(cfg))

    tsks = read_list(lst, C.target.cam_mapping)

    worker = partial(
        process_one,
        images=images,
        segmaps=segmaps,
        templates=templates,
        dst=dst,
        verify=verify,
        zoom=zoom,
        C=C,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
