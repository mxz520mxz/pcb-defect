#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import cv2
import numpy as np
import toml
from easydict import EasyDict as edict
from functools import partial
from glob import glob
import json
from time import time
from multiprocessing import Pool
from loguru import logger
import pickle
from deeppcb.base import resize, imsave, get_gray, get_edge, imread, rescale, scale_H, transform_img, Image
from deeppcb.target import read_list, file_name, process_crop, process_filter, process_align_camera, verify_align_camera, verify_train_gmm, process_train_gmm, process_seg_gmm, get_blank_mask, draw_segmap
from deeppcb.foreign import detect_foreigns, draw_defects
from deeppcb.cell import cut_cells, apply_cell_cuttings

import pylab as plt
import click

stage_dag = {
    'crop': {
        'in': {
            'img': 'start.img',
        },
        'out': ['img'],
    },
    'filter': {
        'in': {
            'img': 'crop.img',
        },
        'out': ['img'],
    },
    'resize_align': {
        'in': {
            'img': 'filter.img',
        },
        'out': ['img'],
    },
    'estimate_camera_align': {
        'in': {
            'img': 'resize_align.img',
        },
        'out': ['tform'],
    },
    'align_camera': {
        'in': {
            'tform': 'estimate_camera_align.tform',
            'img': 'filter.img',
        },
        'out': ['img', 'yrb'],
    },
    'train_gmm': {
        'in': {
            'yrb': 'align_camera.yrb',
        },
        'out': ['gmm'],
    },
    'seg_gmm': {
        'in': {
            'yrb': 'align_camera.yrb',
            'gmm': 'train_gmm.gmm',
        },
        'out': ['segmap'],
    },
    'detect_foreigns': {
        'in': {
            'yrb': 'align_camera.yrb',
            'segmap': 'seg_gmm.segmap',
        },
        'out': ['defects'],
    },
    'cut_cells': {
        'in': {
            'img': 'align_camera.img',
            'segmap': 'seg_gmm.segmap',
        },
        'out': ['cells'],
    },
    'align_cells': {
        'in': {
            'cells': 'cut_cells.cells',
        },
        'out': ['cells'],
    },
}

def get_input(ctx, stage, key):
    print (f"get_input {stage} {key}")
    inp = stage_dag[stage]['in'][key]
    req_stage = inp.split('.')[0]
    if inp not in ctx:
        print (f"get_input {stage} {key} processing {req_stage}")
        stage_processors[req_stage](ctx)
        assert inp in ctx
    return ctx[inp]

def get_img(ctx, stage):
    return get_input(ctx, stage, 'img')

def save_output(ctx, stage, key, v):
    outs = stage_dag[stage]['out']
    assert key in outs
    ctx[f'{stage}.{key}'] = v

def save_img(ctx, stage, v):
    return save_output(ctx, stage, 'img', v)

def free_input(ctx, stage, key='img'):
    inputs = stage_dag[stage]['in']
    if key in inputs and inputs[key] in ctx:
        ctx.pop(inputs[key])

def save_free_img(ctx, stage, v):
    save_img(ctx, stage, v)
    free_input(ctx, stage, 'img')

def run_stage_crop(ctx):
    stage = 'crop'
    name, C = ctx.name, ctx.C
    save_path = ''
    if ctx.save_crop:
        save_path = osp.join(ctx.save_crop, ctx.img_name)
        if osp.exists(save_path):
            save_free_img(ctx, stage, Image.open(save_path))
            return

    img = get_img(ctx, stage)
    tpl_edge = imread(f'{ctx.tpl_dir}/edges_orig/{ctx.cam_id}.png')
    cfg = C.target.crop

    logger.info(f'{name}: process_crop start. save_path "{save_path}"')
    img = process_crop(
        img,
        tpl_edge,
        scale=cfg.scale,
    )
    logger.info(f'{name}: process_crop end')

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        imsave(save_path,img)

    save_free_img(ctx, stage, img)

def run_stage_filter(ctx):
    stage = 'filter'
    name, C = ctx.name, ctx.C

    save_path = ''
    if ctx.save_filter:
        save_path = osp.join(ctx.save_filter, ctx.img_name)
        if osp.exists(save_path):
            save_free_img(ctx, stage, imread(save_path))
            return

    img = get_img(ctx, stage)
    img = np.asarray(img)
    cfg = C.target.filter
    logger.info(f'{name}: process_filter start. save_path "{save_path}"')
    img = process_filter(
        img,
        d=cfg.d,
        sigma_color=cfg.sigma_color,
        sigma_space=cfg.sigma_space,
    )
    logger.info(f'{name}: process_filter end')

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        imsave(save_path, img)

    save_free_img(ctx, stage, img)

def run_stage_resize_align(ctx):
    stage = 'resize_align'
    name, C = ctx.name, ctx.C

    save_path = ''
    if ctx.save_resize_align:
        save_path = osp.join(ctx.save_resize_align, ctx.img_name)
        if osp.exists(save_path):
            save_free_img(ctx, stage, imread(save_path))
            return

    img = np.asarray(get_img(ctx, stage))
    logger.info(f'{name}: process_resize_align start. save_path "{save_path}"')

    ratio = C.base.align_width / img.shape[1]
    if ratio < 1:
        img = rescale(img, ratio)

    logger.info(f'{name}: process_resize_align end')

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        imsave(save_path, img)

    save_img(ctx, stage, img)

def run_stage_estimate_camera_align(ctx):
    stage = 'estimate_camera_align'
    name, C = ctx.name, ctx.C

    save_path = ''
    if ctx.save_transform:
        save_path = osp.join(ctx.save_transform, name+'.json')
        if osp.exists(save_path):
            save_output(ctx, stage, 'tform', json.load(open(save_path)))
            return

    img = np.asarray(get_img(ctx, stage))
    tgt_tpl_dir = osp.join(ctx.tpl_dir, 'target_align')
    align_region = json.load(open(osp.join(tgt_tpl_dir, 'align_region.json')))[ctx.cam_id]
    tpl_distmap = imread(osp.join(tgt_tpl_dir, 'distmaps', ctx.cam_id+'.jpg'), to_gray=True)

    cfg = C.target.align_camera
    print(cfg)

    tpl_w = tpl_distmap.shape[1]
    x0, y0, x1, y1 = align_region['align_bbox']
    gap = int(cfg.gap * tpl_w)
    x0 += gap
    y0 += gap
    x1 -= gap
    y1 -= gap

    init_bbox = [x0, y0, x1, y1]
    ret = process_align_camera(
        img,
        tpl_distmap,
        init_bbox,
        tform_tp=cfg.tform_tp,
        lr=cfg.edge_align.lr,
        max_patience=cfg.edge_align.max_patience,
        max_iters=cfg.edge_align.max_iters,
        msg_prefix=f'{name}: ',
    )

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        json.dump(ret, open(save_path, 'w'), indent=2)

    save_output(ctx, stage, 'tform', ret)
    free_input(ctx, stage, 'img')

    verify = ctx.verify_transform
    if verify:
        os.makedirs(verify, exist_ok=True)
        tpl_img = imread(osp.join(tgt_tpl_dir, 'images', ctx.cam_id+'.jpg'))

        init_f = osp.join(verify, name + '_init.jpg')
        H = np.asarray(ret['H_21'])
        init_canvas, final_canvas = verify_align_camera(H, img, tpl_img)
        imsave(osp.join(verify, name + '_init.jpg'), init_canvas)
        imsave(osp.join(verify, name + '_warp.jpg'), final_canvas)

def run_stage_align_camera(ctx):
    stage = 'align_camera'
    name, C = ctx.name, ctx.C

    save_path = ''
    if ctx.save_aligned_images:
        save_path = osp.join(ctx.save_aligned_images, name+'.jpg')
        if osp.exists(save_path):
            img = imread(save_path)
            save_img(ctx, stage, img)
            yrb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            save_output(ctx, stage, 'yrb', yrb)
            return

    img = np.asarray(get_img(ctx, stage))
    s = img.shape[1] / C.base.align_width
    tform = get_input(ctx, stage, 'tform')
    print('tform',tform)
    print('s',img.shape[1],C.base.align_width)

    H = scale_H(tform['H_21'], s)
    tpl_h, tpl_w = tform['tpl_shape']
    tpl_h = int(tpl_h * s)
    tpl_w = int(tpl_w * s)

    img = transform_img(img, H, (tpl_w, tpl_h))

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        imsave(save_path, img)

    save_free_img(ctx, stage, img)
    yrb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    save_output(ctx, stage, 'yrb', yrb)

def run_stage_train_gmm(ctx):
    stage = 'train_gmm'
    name, C = ctx.name, ctx.C

    save_path = ''
    if ctx.save_train_gmm:
        save_path = osp.join(ctx.save_train_gmm, name+'.pkl')
        if osp.exists(save_path):
            save_output(ctx, stage, 'gmm', pickle.load(open(save_path, 'rb')))
            return

    yrb = np.asarray(get_input(ctx, stage, 'yrb'))
    tpl_segmap = imread(osp.join(ctx.tpl_dir, f'segmaps_orig/{ctx.cam_id}.png'), to_gray=True)

    cfg = C.target.train_gmm
    out, info = process_train_gmm({
        'yrb': yrb
    }, tpl_segmap, img_blank_mask=get_blank_mask(yrb[...,0]), **cfg)

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        pickle.dump(out, open(save_path, 'wb'))

    save_output(ctx, stage, 'gmm', out)

    verify = ctx.verify_train_gmm
    if verify:
        os.makedirs(verify, exist_ok=True)
        canvas = verify_train_gmm(**info)
        imsave(osp.join(verify, name+'_gmm_sample.jpg'), canvas)

def run_stage_seg_gmm(ctx):
    stage = 'seg_gmm'
    name, C = ctx.name, ctx.C

    save_path = ''
    if ctx.save_seg_gmm:
        save_path = osp.join(ctx.save_seg_gmm, name+'.png')
        if osp.exists(save_path):
            segmap = imread(save_path, to_gray=True)
            save_output(ctx, stage, 'segmap', segmap)
            return

    yrb = np.asarray(get_input(ctx, stage, 'yrb'))
    gmm = get_input(ctx, stage, 'gmm')

    segmap = process_seg_gmm({
        'yrb': yrb
    }, gmm, C.classes, blank_mask=get_blank_mask(yrb[...,0]))

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        imsave(save_path, segmap)

    save_output(ctx, stage, 'segmap', segmap)

    verify = ctx.verify_seg_gmm
    if verify:
        os.makedirs(verify, exist_ok=True)
        canvas = draw_segmap(segmap, C.classes)
        imsave(osp.join(verify, name+'.jpg'), canvas)

def run_stage_detect_foreigns(ctx):
    stage = 'detect_foreigns'
    name, C = ctx.name, ctx.C

    save_path = ''
    if ctx.save_foreigns:
        save_path = osp.join(ctx.save_foreigns, name+'.pkl')
        if osp.exists(save_path):
            d = pickle.load(open(save_path, 'rb'))
            buf = np.frombuffer(d['segmap']['data'], dtype='u1')
            d['segmap'] = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            return

    segmap = get_input(ctx, stage, 'segmap')
    yrb = get_input(ctx, stage, 'yrb')
    imh, imw = segmap.shape

    detect_mask = imread(osp.join(ctx.tpl_dir, f'stitched/cell_masks/{ctx.cam_id}.png'), to_gray=True)
    detect_mask = resize(detect_mask, (imw, imh)) > 128
    defects, defect_ctx = detect_foreigns(segmap, {'yrb': yrb}, detect_mask, C)

    segmap = defects['segmap']
    fmt = '.png'
    d = cv2.imencode(fmt, defects['segmap'])[1].tobytes()
    defects['segmap'] = {
        'format': fmt,
        'data': d,
    }

    black_nr = 0
    gray_nr = 0
    light_nr = 0
    for k, o in defects['objects'].items():
        if o['level'] == 'black':
            black_nr += 1
        elif o['level'] == 'gray':
            gray_nr += 1

        if o['type'] == 'shallow_water':
            light_nr += 1

    logger.info(f'{name}: black_nr {black_nr}, gray_nr {gray_nr}, light_nr {light_nr}, group_nr {len(defects["groups"])}')

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        pickle.dump(defects, open(save_path, 'wb'))

    save_output(ctx, stage, 'defects', defects)

    def draw_box_fn(o):
        if o['level'] == 'black':
            return {
                'color': [255, 0, 0],
                'thickness': 10,
            }
        elif o['level'] == 'gray':
            return {
                'color': [255, 255, 0],
                'thickness': 5,
            }
        elif o['area'] > 10:
            return {
                'color': [0, 255, 0],
                'thickness': 5,
            }

    verify = ctx.verify_foreigns
    if verify:
        os.makedirs(verify, exist_ok=True)
        canvas = np.zeros((imh, imw, 4), dtype='u1')
        draw_defects(canvas, defects, defect_ctx, C, box_fn=draw_box_fn)
        Image.fromarray(canvas).save(osp.join(verify, name+'.png'))

def load_cells(src, name):
    assert osp.exists(src)
    p = osp.join(src, f'transforms/{name}-*.pkl')
    cell_names = [file_name(i) for i in sorted(glob(p))]
    cells = {}
    for n in cell_names:
        cell = {
            'tform': pickle.load(open(osp.join(src, f'transforms/{n}.pkl'), 'rb')),
        }
        img_f = osp.join(src, f'images/{n}.jpg')
        if osp.exists(img_f):
            cell['img'] = imread(img_f)
        img_f = osp.join(src, f'segmaps/{n}.png')
        if osp.exists(img_f):
            cell['segmap'] = imread(img_f, to_gray=True)

        cells[n] = cell

    return cells

def run_stage_cut_cells(ctx):
    stage = 'cut_cells'
    name, C = ctx.name, ctx.C
    save_dir = ctx.save_cut_cells
    if save_dir:
        if osp.exists(save_dir):
            cells = load_cells(save_dir, name)
            save_output(ctx, stage, 'cells', cells)
            return

    cfg = edict(json.load(open(osp.join(ctx.tpl_dir, 'stitched/cells.json'))))
    tile_cells = cfg.tile_cells[ctx.cam_id]
    img = get_img(ctx, stage)
    segmap = get_input(ctx, stage, 'segmap')

    rets = cut_cells(img, tile_cells, cfg.pat_sizes, cfg.image_width, order='linear')
    cell_segmaps = apply_cell_cuttings(segmap, rets, order='nearest')

    out = {}
    for ret, cell_segmap in zip(rets, cell_segmaps):
        cell_img = ret.pop('img')

        row, col = ret.pos
        pat = ret.pattern

        cell_name = f'{name}-board:{ctx.board}-cam:{ctx.cam_id}-pat:{pat}-pos:{row:02d}_{col:02d}'

        out[cell_name] = {
            'tform': ret,
            'segmap': cell_segmap,
            'img': cell_img,
        }

    if save_dir:
        for folder in ['images', 'segmaps', 'transforms']:
            os.makedirs(osp.join(save_dir, folder),  exist_ok=True)

        for k, v in out.items():
            dst_f = osp.join(save_dir, 'images', k+'.jpg')
            imsave(dst_f, v['img'])

            dst_f = osp.join(save_dir, 'segmaps', k+'.png')
            imsave(dst_f, v['segmap'])

            dst_f = osp.join(save_dir, 'transforms', k+'.pkl')
            pickle.dump(v['tform'], open(dst_f, 'wb'))

    save_output(ctx, stage, 'cells', out)

def align_cell(name, cell, C):
    cfg = C.align_target
    img = cell['segmap']
    blank_mask = img == 0
    fg_mask = img & C.classes.copper.label > 0
    fg_mask[blank_mask] = False
    blank_mask = binary_dilation(blank_mask, disk(cfg.valid_margin))

    conts = extract_cell_contours(fg_mask, blank_mask, min_len=cfg.min_length)


def run_stage_align_cells(ctx):
    stage = 'align_cells'
    name, C = ctx.name, ctx.C
    save_dir = ctx.save_align_cells
    if save_dir:
        if osp.exists(save_dir):
            cells = load_cells(save_dir, name)
            save_output(ctx, stage, 'cells', cells)
            return

    cells = get_input(ctx, stage, 'cells')
    for k, v in cells.items():
        align_cell(k, v, C)

stage_processors = {
    'start': lambda x: x,
    'crop': run_stage_crop,
    'filter': run_stage_filter,
    'resize_align': run_stage_resize_align,
    'estimate_camera_align': run_stage_estimate_camera_align,
    'align_camera': run_stage_align_camera,
    'train_gmm': run_stage_train_gmm,
    'seg_gmm': run_stage_seg_gmm,
    'detect_foreigns': run_stage_detect_foreigns,
    'cut_cells': run_stage_cut_cells,
    'align_cells': run_stage_align_cells,
}

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--debug', is_flag=True)
@click.option('--workspace', default='.')
@click.option('--templates', default='templates')
@click.option('--save_crop', default='')
@click.option('--save_filter', default='')
@click.option('--save_resize_align', default='')
@click.option('--save_transform', default='')
@click.option('--save_aligned_images', default='')
@click.option('--save_train_gmm', default='')
@click.option('--save_seg_gmm', default='')
@click.option('--save_foreigns', default='')
@click.option('--save_cut_cells', default='')
@click.option('--save_align_cells', default='')
@click.option('--verify_transform', default='')
@click.option('--verify_train_gmm', default='')
@click.option('--verify_seg_gmm', default='')
@click.option('--verify_foreigns', default='')
@click.option('--verify_align_cells', default='')
@click.argument('stage')
@click.argument('img')
def main(cfg, debug, workspace, templates,
         save_crop,
         save_filter,
         save_resize_align,
         save_transform,
         save_aligned_images,
         save_train_gmm,
         save_seg_gmm,
         save_foreigns,
         save_cut_cells,
         save_align_cells,
         verify_transform,
         verify_train_gmm,
         verify_seg_gmm,
         verify_foreigns,
         verify_align_cells,
         stage, img):

    C = edict(toml.load(cfg))

    ws = workspace
    cls_info = read_list(osp.join(workspace, 'list.txt'), C.target.cam_mapping, return_dict=True)

    img_f = img
    img_name = osp.basename(img_f)
    name = file_name(img_name)
    info = cls_info[name]

    ctx = edict({
        'C': C,
        'name': name,
        'img_name': osp.basename(img_f),
        'tpl_dir': osp.join(ws, templates, info.board),
        'board': info.board,
        'cam_id': info.cam_id,

        'save_crop': save_crop,
        'save_filter': save_filter,
        'save_resize_align': save_resize_align,
        'save_transform': save_transform,
        'save_aligned_images': save_aligned_images,
        'save_train_gmm': save_train_gmm,
        'save_seg_gmm': save_seg_gmm,
        'save_foreigns': save_foreigns,
        'save_cut_cells': save_cut_cells,
        'save_align_cells': save_align_cells,

        'verify_transform': verify_transform,
        'verify_train_gmm': verify_train_gmm,
        'verify_seg_gmm': verify_seg_gmm,
        'verify_foreigns': verify_foreigns,
        'verify_align_cells': verify_align_cells,

        'start.img': Image.open(img_f),
    })

    stage_processors[stage](ctx)

if __name__ == "__main__":
    main()
