import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import cv2
import numpy as np
import toml
from copy import deepcopy
from easydict import EasyDict as edict
from functools import partial
from glob import glob
import json
from time import time
from multiprocessing import Pool
from loguru import logger
import pickle
from deeppcb.base import resize, imsave, get_gray, get_edge, imread, rescale, scale_H, transform_img, Image
from deeppcb.target import read_list, file_name, process_crop, process_filter, process_align_camera, process_gmm_seg, get_blank_mask, get_gmm_img, get_feature_img
from deeppcb.draw import verify_align_camera, draw_defects
from deeppcb.ood import process_ood_seg, verify_ood_seg
from deeppcb.foreign import detect_foreigns, draw_defects
from deeppcb.deviation import detect_deviations
from deeppcb.utils import run_parallel

import pylab as plt
import click

# cv2.setNumThreads(2)

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
        # 'out': ['img', 'yrb'],
        'out': ['img'],
    },
    'seg_gmm': {
        'in': {
            # 'yrb': 'align_camera.yrb',
            'img': 'align_camera.img',
        },
        'out': ['segmap'],
    },
    'seg_ood': {
        'in': {
            'img': 'align_camera.img',
            'gmm': 'seg_gmm.segmap',
        },
        'out': ['segmap_ood'],
    },
    'detect_foreigns': {
        'in': {
            'img': 'align_camera.img', #lab or img
            'segmap': 'seg_ood.segmap_ood',
        },
        'out': ['defects'],
    },
    'detect_deviations': {
        'in': {
            'img': 'align_camera.img',
            'segmap': 'seg_gmm.segmap',
        },
        'out': ['defects'],
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
    tpl_edge = imread(f'{ctx.tpl_dir}/target_1x/edges/{ctx.cam_id}.png')
    ys, xs = np.where(tpl_edge)
    tpl_cx, tpl_cy = xs.mean(), ys.mean()
    h, w = tpl_edge.shape[:2]

    cfg = C.target.crop

    logger.info(f'{name}: process_crop start. save_path "{save_path}"')
    img = process_crop(
        img,
        (tpl_cx, tpl_cy), 
        (w, h),
        scale=cfg.edge_scale,
    )
    logger.info(f'{name}: process_crop end')

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        imsave(save_path,img)

    save_free_img(ctx, stage, img)

    return ctx, stage, img
    
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
        d=cfg.d['1x'], # scale
        sigma_color=cfg.sigma_color,
        sigma_space=cfg.sigma_space['1x'],
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

    # ratio = C.base.align_width / img.shape[1]
    ratio = C.base.align_scale
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
    tgt_tpl_dir = osp.join(ctx.tpl_dir, 'target_2x')
    align_region = json.load(open(osp.join(tgt_tpl_dir, 'align_region.json')))[ctx.cam_id]
    tpl_distmap = imread(osp.join(tgt_tpl_dir, 'distmaps', ctx.cam_id+'.jpg'), to_gray=True)

    cfg = C.target.align_camera
    print(cfg)

    tpl_h, tpl_w = tpl_distmap.shape[:2]
    x0, y0, x1, y1 = align_region['align_bbox']
    padding = int(cfg.padding / 1)
    x0 += padding
    y0 += padding
    x1 -= padding
    y1 -= padding

    init_bbox = [x0, y0, x1, y1]
    ret = process_align_camera(
        img,
        tpl_distmap,
        init_bbox,
        tform_tp=cfg.tform_tp,
        # lr=cfg.edge_align.lr,
        # max_patience=cfg.edge_align.max_patience,
        # max_iters=cfg.edge_align.max_iters,
        msg_prefix=f'{name}: ',
        **cfg.edge_align,
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
        if tpl_img.ndim == 2:
            tpl_img = cv2.cvtColor(tpl_img, cv2.COLOR_GRAY2RGB)

        init_f = osp.join(verify, name + '_init.jpg')
        H = np.asarray(ret['H_21'])
        init_canvas, final_canvas = verify_align_camera(H, img, tpl_img)
        imsave(osp.join(verify, name + '_init.jpg'), init_canvas)
        imsave(osp.join(verify, name + '_warp.jpg'), final_canvas)

def run_stage_align_camera(ctx):
    stage = 'align_camera'
    name, C = ctx.name, ctx.C
    #print('ctx', ctx)

    save_path = ''
    if ctx.save_aligned_images:
        save_path = osp.join(ctx.save_aligned_images, name+'.jpg')
        if osp.exists(save_path):
            img = imread(save_path)
            save_img(ctx, stage, img)
            yrb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            # save_output(ctx, stage, 'yrb', yrb)
            save_output(ctx, stage, 'img', img)
            return

    img = np.asarray(get_img(ctx, stage))
    # s = img.shape[1] / C.base.align_width
    s = 1 / C.base.align_scale
    tform = get_input(ctx, stage, 'tform')
    print('s',s)

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
    # save_output(ctx, stage, 'yrb', yrb)
    save_output(ctx, stage, 'img', img)

def draw_segmap(img, C):
    imh, imw = img.shape
    canvas = np.zeros((imh, imw, 3), dtype='u1')

    for cls_name, v in C.items():
        canvas[img & v.label > 0] = v.color
    
    return canvas

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

    cfg = C.target.gmm_seg

    img = np.asarray(get_img(ctx, stage))
    # img = np.asarray(get_input(ctx, stage, 'img'))
    imdic = get_gmm_img(img, cfg.feature)
    img = get_feature_img(imdic, cfg.feature)

    segmap = process_gmm_seg(        
        img,
        C.classes,
        blank_mask=get_blank_mask(img[...,0]),
        sample_nr=cfg.sample_nr,
        chull_scale=1,
        random_seed=cfg.random_seed,
    )

    bw = cfg.blank_border_width
    if bw:
        segmap[:bw, :] = 0
        segmap[-bw:, :] = 0
        segmap[:, :bw] = 0
        segmap[:, -bw:] = 0

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        imsave(save_path, segmap)

    save_output(ctx, stage, 'segmap', segmap)

    verify = ctx.verify_seg_gmm
    if verify:
        os.makedirs(verify, exist_ok=True)
        canvas = draw_segmap(segmap, C.classes)
        imsave(osp.join(verify, name+'.jpg'), canvas)

def run_stage_seg_ood(ctx):
    stage = 'seg_ood'
    name, C = ctx.name, ctx.C

    save_path = ''
    if ctx.save_seg_ood:
        save_path = osp.join(ctx.save_seg_ood, name+'.png')
        if osp.exists(save_path):
            segmap = imread(save_path, to_gray=True)
            save_output(ctx, stage, 'segmap_ood', segmap)
            return

    tpl_segmap = imread(f'{ctx.tpl_dir}/target_1x/segmaps/{ctx.cam_id}.png', to_gray=True)

    img = np.asarray(get_input(ctx, stage, 'img')) # lab?
    segmap = get_input(ctx, stage, 'gmm')

    cfg = deepcopy(C.target.ood_seg.copper)
    # for k in ['segmap_shrink', 'edge_region_radius']:         # to do: zoom
    #     update_zoomed_len(cfg, k, zoom)

    copper_mask = segmap == C.classes.copper.label
    copper_ood_mask, copper_info = process_ood_seg(img, copper_mask, cfg)
    segmap[copper_ood_mask] |= C.classes.wl_copper.label

    cfg = deepcopy(C.target.ood_seg.bg)
    # for k in ['segmap_shrink', 'edge_region_radius']:
    #     update_zoomed_len(cfg, k, zoom)

    bg_mask = segmap == C.classes.bg.label
    bg_shadow_mask = tpl_segmap == C.classes.bg_shadow

    bg_ood_mask, bg_info = process_ood_seg(img, bg_mask, cfg, shadow_mask=bg_shadow_mask)
    segmap[bg_ood_mask] |= C.classes.wl_bg.label

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        imsave(save_path, segmap)

    save_output(ctx, stage, 'segmap_ood', segmap)

    verify = ctx.verify_seg_ood
    if verify:
        os.makedirs(verify, exist_ok=True)

        segmaps = {
            'copper': copper_info['mask'],
            'bg': bg_info['mask']
        }
        sample_points = {
            'copper': copper_info['sample_points'],
            'bg': bg_info['sample_points']
        }

        canvas = verify_ood_seg(img, segmaps, sample_points)

        imsave(osp.join(verify, name+'_ood_samples.jpg'), canvas)

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
    img = get_input(ctx, stage, 'img')
    imh, imw = segmap.shape

    no_valid_mask = False # to do
    mask = None
    if no_valid_mask == False:
        tile_valid_mask_f = osp.join(ctx.tpl_dir, f'target_1x/valid_masks/{ctx.cam_id}.png') # target_1x: no in .py??
        #print('tile_valid_mask_f',tile_valid_mask_f)
        tpl_valid_mask = imread(tile_valid_mask_f, to_gray=True)
        mask = np.zeros_like(segmap, dtype=bool)
        mask[tpl_valid_mask == 255] = True
    else:
        print('no valid mask')
        mask = True

    cfg = deepcopy(C.foreign)
    # for k in [                                    # to do: zoom
    #         'crop_border',
    #         'copper_margin',
    #         'bg_margin',
    #         'cluster.cluster_dist',
    #         # 'cluster.max_edge_points',

    #         'inland_sea.surr_radius',
    #         'inland_sea.floodfill_tol',

    #         'insea_land.surr_radius',
    #         'insea_land.floodfill_tol',

    #         'deep_water.surr_radius',
    #         'deep_water.min_intensity_var',
    #         'deep_water.min_rb_var',

    #         'high_sand.surr_radius',
    #         'high_sand.min_intensity_var',
    #         'high_sand.min_rb_var',

    #         'shallow_water.surr_radius',
    #         'shallow_water.floodfill_tol',
    #         'shallow_water.max_intensity_range',

    #         'shallow_sand.surr_radius',
    #         'shallow_sand.floodfill_tol',
    #         'shallow_sand.max_intensity_range',

    # ]:
    #     update_zoomed_len(cfg, k, zoom)

    # for k in [
    #         'inland_sea.max_area',
    #         'inland_sea.min_area',

    #         'insea_land.max_area',
    #         'insea_land.min_area',

    #         'small_pond.max_area',

    #         'small_reef.max_area',
            
    # ]:
    #     update_zoomed_area(cfg, k, zoom)

    defects, ctx_foreigns = detect_foreigns(segmap, img, C.classes, cfg, mask=mask)

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
    for k, obj in defects['objects'].items():
        if obj['level'] == 'black':
            black_nr += 1
        elif obj['level'] == 'gray':
            gray_nr += 1
        if obj['type'] == 'shallow_water':
            light_nr += 1

    logger.info(f'{name}: black_nr {black_nr}, gray_nr {gray_nr}, light_nr {light_nr}, group_nr {len(defects["groups"])}')

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        pickle.dump(defects, open(save_path, 'wb'))

    # save_output(ctx, stage, 'defects', defects)        # 'int' error??????????????????

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
        # draw_defects(canvas, defects, box_fn=draw_box_fn, cfg=C.foreign)
        detect_type = 'foreigns'
        draw_defects(detect_type, name, img, canvas, defects, box_fn=draw_box_fn, cfg=C.foreign) # save patches
        Image.fromarray(canvas).save(osp.join(verify, name+'.png'))

    return verify

def run_stage_detect_deviations(ctx):
    stage = 'detect_deviations'
    name, C = ctx.name, ctx.C

    save_path = ''
    if ctx.save_deviations:
        save_path = osp.join(ctx.save_deviations, name+'.pkl')
        if osp.exists(save_path):
            d = pickle.load(open(save_path, 'rb'))
            buf = np.frombuffer(d['segmap']['data'], dtype='u1')
            d['segmap'] = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            return

    tpl_distmap = imread(f'{ctx.tpl_dir}/target_1x/distmaps/{ctx.cam_id}.jpg', to_gray=True)
    tpl_segmap = imread(f'{ctx.tpl_dir}/target_1x/segmaps/{ctx.cam_id}.png', to_gray=True)

    segmap = get_input(ctx, stage, 'segmap')
    img = get_input(ctx, stage, 'img')
    imh, imw = segmap.shape

    no_valid_mask = False # to do
    mask = None
    if no_valid_mask == False:
        tile_valid_mask_f = osp.join(ctx.tpl_dir, f'target_1x/valid_masks/{ctx.cam_id}.png') # target_1x: no in .py??
        print('tile_valid_mask_f',tile_valid_mask_f)
        tpl_valid_mask = imread(tile_valid_mask_f, to_gray=True)
        mask = np.zeros_like(segmap, dtype=bool)
        mask[tpl_valid_mask == 255] = True
    else:
        align_region = json.load(open(osp.join(ctx.tpl_dir, f'align_region.json')))
        print('no valid mask')
        x0, y0, x1, y1 = align_region[str(ctx.cam_id)]['align_bbox']
        mask = np.zeros_like(segmap, dtype=bool)
        mask[y0:y1, x0:x1] = True

    # C_1x = deepcopy(C)                          
    # for k in [                                         # to do
    #         'border_gap',
    #         'align_contour_margin',
    #         'connect_len',
    #         'coarse_far_dist_th',
    #         # 'coarse_far_ratio',
    #         'coarse_near_dist_th',
    #         'strict_dist_th',
    #         # 'strict_ratio',
    #         'refine_margin',
    #         'refine_dist_th',
            
    #         'cluster.cluster_dist',
    #         # 'cluster.max_edge_points',
    # ]:
    #     update_zoomed_len(C.deviation, k, zoom)

    # cfg = deepcopy(C.deviaitons) # no need?

    defects, ctx_deviations = detect_deviations(name, segmap, tpl_segmap, tpl_distmap, C, mask=mask)
    fmt = '.png'
    d = cv2.imencode(fmt, defects['segmap'])[1].tobytes()
    defects['segmap'] = {
        'format': fmt,
        'data': d,
    }

    convex_nr = 0
    concave_nr = 0
    for k, obj in defects['objects'].items():
        if obj['type'] == 'convex':
            convex_nr += 1
        elif obj['type'] == 'concave':
            concave_nr += 1

    logger.info(f'{name}: convex_nr {convex_nr}, concave_nr {concave_nr}') # group?????????????

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        pickle.dump(defects, open(save_path, 'wb'))

    # save_output(ctx, stage, 'defects', defects)        # 'int' error??????????????????

    verify = ctx.verify_deviations
    if verify:
        def draw_box_fn(o):
            if 'type' not in o or o['type'] == 'concave':
                return {
                    'color': [255, 0, 0],
                    'thickness': 10,
                }
            elif o['type'] == 'convex':
                return {
                    'color': [255, 255, 0],
                    'thickness': 5,
                }

        os.makedirs(verify, exist_ok=True)
        canvas = np.zeros((imh, imw, 4), dtype='u1')
        # draw_defects(canvas, defects, box_fn=draw_box_fn, cfg=C.deviation)
        detect_type = 'deviations'
        draw_defects(detect_type, name, img, canvas, defects, box_fn=draw_box_fn, cfg=C.deviation) # save patches
        Image.fromarray(canvas).save(osp.join(verify, name+'.png'))

    return verify

def run_stage_detect_foreigns_deviations(ctx):
    verify_deviations = run_stage_detect_deviations(ctx)
    verify_foreigns = run_stage_detect_foreigns(ctx)

    return verify_deviations, verify_foreigns

stage_processors = {
    'start': lambda x: x,
    'crop': run_stage_crop,
    'filter': run_stage_filter,
    'resize_align': run_stage_resize_align,
    'estimate_camera_align': run_stage_estimate_camera_align,
    'align_camera': run_stage_align_camera,
    'seg_gmm': run_stage_seg_gmm,
    'seg_ood': run_stage_seg_ood,
    'detect_foreigns': run_stage_detect_foreigns,
    'detect_deviations': run_stage_detect_deviations,
    'detect_foreigns_deviations': run_stage_detect_foreigns_deviations,
}

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--debug', is_flag=True)
@click.option('--workspace', default='.')
@click.option('--templates', default='templates')
@click.option('--save_crop', default='save_crop')
@click.option('--save_filter', default='save_filter')
@click.option('--save_resize_align', default='save_resize_align')
@click.option('--save_transform', default='save_transform')
@click.option('--save_aligned_images', default='save_aligned_images')
@click.option('--save_seg_gmm', default='save_seg_gmm')
@click.option('--save_seg_ood', default='save_seg_ood')
@click.option('--save_foreigns', default='save_foreigns')
@click.option('--save_deviations', default='save_deviations')
@click.option('--verify_transform', default='verify_transform')
@click.option('--verify_seg_gmm', default='verify_seg_gmm')
@click.option('--verify_seg_ood', default='verify_seg_ood')
@click.option('--verify_foreigns', default='verify_foreigns')
@click.option('--verify_deviations', default='verify_deviations')
@click.option('--stage', default='detect_foreigns_deviations')
@click.option('--img', default='images0/20211024T013639S465_cam1_11417.jpg')

# @click.argument('stage')
# @click.argument('img')
def main(cfg, debug, workspace, templates,
         save_crop,
         save_filter,
         save_resize_align,
         save_transform,
         save_aligned_images,
         save_seg_gmm,
         save_seg_ood,
         save_foreigns,
         save_deviations,
         verify_transform,
         verify_seg_gmm,
         verify_seg_ood,
         verify_foreigns,
         verify_deviations,
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
        'save_seg_gmm': save_seg_gmm,
        'save_seg_ood': save_seg_ood,
        'save_foreigns': save_foreigns,
        'save_deviations': save_deviations,

        'verify_transform': verify_transform,
        'verify_seg_gmm': verify_seg_gmm,
        'verify_seg_ood': verify_seg_ood,
        'verify_foreigns': verify_foreigns,
        'verify_deviations': verify_deviations,

        'start.img': Image.open(img_f),
    })

    stage_processors[stage](ctx)

if __name__ == "__main__":
    main()
