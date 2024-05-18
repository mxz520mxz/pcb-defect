#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import math as M
import numpy as np
import numpy.linalg as npl
from skimage.transform import SimilarityTransform, AffineTransform, ProjectiveTransform
from glob import glob
from PIL import Image, ImageDraw
import networkx as nx
from deeppcb.align_menpo import menpofit_lk
from deeppcb.align_ocv import ocv_find_ecc
from deeppcb.align_edge import align_edge
from deeppcb.base import estimate_transform, transform_img, imsave, imread
from deeppcb.stitch import stitch_images
import toml
import re
from easydict import EasyDict as edict
import pylab as plt
from tqdm import tqdm
import json
import click

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def process_pair(pil0, pil1, matched_pts, mask0, cfg):
    imw0, imh0 = pil0.size
    imw1, imh1 = pil1.size
    assert imw0 == imw1

    max_h = max(imh0, imh1)
    min_h = min(imh0, imh1)

    xdiff = np.mean([a - b for (a, _), (b, _) in matched_pts])
    ydiff = np.mean([a - b for (_, a), (_, b) in matched_pts])
    hoverlap = imw0 - xdiff

    moving_hgap = cfg.moving_hgap * imw0
    moving_x0 = imw0 - hoverlap + moving_hgap
    moving_x1 = imw0 - moving_hgap
    assert moving_x1 > moving_x0

    moving_vgap0, moving_vgap1 = [int(i * max_h) for i in cfg.moving_vgap]
    moving_vs = [i for (_, i), _ in matched_pts]
    moving_v0 = min(moving_vs)
    moving_v1 = max(moving_vs)

    moving_y0 = max(moving_v0 - moving_vgap0, 0)
    moving_y1 = min(moving_v1 + moving_vgap1, min_h)

    if mask0 is not None:
        mask0 = mask0.crop((moving_x0, moving_y0, moving_x1, moving_y1))
        mask0 = np.asarray(mask0) > 128

    moving_img = pil0.crop((moving_x0, moving_y0, moving_x1, moving_y1))
    fixed_img = pil1.crop((0, 0, hoverlap, imh1))

    moving_w, moving_h = moving_img.size

    dx = moving_x0 - xdiff
    dy = moving_y0 - ydiff
    init_bbox = [dx, dy, dx + moving_w, dy + moving_h]

    if cfg.align_method == 'menpofit_lk':
        ## TODO
        pass
        # similarity has BUG, use affine instead
        # refined_bbox = menpofit_lk(moving_img, fixed_img, init_bbox,
        #                            tform_tp='affine')['fixed_bbox']
    elif cfg.align_method == 'opencv_ecc':
        ret = ocv_find_ecc(moving_img, fixed_img, init_bbox, tform_tp=cfg.tform_tp)

    elif cfg.align_method == 'edge_align':
        if cfg.edge_align.H_sx_pix < 1:
            H_sx = cfg.edge_align.H_sx_pix
        else:
            H_sx = cfg.edge_align.H_sx_pix / fixed_img.width

        if cfg.edge_align.H_sy_pix < 1:
            H_sy = cfg.edge_align.H_sy_pix
        else:
            H_sy = cfg.edge_align.H_sy_pix / fixed_img.height

        ret = align_edge(
            np.asarray(moving_img), np.asarray(fixed_img),
            init_bbox,
            moving_mask=mask0,
            tform_tp=cfg.tform_tp,
            H_sx=H_sx,
            H_sy=H_sy,
            **cfg.edge_align)
    else:
        raise

    H_fixed_moving = ret['H_20']
    H_moving_orig = np.array([
        [1, 0, -moving_x0],
        [0, 1, -moving_y0],
        [0, 0, 1]
    ])
    H_10 = H_fixed_moving.dot(H_moving_orig)

    xs0 = [
        [moving_x0, moving_y0],
        [moving_x0, moving_y0 + moving_h],
        [moving_x0 + moving_w, moving_y0 + moving_h],
        [moving_x0 + moving_w, moving_y0]
    ]

    xs1 = ProjectiveTransform(H_10)(xs0)

    return H_10, {
        'xs0': xs0,
        'xs1': xs1,
    }

def get_images_width(img_dir):
    widths = set(Image.open(img_f).size[0] for img_f in glob(img_dir+'/*.jpg'))
    assert len(widths) == 1, f"multiple widths {widths}"
    return next(iter(widths))

@click.command()
@click.option('--data_dir', default='.')
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--crop', default='vcrop.json')
@click.option('--ann', default='annotations/annotation.toml')
@click.option('--verify', is_flag=True)
@click.option('--verify_stitch_factor', default=1.3)
@click.option('--cache_dir', default='cache')
@click.option('--verify_dir', default='verify')
@click.option('--stitch_masks', default='stitch_masks')
@click.argument('images')
@click.argument('images_orig')
@click.argument('stitched_dir')
def main(data_dir, cfg, crop, ann, verify, verify_stitch_factor, cache_dir, verify_dir, stitch_masks, images, images_orig, stitched_dir):
    data_dir = osp.abspath(data_dir).rstrip('/')
    cfg = edict(toml.load(open(cfg))['golden']['stitch'])
    cfg['ann_orig'] = toml.load(open(ann))

    crop = json.load(open(crop))
    y_offs = {k: v['vrange'][0] for k, v in crop.items()}

    assert osp.exists(images_orig)
    assert osp.exists(images)

    width_orig = get_images_width(images_orig)
    width_stitch = get_images_width(images)

    cache_dir = osp.join(data_dir, cache_dir)
    verify_dir = osp.join(data_dir, verify_dir)
    stitched_dir = osp.join(data_dir, stitched_dir)

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(verify_dir, exist_ok=True)
    os.makedirs(stitched_dir, exist_ok=True)

    cam_names = [str(i) for i in cfg.cam_names]
    cam_pairs = [(str(u), str(v)) for u, v in cfg.cam_pairs]

    img_dic = {file_name(i): {'filename': osp.join(images, i)}
               for i in os.listdir(images)}

    scale = width_stitch / width_orig

    cfg.annotations = edict({
        'rotation': {
            k: [[x*scale, (y - y_offs[k])*scale] for x, y in v]
            for k, v in cfg.ann_orig.rotation.items()
        },
        'matches': {
            k: [[x*scale, (y - y_offs[k.split('_')[0]])*scale] for x, y in v]
            for k, v in cfg.ann_orig.matches.items()
        }
    })

    pils = {}
    verify_pils = {}

    g = nx.DiGraph()

    tform = cfg.tform_tp

    if tform == 'similarity':
        Transform = SimilarityTransform
    elif tform == 'affine':
        Transform = AffineTransform
    elif tform == 'projective':
        Transform = ProjectiveTransform
    else:
        raise

    for idx, (i, j) in enumerate(tqdm(cam_pairs)):
        img_f0 = img_dic[i]['filename']
        img_f1 = img_dic[j]['filename']

        if i not in pils:
            pils[i] = Image.open(img_f0)
            if verify:
                verify_pils[i] = pils[i].copy()

        if j not in pils:
            pils[j] = Image.open(img_f1)
            if verify:
                verify_pils[j] = pils[j].copy()

        pil0 = pils[i]
        pil1 = pils[j]
        img_dic[i]['size'] = pil0.size
        img_dic[j]['size'] = pil1.size

        ann_matches = cfg.annotations.matches
        matched_pts = list(zip(ann_matches[f'{i}_{j}'], ann_matches[f'{j}_{i}']))
        mask_f = osp.join(stitch_masks, i+'.png')
        mask0 = None
        if osp.exists(mask_f):
            mask0 = Image.open(mask_f).convert('L')

        T_10, info = process_pair(pil0, pil1, matched_pts, mask0, cfg)
        T_01 = npl.inv(T_10)

        xs0, xs1 = info['xs0'], info['xs1']

        g.add_edge(i, j, T=T_01)
        g.add_edge(j, i, T=T_10)

        if verify:
            v_pil0 = verify_pils[i]
            v_pil1 = verify_pils[j]
            draw0 = ImageDraw.Draw(v_pil0)
            draw1 = ImageDraw.Draw(v_pil1)
            for (u0, v0), (u1, v1) in zip(xs0, xs1):
                draw0.ellipse((u0-10, v0-10, u0+10, v0+10))
                draw1.ellipse((u1-10, v1-10, u1+10, v1+10))

            draw0.polygon([tuple(i) for i in xs0], outline='blue')
            draw1.polygon([tuple(i) for i in xs1], outline='blue')

            imsave(osp.join(verify_dir, osp.basename(img_f0)),v_pil0)
            if idx == len(cam_pairs) - 1:
                imsave(osp.join(verify_dir, osp.basename(img_f1)),v_pil1)

    ref_cam = str(cfg.ref_cam)

    rot_pts = cfg.annotations.rotation[ref_cam]
    (x0, y0), (x1, y1) = rot_pts
    theta = M.pi/2 - M.atan2(y1 - y0, x1 - x0)

    ref_R = SimilarityTransform(rotation=theta).params

    Ts = {
        ref_cam: np.asarray(ref_R),
    }
    for cam in cam_names:
        if cam == ref_cam:
            continue

        p = nx.shortest_path(g, ref_cam, cam)
        T = Ts[ref_cam]
        for i, j in zip(p[:-1], p[1:]):
            T_ij = g.edges[i, j]['T']
            T = T.dot(T_ij)

        Ts[cam] = T

    warp_bounds = {}
    for k, T in Ts.items():
        T = Transform(T)

        w, h = img_dic[k]['size']
        bounds = [[0, 0], [w, 0], [w, h], [0, h]]
        trans_bounds = T(bounds)
        warp_bounds[k] = trans_bounds

    x_min = min([v[:,0].min() for v in warp_bounds.values()])
    x_max = max([v[:,0].max() for v in warp_bounds.values()])
    y_min = min([v[:,1].min() for v in warp_bounds.values()])
    y_max = max([v[:,1].max() for v in warp_bounds.values()])

    o_w = M.ceil(x_max)
    o_h = M.ceil(y_max)

    Ts_f = osp.join(stitched_dir, 'Ts.json')
    json.dump({
        'image_width': width_stitch,
        'golden_size': [o_w, o_h],
        'golden_Ts_wc': {k: v.tolist() for k, v in Ts.items()},
    }, open(Ts_f, 'w'), indent=2, sort_keys=True)

    ret = stitch_images(pils, Ts, (o_w, o_h, 3))

    dst_f = osp.join(stitched_dir, "stitched_image.jpg")
    imsave(dst_f, ret.img)
    dst_f = osp.join(stitched_dir, "stitched_acc_mask.png")
    imsave(dst_f, ret.acc_mask)

    if verify:
        for p in ret.parts:
            imsave(osp.join(stitched_dir, f'warp_img_{p.name}.jpg'), p.img)
            imsave(osp.join(stitched_dir, f'warp_mask_{p.name}.png'), p.mask.astype('u1') * 255)

        v_canvas = np.zeros((o_h, o_w, 3), dtype='f4')
        for k, T in Ts.items():
            v_img = verify_pils[k]
            v_trans_img = transform_img(v_img, T, (o_w, o_h))
            v_canvas += np.array(v_trans_img)

        dst_f = osp.join(stitched_dir, "stitched_image_verify.jpg")
        print (f"saving verify canvas {dst_f}")

        mask = ret.acc_mask
        overlap_mask = mask >= 2
        v_canvas = v_canvas / (mask[...,None] + 1e-3)
        v_canvas[overlap_mask] *= verify_stitch_factor
        v_canvas = v_canvas.clip(0, 255)
        imsave(dst_f, v_canvas.astype('u1'))

if __name__ == "__main__":
    main()
