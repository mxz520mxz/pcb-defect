#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import numpy as np
import toml
import json
from easydict import EasyDict as edict
from skimage.morphology import disk
import cv2
from deeppcb.base import Image, imsave, pil_rescale
import click

def mod_bbox(bbox, division=8):
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    w = int((w // division) * division)
    h = int((h // division) * division)
    return x0, y0, x0+w, y0+h

def load_cad_board(src, bbox, scale, etching_radius, shadow_color=128, division=8):

    img = pil_rescale(Image.open(osp.join(src, 'front.png')), scale, order='nearest')
    img = img.convert('L')

    x0, y0, x1, y1 = mod_bbox([i*scale for i in bbox], division)
    # img = img.crop((x0, y0, x1, y1))
    # for target gmm seg
    v_gap = 20
    img = img.crop((x0, y0-v_gap, x1, y1)) 

    kernel = disk(etching_radius)
    img = np.asarray(img)
    img = cv2.erode(img, kernel)

    back_f = osp.join(src, 'back.png')
    if osp.exists(back_f):
        # back = pil_rescale(Image.open(back_f), scale, order='nearest').transpose(Image.FLIP_LEFT_RIGHT)
        back = pil_rescale(Image.open(back_f), scale, order='nearest')
        back = back.convert('L')

        # back = back.crop((x0, y0, x1, y1))
        # for target gmm seg
        back = back.crop((x0, y0-v_gap, x1, y1)) 

        back = np.asarray(back)
        back = cv2.erode(back, kernel)

        back[img > 0] = 0
        img[back > 0] = shadow_color
        return Image.fromarray(img), [x0, y0, x1, y1]
    else:
        return Image.fromarray(img), [x0, y0, x1, y1]


def mod_len(l, division):
    return (l // division) * division

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--ann', default='annotations/annotation.toml')
# @click.argument('src')
# @click.argument('dst')
# def main(cfg, ann, src, dst):
def main(cfg, ann):

    cfg = '/home/vision/users/dengsx/pcb_py/deeppcb/config/config.toml'
    ann = '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/BR0100P04W0070087A1_L2/annotations/annotation.toml'
    template_cad_1x = '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/BR0100P04W0070087A1_L2/cad_image_1x.png'
    template_cad_2x = '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/BR0100P04W0070087A1_L2/cad_image_2x.png'
    defect_results_2x = '/home/vision/users/dengsx/pcb_py/data_bench_server/result_2x/8530/A'

    ann = edict(toml.load(open(ann)))
    matches = {k: np.asarray(v, dtype='f4') for k, v in ann.matches.items()}
    cam_ids = sorted([i.split('_')[1] for i in matches if i.startswith('cad_')])
    x0, y0, x1, y1 = ann.board.bbox
    C = edict(toml.load(open(cfg)))
    base_scale = C.target.base_dpi / ann.dpi

    print(matches)
    print(cam_ids)
    print(x0, y0, x1, y1)

    cad_2x = cv2.imread(template_cad_2x)
    b_channel, g_channel, r_channel = cv2.split(cad_2x) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道
    cad_2x = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    print(cad_2x.shape)

    deviations_dir = osp.join(defect_results_2x, 'defect_global/verify_deviations/')
    foreigns_dir = osp.join(defect_results_2x, 'defect_global/verify_foreigns/')

    for id,name in enumerate(os.listdir(deviations_dir)):
        deviations_cam1 = cv2.imread(osp.join(deviations_dir, name))
        if 'cam1' in name:
            cam1_dev_f = osp.join(deviations_dir, name)
        if 'cam2' in name:
            cam2_dev_f = osp.join(deviations_dir, name)
        if 'cam3' in name:
            cam3_dev_f = osp.join(deviations_dir, name)
        if 'cam4' in name:
            cam4_dev_f = osp.join(deviations_dir, name)
        if 'cam5' in name:
            cam5_dev_f = osp.join(deviations_dir, name)


    # for cam_id in cam_ids:
    #     x, y = matches[f'cad_{cam_id}'] - [x0, y0]
    #     # matches[f'cad_{cam_id}'] *= base_scale
    #     deviations_cam = cv2.imread(f'cam{cam_id}_dev_f', cv2.IMREAD_UNCHANGED)
    #     img_bg[y2:y1,x1:x2] = img_white


    # scale rest
    w, h = img.size
    cur_matches = {k: v * scale for k, v in matches.items()}
    cur_image_width = scale * ann.image_width
    cur_body_bbox = body_bbox * scale

    w_th = cur_image_width / 2


    for ci, cam_id in enumerate(cam_ids):
        tile_pts = np.array(cur_matches[f'{cam_id}_cad'])
        cad_pts = np.array(cur_matches[f'cad_{cam_id}'])

        if ci == 0:
            tile_mean_x = tile_pts.mean(0)[0]
            cad_mean_x = cad_pts.mean(0)[0]
            tile_dx = cur_image_width - tile_mean_x
            cad_cut_x1 = cad_mean_x + tile_dx
            cad_cut_x0 = 0

        elif ci == len(cam_ids) - 1:
            tile_mean_x = tile_pts.mean(0)[0]
            cad_mean_x = cad_pts.mean(0)[0]
            tile_dx = tile_mean_x
            cad_cut_x0 = cad_mean_x - tile_dx
            cad_cut_x1 = w

        else:
            pts_l = [[a, b] for a, b in zip(tile_pts, cad_pts) if a[0] < w_th]
            pts_r = [[a, b] for a, b in zip(tile_pts, cad_pts) if a[0] > w_th]

            tile_mean_x = np.array([a for a, b in pts_l]).mean(0)[0]
            cad_mean_x = np.array([b for a, b in pts_l]).mean(0)[0]
            tile_dx = tile_mean_x
            cad_cut_x0 = cad_mean_x - tile_dx

            tile_mean_x = np.array([a for a, b in pts_r]).mean(0)[0]
            cad_mean_x = np.array([b for a, b in pts_r]).mean(0)[0]
            tile_dx = cur_image_width - tile_mean_x
            cad_cut_x1 = cad_mean_x + tile_dx

        cad_cut_y0 = 0
        cad_cut_y1 = h

        tile_img = img.crop((cad_cut_x0, cad_cut_y0, cad_cut_x1, cad_cut_y1))
        tile_img = np.asarray(tile_img)


if __name__ == "__main__":
    main()
