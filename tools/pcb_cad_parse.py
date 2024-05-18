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

    #kernel = disk(etching_radius)
    img = np.asarray(img)
    #img = cv2.erode(img, kernel)

    back_f = osp.join(src, 'back.png')
    if osp.exists(back_f):
        # back = pil_rescale(Image.open(back_f), scale, order='nearest').transpose(Image.FLIP_LEFT_RIGHT)
        back = pil_rescale(Image.open(back_f), scale, order='nearest')
        back = back.convert('L')

        # back = back.crop((x0, y0, x1, y1))
        # for target gmm seg
        back = back.crop((x0, y0-v_gap, x1, y1)) 

        back = np.asarray(back)
        #back = cv2.erode(back, kernel)

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
@click.option('--use_valid_mask', default=True)
# @click.option('--no_valid_mask', is_flag=True)
@click.argument('src')
@click.argument('dst')
def main(cfg, ann, use_valid_mask, src, dst):
# def main(cfg, ann, no_valid_mask, src, dst):
    C = edict(toml.load(open(cfg)))

    ann = edict(toml.load(open(ann)))
    body_bbox = np.array(ann.body.bbox, dtype='f4')

    base_scale = C.target.base_dpi / ann.dpi
    print('base_scale is',base_scale)

    matches = {k: np.asarray(v, dtype='f4') for k, v in ann.matches.items()}
    cam_ids = sorted([i.split('_')[1] for i in matches if i.startswith('cad_')])
    print('cam_ids is ',cam_ids)

    shadow_color = 128
    division = C.base.division

    ## step 1. load and base_scale
    x0, y0, x1, y1 = ann.board.bbox

    # cad image
    crop_img_f = osp.join(dst, 'cad_image_1x.png')
    if not osp.exists(crop_img_f):
        img0, crop_bbox = load_cad_board(
            src,
            (x0, y0, x1, y1),
            scale=base_scale,
            etching_radius=ann.etching_radius,
            shadow_color=shadow_color,
            division=division,
        )
        imsave(crop_img_f, img0)

    else:
        img0 = Image.open(crop_img_f)

    # --- cad valid mask 
    if use_valid_mask == True:
        crop_valid_mask_f = osp.join(dst, 'cad_valid_mask_1x.png')
        if not osp.exists(crop_valid_mask_f):
            # if not no_valid_mask:
                valid_mask_f = 'annotations/valid_mask.png'
                valid_mask_src = pil_rescale(Image.open(valid_mask_f), base_scale, order='nearest')
                valid_mask_src = valid_mask_src.convert('L')
                x00, y00, x11, y11 = crop_bbox
                v_gap = 20
                valid_mask0 = valid_mask_src.crop((x00, y00-v_gap, x11, y11)) 
                imsave(crop_valid_mask_f, valid_mask0)
        else:
            valid_mask0 = Image.open(crop_valid_mask_f)

    # base_scale rest
    for cam_id in cam_ids:
        matches[f'cad_{cam_id}'] -= [x0, y0]
        matches[f'cad_{cam_id}'] *= base_scale

    body_bbox = (body_bbox.reshape(-1, 2) - [x0, y0]).flatten()
    body_bbox *= base_scale

    segmap_f = osp.join(dst, 'cad_segmap_1x.png')
    if not osp.exists(segmap_f):
        im = np.asarray(img0)
        canvas = np.zeros(im.shape[:2], dtype='u1')
        shadow_mask = im == shadow_color
        canvas[im == 255] = C.classes.copper.label
        canvas[im == 0] = C.classes.bg.label
        canvas[shadow_mask] = C.classes.bg_shadow.label
        imsave(segmap_f, canvas)

    # step 2. scale
    scale = C.base.align_scale
    resize_f = osp.join(dst, 'cad_image_2x.png')
    if not osp.exists(resize_f):
        w, h = img0.size
        img1 = img0.resize((int(w*C.base.align_scale), int(h*scale)), resample=Image.NEAREST)
        imsave(resize_f, img1)
    else:
        img1 = Image.open(resize_f)

    # --- cad valid mask 
    if use_valid_mask == True:
        resize_valid_mask_f = osp.join(dst, 'cad_valid_mask_2x.png')
        if not osp.exists(resize_valid_mask_f):
            w, h = valid_mask0.size
            valid_mask1 = valid_mask0.resize((int(w*C.base.align_scale),int(h*scale)), resample=Image.NEAREST)
            imsave(resize_valid_mask_f, valid_mask1)
        else:
            valid_mask1 = Image.open(resize_valid_mask_f)

    for scale, img, out_dir in [
            (1.0, img0, 'target_1x'),
            (1.0 / 2, img1, 'target_2x'),
    ]:
        # scale rest
        w, h = img.size
        cur_matches = {k: v * scale for k, v in matches.items()}
        cur_image_width = scale * ann.image_width
        cur_body_bbox = body_bbox * scale

        print('cur_image_width',cur_image_width)
        w_th = cur_image_width / 2

        # step 3. generate target_align
        img_dir = osp.join(dst, f'{out_dir}/images')
        os.makedirs(img_dir, exist_ok=True)
        segmap_dir = osp.join(dst, f'{out_dir}/segmaps')
        os.makedirs(segmap_dir, exist_ok=True)
        
        if use_valid_mask == True:
            valid_mask_dir = osp.join(dst, f'{out_dir}/valid_masks')
            os.makedirs(valid_mask_dir, exist_ok=True)

        align_regions = {}
        cut_regions = {}
        body_x0, body_y0, body_x1, body_y1 = cur_body_bbox

        for ci, cam_id in enumerate(cam_ids):
            print('cam id is ',cam_id)
            print('ci is ',ci)
            tile_pts = np.array(cur_matches[f'{cam_id}_cad'])
            cad_pts = np.array(cur_matches[f'cad_{cam_id}'])
            print('title pts shape is ',tile_pts.shape)
            print('cad pts shape is ',cad_pts.shape)

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
                print('tile_pts ',tile_pts)
                print('cad_pts ',cad_pts)
                print('w_th ',w_th)
                pts_l = [[a, b] for a, b in zip(tile_pts, cad_pts) if a[0] < w_th]
                pts_r = [[a, b] for a, b in zip(tile_pts, cad_pts) if a[0] > w_th]
                print('pts_r',pts_r)
                print('pts_l',pts_l)

                tile_mean_x = np.array([a for a, b in pts_l]).mean(0)[0]
                cad_mean_x = np.array([b for a, b in pts_l]).mean(0)[0]
                tile_dx = tile_mean_x
                cad_cut_x0 = cad_mean_x - tile_dx

                tile_mean_x = np.array([a for a, b in pts_r]).mean(0)[0]
                print('tile mean x is ',tile_mean_x)
                cad_mean_x = np.array([b for a, b in pts_r]).mean(0)[0]
                tile_dx = cur_image_width - tile_mean_x
                cad_cut_x1 = cad_mean_x + tile_dx

            cad_cut_y0 = 0
            cad_cut_y1 = h
            cad_cut_x0 = mod_len(cad_cut_x0, int(division * scale))
            cad_cut_x1 = mod_len(cad_cut_x1, int(division * scale))

            tile_img = img.crop((cad_cut_x0, cad_cut_y0, cad_cut_x1, cad_cut_y1))
            tile_img = np.asarray(tile_img)

            canvas = np.zeros_like(tile_img)
            shadow_mask = tile_img == shadow_color
            canvas[tile_img == 255] = C.classes.copper.label
            canvas[tile_img == 0] = C.classes.bg.label
            canvas[shadow_mask] = C.classes.bg_shadow.label

            tile_f = osp.join(segmap_dir, f'{cam_id}.png')
            imsave(tile_f, canvas)

            tile_img[shadow_mask] = 0

            img_f = osp.join(img_dir, f'{cam_id}.jpg')
            imsave(img_f, tile_img)

            if use_valid_mask == True:
                if scale == 1.0:
                    tile_valid_mask = valid_mask0.crop((cad_cut_x0, cad_cut_y0, cad_cut_x1, cad_cut_y1))
                    tile_valid_mask_f = osp.join(valid_mask_dir, f'{cam_id}.png')
                    imsave(tile_valid_mask_f, tile_valid_mask)
                if scale == 0.5:
                    tile_valid_mask = valid_mask1.crop((cad_cut_x0, cad_cut_y0, cad_cut_x1, cad_cut_y1))
                    tile_valid_mask_f = osp.join(valid_mask_dir, f'{cam_id}.png')
                    imsave(tile_valid_mask_f, tile_valid_mask)

            align_regions[cam_id] = {
                'align_bbox': [
                    int(max(0, body_x0 - cad_cut_x0)),
                    int(body_y0 - cad_cut_y0),
                    int(min(cad_cut_x1 - cad_cut_x0, body_x1 - cad_cut_x0)),
                    int(body_y1 - cad_cut_y0),
                ],
            }
            cut_regions[cam_id] = {
                'cut_bbox':[
                    cad_cut_x0,
                    cad_cut_y0,
                    cad_cut_x1,
                    cad_cut_y1,
                ],
            }

        json.dump(align_regions, open(osp.join(dst, f'{out_dir}/align_region.json'), 'w'), indent=2)
        json.dump(cut_regions, open(osp.join(dst, f'{out_dir}/cut_region.json'), 'w'), indent=2)

if __name__ == "__main__":
    main()
