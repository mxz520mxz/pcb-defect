#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import numpy as np
import numpy.linalg as npl
from collections import defaultdict, Counter
from easydict import EasyDict as edict
import cv2
from skimage.transform import ProjectiveTransform
from skimage.feature import peak_local_max
from skimage.draw import polygon
import toml
import json
from sklearn.cluster import DBSCAN
from deeppcb.base import imsave, rescale, get_gray, imread, Image
from deeppcb.stitch import inverse_project_cells
from pprint import pprint
import click
import pylab as plt

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def find_order(xs, eps):
    xs = np.asarray(xs)
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(xs[..., None])
    xs_labels = sorted(zip(xs.tolist(), labels, range(len(xs))))
    rank = -1
    prev_lbl = None
    out = []
    for x, lbl, id in xs_labels:
        if lbl != prev_lbl:
            rank += 1
            prev_lbl = lbl

        out.append((id, rank))

    out = sorted(out)
    return [r for _, r in out]

def remove_duplicate_cells(cells, pat_layout):
    pos_dic = defaultdict(list)
    for cell in cells:
        pos_dic[cell['pos']].append(cell)

    pat_dic = None
    if pat_layout:
        pat_dic = {(r, c): k for k, v in pat_layout['positions'].items() for _, r, c in v}

    out = []
    for k, v in pos_dic.items():
        if pat_dic:
            v = sorted([i for i in v if pat_dic[k] == i['pattern']], key=lambda x: x['score'], reverse=True)
            if len(v) == 0:
                print (k)
                breakpoint()

            cell = v[0]
            out.append(cell)
            continue

        if len(v) == 1:
            cell = v[0]
            out.append(cell)
            continue

        scores = [i['score'] for i in v]
        idx = np.argmax(scores)
        out.append(v[idx])

    return out

def get_mask_roi(mask):
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def stats_cells(cells):
    pats = sorted([i['pattern'] for i in cells])
    pat_set = sorted(set(pats))
    counts = Counter(pats)
    counts = {k: counts[k] for k in pat_set}
    return counts

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--pat_layout', default='annotations/pattern_layout.json')
@click.option('--tform', default='stitched/Ts.json')
@click.option('-o', '--out_f', default='stitched/cells.json')
@click.option('--match_scale', default=0.25)
@click.option('--match_nms_th', default=0.7)
@click.option('--debug', is_flag=True)
@click.option('--verify', default='verify')
@click.option('--stitched_segmap', default='stitched/stitched_segmap.png')
@click.option('--stitched_img', default='stitched/stitched_image.jpg')
@click.option('--stitched_patterns', default='stitched/patterns')
@click.option('--images_stitch', default='images_stitch')
@click.option('--min_inv_proj_overlap', default=0.02)
@click.option('--exclude_border', default=0.01)
def main(cfg, pat_layout, tform, out_f, match_scale, match_nms_th, debug, verify, stitched_segmap, stitched_img, stitched_patterns, images_stitch, min_inv_proj_overlap, exclude_border):
    C = edict(toml.load(open(cfg)))

    C_T = edict(json.load(open(tform)))
    if osp.exists(pat_layout):
        pat_layout = json.load(open(pat_layout))
    else:
        pat_layout = None

    pattern_colors = C.pattern_colors

    ref_width = C_T.image_width
    exclude_border = int(exclude_border * ref_width)

    Ts = {k: np.array(v) for k, v in C_T.golden_Ts_wc.items()}

    min_inv_proj_overlap *= ref_width

    stitched_img = imread(stitched_img)
    # stitched_gray = get_gray(stitched_img)
    # stitched_segmap = imread(stitched_segmap, to_gray=True)
    if verify:
        canvas = stitched_img.copy()

    tgt_img = stitched_img
    tgt_img_match = tgt_img
    if match_scale < 1:
        tgt_img_match = rescale(tgt_img, match_scale)

    match_method = cv2.TM_CCOEFF_NORMED
    # match_method = cv2.TM_SQDIFF

    found_cells = []

    min_match_h, min_match_w = float('inf'), float('inf')
    pat_sizes = {}

    factor = 0.8

    for pat_f in os.listdir(stitched_patterns):
        pat_name = file_name(pat_f)
        pat = imread(osp.join(stitched_patterns, pat_f), to_gray=True)

        pat_roi_mask = np.bitwise_and(pat, C.pattern.roi.label)
        if pat_roi_mask.sum() == 0:
            pat_roi_mask = np.bitwise_and(pat, C.pattern.border_points.extend_label)

        ys, xs = np.where(pat_roi_mask > 0)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        pat_match_region = np.bitwise_and(pat, C.pattern.match_region.label)
        if pat_match_region.sum() == 0:
            mask = np.bitwise_and(pat, C.pattern.border_points.label)
            xx0, yy0, xx1, yy1 = get_mask_roi(mask)
            pat_match_region[yy0:yy1, xx0:xx1] = 255

        tpl_img = tgt_img[y0:y1, x0:x1]
        tpl_mask = pat_match_region[y0:y1, x0:x1]

        # stats
        pat_sizes[pat_name] = [tpl_img.shape[1], tpl_img.shape[0]]

        xx0, yy0, xx1, yy1 = get_mask_roi(pat_match_region)
        min_match_w = min(min_match_w, (xx1 - xx0) * factor)
        min_match_h = min(min_match_h, (yy1 - yy0) * factor)

        tpl_img_match = rescale(tpl_img, match_scale)
        tpl_mask_match = rescale(tpl_mask, match_scale)
        result = cv2.matchTemplate(tgt_img_match, tpl_img_match, method=match_method, mask=tpl_mask_match)
        # factor for safe margin

        nms_dist = int(min(x1 - x0, y1 - y0) * factor * match_scale)
        # must have exclude_border, or exclude_border=nms_dist
        coords = peak_local_max(result, nms_dist, threshold_abs=match_nms_th, exclude_border=exclude_border)[..., ::-1]
        scores = result[coords[:, 1], coords[:, 0]]

        coords = (coords / match_scale).astype('i4')
        boxes = np.hstack((coords, coords))
        boxes[:,2] += tpl_img.shape[1]
        boxes[:,3] += tpl_img.shape[0]

        dists = npl.norm(coords - [x0, y0], axis=1)

        found_cells += [
            {'box': b, 'score': float(s), 'pattern': pat_name, 'is_ref': bool(d < 10)}
            for b, s, d in zip(boxes, scores, dists)
        ]

        if verify:
            for x0, y0, x1, y1 in boxes:
                pt0 = (int(x0), int(y0))
                pt1 = (int(x1), int(y1))
                cv2.rectangle(canvas, pt0, pt1, pattern_colors[pat_name], 1)

        if debug:
            plt.figure(f"score map {pat_name}")
            plt.imshow(result)

    if debug:
        plt.show()

    if verify:
        dst_f = osp.join(verify, 'stitched_match_cells.jpg')
        imsave(dst_f, canvas)

    xs = [v['box'][0] for v in found_cells]
    cols = find_order(xs, min_match_w - 1)
    ys = [v['box'][1] for v in found_cells]
    rows = find_order(ys, min_match_h - 1)

    for cell, r, c in zip(found_cells, rows, cols):
        cell['pos'] = r, c
    print('found_cells',len(found_cells),found_cells)
    print (f"before remove dup {len(found_cells)} {stats_cells(found_cells)}")
    found_cells = remove_duplicate_cells(found_cells, pat_layout)
    print (f"after remove dup {len(found_cells)} {stats_cells(found_cells)}")

    tile_cells =  inverse_project_cells(found_cells, Ts, ref_width, min_inv_proj_overlap)

    out_cells = [
        {
            'pos': list(v['pos']),
            'box': v['box'].tolist(),
            'pattern': v['pattern'],
            'score': v['score'],
            'angle': 0,
            'is_ref': v['is_ref'],
        }
        for v in found_cells
    ]

    ref_cell = {}
    for v in out_cells:
        if v['is_ref']:
            ref_cell[v['pattern']] = v

    out = {
        'image_width': ref_width,
        'grid_shape': (max(rows) + 1, max(cols) + 1),
        'ref_cell': ref_cell,
        'cells': out_cells,
        'pat_sizes': pat_sizes,
        'tile_cells': tile_cells,
    }

    json.dump(out, open(out_f, 'w'), indent=2)

    counts = stats_cells(out['cells'])
    print (f"final {len(out['cells'])} boxes detected with pattern_number {counts}")

    cell_masks_dir = osp.join(osp.dirname(out_f), 'cell_masks')
    os.makedirs(cell_masks_dir, exist_ok=True)
    for cam_id, v in tile_cells.items():
        imw, imh = Image.open(osp.join(images_stitch, f'{cam_id}.jpg')).size
        canvas = np.zeros((imh, imw), dtype='u1')
        for c in v:
            coords = np.array(c['corners']).astype('i4')
            rr, cc = polygon(coords[:,1].clip(0, imh-1), coords[:,0].clip(0, imw-1))
            canvas[rr, cc] = 255

        imsave(osp.join(cell_masks_dir, cam_id+'.png'), canvas)

    if verify:
        canvas = stitched_img.copy()
        for v in out['cells']:
            x0, y0, x1, y1 = list(map(int, v['box']))
            color = pattern_colors[v['pattern']]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1)
            r, c = v['pos']
            txt = f'({r}, {c})'
            if v['is_ref']:
                txt = 'REF: '+txt
            cv2.putText(canvas, txt, (x0 + 5, y0 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        dst_f = osp.join(verify, 'stithced_final_cells.jpg')
        imsave(dst_f, canvas)

        for cam_id in Ts:
            canvas = imread(osp.join(images_stitch, f'{cam_id}.jpg'))
            for tile_cell in tile_cells[cam_id]:
                color = pattern_colors[tile_cell['pattern']]
                pos = tile_cell['pos']
                corners = np.asarray(tile_cell['corners'], dtype='i4')
                x0, y0 = corners[0]
                cv2.polylines(canvas, [corners], True, color, 1)
                cv2.putText(canvas, f'{tuple(pos)}', (x0 + 5, y0 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            dst_f = osp.join(verify, f"inv_proj_cells_{cam_id}.jpg")
            imsave(dst_f, canvas)

if __name__ == "__main__":
    main()
