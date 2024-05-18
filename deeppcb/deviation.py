import cv2
import numpy as np
import numpy.linalg as npl
from easydict import EasyDict as edict
from sklearn.cluster import DBSCAN
from skimage.morphology import disk
from skimage import measure
from skimage.transform import ProjectiveTransform
from tqdm import tqdm
from .base import binary_erosion
from .align_edge import align_edge
from .defect import cluster_defects, build_groups
import pylab as plt
import os
import os.path as osp

def unique_contour(c):
    prev = None
    out = []
    for i in c:
        ti = tuple(i)
        if ti != prev:
            prev = ti
            out.append(i)
    return np.asarray(out)

def filter_contours(conts, crop_mask):
    out = []
    for c in conts:
        c = unique_contour(c)
        sel = crop_mask[c[:,1], c[:,0]] > 0
        if sel.sum():
            out.append(c[sel])

    return out

def check_is_closed(c):
    n = len(c)
    return (c[n-1] == c[0]).all()

def split_parts(c, breakpoints):
    if not len(breakpoints):
        return [c]

    if len(breakpoints) == 1:
        bp = breakpoints[0]
        c = list(c[bp+1:]) + list(c[0:bp+1])
        return [c]

    out = []
    n = len(c)
    off = breakpoints[-1] + 1 - n
    for bp in breakpoints:
        if off < 0:
            part = list(c[off:]) + list(c[:bp+1])
        else:
            part = c[off:bp+1]

        out.append(part)
        off = bp + 1

    return out

def split_contours(conts):
    out = []
    for c in conts:
        n = len(c)
        is_closed = check_is_closed(c)

        nn_dists = npl.norm(c[:n-1] - c[1:], axis=1)
        if not is_closed:
            last_d = npl.norm(c[n-1] - c[0])
            nn_dists = np.concatenate((nn_dists, [last_d]))

        breakpoints = np.where(nn_dists > 1.5)[0]
        parts = split_parts(c, breakpoints)
        out += parts
    return out

def extract_contours(img, valid_mask, inside_mask=None):
    crop_mask = valid_mask
    conts = measure.find_contours(img)
    conts = [i[:,::-1].astype('i4') for i in conts]
    conts = filter_contours(conts, crop_mask)
    conts = split_contours(conts)
    conts = [np.asarray(c) for c in conts]
    if inside_mask is not None:
        conts = [c for c in conts if any(inside_mask[c[:,1], c[:,0]])]
    return dict(enumerate(conts))

def flood_fill_segments(cont, dists, seeds, connect_len=2):
    out = []
    n = len(cont)
    visited = set()
    for seed in seeds:
        if seed in visited:
            continue

        visited.add(seed)

        patience = 0
        l = [seed]
        prev_dist = float(dists[seed])
        for j in range(seed + 1, n):
            if j in visited:
                continue
            visited.add(j)

            d = float(dists[j])
            if j in seeds:
                prev_dist = d
                l.append(j)
                continue

            if d < prev_dist:
                prev_dist = d
                patience = 0 # refresh patience
                l.append(j)
                continue

            patience += 1
            if patience <= connect_len:
                l.append(j)
                continue
            else:
                break

        prev_dist = float(dists[seed])
        for j in range(seed-1, -1, -1):
            if j in visited:
                continue
            visited.add(j)

            d = float(dists[j])
            if j in seeds:
                prev_dist = d
                l.insert(0, j)
                continue

            if d < prev_dist:
                prev_dist = d
                patience = 0
                l.insert(0, j)
                continue

            patience += 1
            if patience <= connect_len:
                l.insert(0, j)
                continue
            else:
                break

        out.append(l)

    return out


def filter_conts_by_dist(conts, tpl_copper, tpl_distmap, far_dist_th, far_ratio, near_dist_th, connect_len, obj_level='black', inside_mask=None):
    has_inside_mask = inside_mask is not None
    if has_inside_mask:
        outside_mask = ~inside_mask
    cand_conts = {}
    tgt_conts = {}
    # for k, c in tqdm(conts.items()):
    for k, c in conts.items():
        is_closed = check_is_closed(c)
        dists = tpl_distmap[c[:,1], c[:,0]]
        if dists.max() <= near_dist_th:
            continue
        sel = dists > far_dist_th
        if sel.sum() == 0:
            cand_conts[k] = c
            continue

        if sel.sum() >=  far_ratio * len(c): # mainly external
            tgt_conts[k, 0] = c
            continue

        seeds = np.where(sel)[0]
        black_segments = flood_fill_segments(c, dists, seeds, connect_len=connect_len)
        rest_sel = np.ones(len(c), dtype=bool)
        for pi, psel in enumerate(black_segments):
            p = c[psel]
            rest_sel[psel] = False
            if has_inside_mask and outside_mask[p[:,1], p[:,0]].all():
                continue
            center = tuple(p.mean(axis=0).astype('i4'))
            tgt_conts[k, center] = p

        rest_c = c[rest_sel]
        if not len(rest_c):
            continue

        rest_dists = dists[rest_sel]
        if rest_dists.max() <= near_dist_th:
            continue

        cand_conts[k] = rest_c

    return cand_conts, tgt_conts

def align_contour(c, bbox, tpl_distmap, verbose=False):
    h, w = tpl_distmap.shape
    x0, y0, x1, y1 = bbox
    crop_distmap = tpl_distmap[y0:y1, x0:x1]
    ch, cw = crop_distmap.shape[:2]

    c_init = c - [x0, y0]
    ret = align_edge({
        'shape': [ch, cw],
        'points': c_init,
    }, {'distmap': crop_distmap}, tform_tp='affine', verbose=verbose, dev='cpu')

    H = ret['H_20']
    H = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ]).dot(H).dot(
        np.array([
            [1, 0, -x0],
            [0, 1, -y0],
            [0, 0, 1],
        ])
    )
    return H

def transform_contour(H, c, shape):
    h, w = shape
    c_final = ProjectiveTransform(H)(c).astype('i4')
    c_final = np.hstack((c_final[:,[0]].clip(min=0, max=w-1), c_final[:,[1]].clip(min=0, max=h-1)))
    c_final = unique_contour(c_final)
    return c_final

# def align_contours(conts, tpl_distmap, margin=20):
#     out = {}
#     h, w = tpl_distmap.shape[:2]
#     for k, c in tqdm(conts.items()):
#         # assert no much duplicated
#         assert abs(len(c) - len(set(tuple(i) for i in c))) < 2, '{} != {}'.format(len(c), len(set(tuple(i) for i in c)))
#         x0, y0 = c.min(axis=0)
#         x1, y1 = c.max(axis=0)

#         x0 = max(x0 - margin, 0)
#         y0 = max(y0 - margin, 0)
#         x1 = min(x1 + margin, w-1)
#         y1 = min(y1 + margin, h-1)

#         H = align_contour(c, (x0, y0, x1, y1), tpl_distmap)
#         c_final = transform_contour(H, c, tpl_distmap.shape)
#         out[k] = c_final

#     return out

def align_contours(name, inside_mask, segmap, conts, tpl_distmap, margin=20):
    repeat_conts = 0
    out = {}
    h, w = tpl_distmap.shape[:2]

    for k, c in conts.items():
    # for k, c in tqdm(conts.items()):
        # assert no much duplicated
        # assert abs(len(c) - len(set(tuple(i) for i in c))) < 2, '{} != {}'.format(len(c), len(set(tuple(i) for i in c)))
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)

        x0 = max(x0 - margin, 0)
        y0 = max(y0 - margin, 0)
        x1 = min(x1 + margin, w-1)
        y1 = min(y1 + margin, h-1)

        H = align_contour(c, (x0, y0, x1, y1), tpl_distmap)
        c_final = transform_contour(H, c, tpl_distmap.shape)
        out[k] = c_final

        #if abs(len(c) - len(set(tuple(i) for i in c))) >= 2:
        #    repeat_conts += 1
        #     print('k,c',k,len(c),c)
        #     fig, ax = plt.subplots()
        #     show_segmap = True
        #     ax.imshow(inside_mask)
        #     if show_segmap:
        #         ax.imshow(segmap)
        #         ax.plot(c[:,0], c[:,1], color='k')

    #print('repeat_conts',repeat_conts)
    #check_dir = 'check_repeat_conts'
    #os.makedirs(check_dir, exist_ok=True)
    #if repeat_conts > 0:
    #    # plot
    #    # plt.savefig(f"{name}_check_repeat_conts.png")
    #    # plt.show()

        # just check name
    #    cv2.imwrite(osp.join(check_dir, f"{name}.jpg"), segmap) 

    return out

def refine_objects(objs, conts, tpl_distmap, margin, dist_th, connect_len):
    h, w = tpl_distmap.shape[:2]
    tgt_conts = {}
    # for (k, center), o in tqdm(objs.items()):
    for (k, center), o in objs.items():
        cont = conts[k]

        x0, y0 = o.min(axis=0)
        x1, y1 = o.max(axis=0)

        x0 = max(x0 - margin, 0)
        y0 = max(y0 - margin, 0)
        x1 = min(x1 + margin, w-1)
        y1 = min(y1 + margin, h-1)

        sel = (x0 <= cont[:, 0]) & (cont[:, 0] <= x1) & (cont[:, 1] >= y0) & (cont[:, 1] <= y1)
        sel_cont = cont[sel]

        H = align_contour(sel_cont, (x0, y0, x1, y1), tpl_distmap, verbose=False)
        aligned_o = transform_contour(H, o, tpl_distmap.shape)
        dists = tpl_distmap[aligned_o[:,1], aligned_o[:,0]]
        sel = dists > dist_th
        if sel.sum() == 0:
            continue

        seeds = np.where(sel)[0]
        black_segments = flood_fill_segments(aligned_o, dists, seeds, connect_len=connect_len)
        for pi, psel in enumerate(black_segments):
            p = aligned_o[psel]
            center = tuple(p.mean(axis=0).astype('i4'))
            tgt_conts[k, center] = p

    return tgt_conts

def detect_deviations(name, segmap, tpl_segmap, tpl_distmap, C, mask=None):
    cfg = C.deviation

    h, w = segmap.shape[:2]
    valid_mask = segmap != C.classes.blank.label
    if mask is not None:
        valid_mask &= mask

    valid_mask = binary_erosion(valid_mask, 1)
    inside_mask = binary_erosion(valid_mask, cfg.border_gap)
    copper = segmap == C.classes.copper.label
    tpl_copper = tpl_segmap == C.classes.copper.label
    conts = extract_contours(copper, valid_mask, inside_mask=inside_mask)

    # fig, ax = plt.subplots()
    # ax.imshow(inside_mask)
    # for k, c in conts.items():
    #     ax.plot(c[:,0], c[:,1])
    # plt.show()

    print ("extracted conts", len(conts))
    obj_conts = {}
    conts, objs = filter_conts_by_dist(
        conts,
        tpl_copper,
        tpl_distmap,
        far_dist_th=cfg.coarse_far_dist_th,
        far_ratio=cfg.coarse_far_ratio,
        near_dist_th=cfg.coarse_near_dist_th,
        connect_len=cfg.connect_len,
        obj_level='black',
        inside_mask=inside_mask,
    )

    obj_conts.update(objs)

    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # show_segmap = True
    # ax.imshow(inside_mask)
    # if show_segmap:
    #     ax.imshow(segmap)
    #     for k, c in conts.items():
    #         ax.plot(c[:,0], c[:,1], color='k')

    # for k, p in objs.items():
    #     ax.plot(p[:,0], p[:,1], color='red')
    # plt.show()

    print ("filter_by_dist", len(conts))
    aligned_conts = align_contours(name, inside_mask, segmap, conts, tpl_distmap, margin=cfg.align_contour_margin)
    rest_conts, gray_objs = filter_conts_by_dist(
        aligned_conts,
        tpl_copper,
        tpl_distmap,
        far_dist_th=cfg.strict_dist_th,
        far_ratio = cfg.strict_ratio,
        near_dist_th=cfg.strict_dist_th,
        connect_len=cfg.connect_len,
        inside_mask=inside_mask,
    )

    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # show_segmap = True
    # ax.imshow(inside_mask)
    # if show_segmap:
    #     ax.imshow(segmap)
    #     for k, c in aligned_conts.items():
    #         ax.plot(c[:,0], c[:,1], color='k')

    # print ("gray_objs", len(gray_objs))
    # for k, p in gray_objs.items():
    #     ax.plot(p[:,0], p[:,1], color='red')
    # plt.show()

    if gray_objs:
        objs = refine_objects(
            gray_objs, aligned_conts, tpl_distmap,
            margin=cfg.refine_margin,
            dist_th=cfg.refine_dist_th,
            connect_len=cfg.connect_len,
        )
        obj_conts.update(objs)

    #print ("align and filter_by_dist", len(rest_conts))

    deviation_map = np.zeros((h, w), dtype='u1')

    #print ("detected objects", len(obj_conts))

    objects = {}
    for key, p in obj_conts.items():
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)

        if tpl_copper[p[:,1], p[:,0]].any():
            tp = 'concave'
        else:
            tp = 'convex'

        o = {
            'id': key,
            'type': tp,
            'level': 'black',
            'contour': p,
            'centroid': p.mean(axis=0),
            'bbox': [x0, y0, x1, y1],
            'area': len(p),
        }

        deviation_map[p[:, 1], p[:, 0]] = cfg[tp].label

        objects[key] = o

    groups = {}
    if len(objects):
        print('deviation start cluster')
        grp_labels = cluster_defects(objects, segmap.shape, **cfg.cluster)
        print('deviation end cluster')
        for o, g in zip(objects.values(), grp_labels):
            if g >= 0:
                o['group'] = str(g)
        groups = build_groups(objects)

    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # show_segmap = True
    # ax.imshow(inside_mask)
    # if show_segmap:
    #     ax.imshow(segmap)
    #     for k, c in rest_conts.items():
    #         ax.plot(c[:,0], c[:,1], color='k')

    # for k, o in objects.items():
    #     p = o['contour']
    #     ax.plot(p[:,0], p[:,1])
    #     ax.scatter(p[:,0], p[:,1], color='red')

    #     # cur_c = aligned_conts[k[0]]
    #     # ax.plot(cur_c[:,0], cur_c[:,1], color='k')

    # plt.show()
    return {
        'detector_config': cfg,
        'objects': objects,
        'groups': groups,
        'segmap': deviation_map,
    }, edict({
    })
