import numpy as np
import random
import cv2
from easydict import EasyDict as edict
from collections import defaultdict
from time import time
from scipy.ndimage.morphology import distance_transform_edt
from skimage.filters import sobel
from skimage import measure
from skimage.segmentation import watershed, chan_vese
from skimage.morphology import remove_small_objects, remove_small_holes, binary_dilation, binary_erosion, disk
from pprint import pprint

from .base import imsave
from .segmentation import morphological_chan_vese
from .draw import draw_defects
from .defect import cluster_defects, build_groups

import pylab as plt
from matplotlib import cm
import json

def format_rprop(r, **kws):
    out = kws.copy()
    for k in ['area']:
        out[k] = getattr(r, k)

    out['centroid'] = r.centroid[::-1]
    y0, x0, y1, x1 = r.bbox
    out['bbox'] = [x0, y0, x1, y1]
    out['mask'] = r.image
    return out

def expand_rprop_mask(rp, gap, shape):
    imh, imw = shape
    yy0, xx0, yy1, xx1 = rp.bbox
    x0 = max(xx0 - gap, 0)
    y0 = max(yy0 - gap, 0)
    x1 = min(xx1 + gap, imw)
    y1 = min(yy1 + gap, imh)

    off_x = xx0 - x0
    off_y = yy0 - y0

    in_mask = rp.image
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    mask[off_y:off_y+in_mask.shape[0], off_x:off_x+in_mask.shape[1]] = in_mask
    return mask, [x0, y0, x1, y1]

def find_topk_points(img, mask, topk=1, reverse=False):
    ys, xs = np.where(mask)
    intensities = img[ys, xs]
    if reverse:
        idxs = np.argsort(intensities)[-topk:][::-1]
    else:
        idxs = np.argsort(intensities)[:topk]

    return list(zip(xs[idxs], ys[idxs]))

def segment_object_from_seeds(img, seeds, bg_hint, valid_mask, method, **kws):
    if method == 'watershed':
        xs, ys = zip(*seeds)
        marker = np.zeros(img.shape[:2], dtype='u1')
        marker[ys, xs] = 1
        marker[bg_hint] = 2

        edge = sobel(img)
        ws = watershed(edge, marker)
        fill = ws == 1
        if valid_mask is not None:
            fill &= valid_mask
        return fill

    elif method == 'floodfill':
        tot_fill = np.zeros(img.shape[:2], dtype=bool)
        lo_tol = kws.get('lo_tol', kws.get('tol', 5))
        up_tol = kws.get('up_tol', kws.get('tol', 5))
        for x, y in seeds:
            canvas = img.copy()
            cv2.floodFill(canvas, None, (x, y), 1, loDiff=lo_tol, upDiff=up_tol)
            tot_fill |= canvas == 1

        tot_fill &= ~bg_hint
        if valid_mask is not None:
            tot_fill &= valid_mask
        return tot_fill

    elif method == 'morphological_chan_vese':
        if not isinstance(seeds, np.ndarray):
            xs, ys = zip(*seeds)
            seeds = np.zeros(img.shape[:2], dtype=bool)
            seeds[ys, xs] = True

        fill = morphological_chan_vese(img, seeds, valid_mask, 10)
        return fill

def update_foreign_map(m, objs, val):
    for o in objs:
        x0, y0, x1, y1 = o['bbox']
        m[y0:y1, x0:x1][o['mask']] = val

def is_obj_overlap_mask(rp, m):
    y0, x0, y1, x1 = rp.bbox
    return m[y0:y1, x0:x1][rp.image].sum() > 0

def detect_holes(roi, wl_mask, imdic, cfg):
    I = imdic['I']
    imh, imw = I.shape
    clean = remove_small_holes(roi, cfg.max_area, connectivity=2)

    diff = clean ^ roi
    rprops = measure.regionprops(measure.label(diff))

    # need flood fill to expand inland_sea
    radius = cfg.surr_radius

    tgt_mask = np.zeros((imh, imw), dtype=bool)
    for rp in rprops:
        if is_obj_overlap_mask(rp, tgt_mask):
            continue

        yy0, xx0, yy1, xx1 = rp.bbox
        if xx0 < cfg['crop_border'] or xx1 > imw - cfg['crop_border']:
            continue

        rp_mask, (x0, y0, x1, y1) = expand_rprop_mask(rp, radius, (imh, imw))
        sub_img = I[y0:y1, x0:x1]
        sub_valid = wl_mask[y0:y1, x0:x1]

        ys, xs = np.where(rp_mask)
        seeds = list(zip(xs, ys))
        meth = cfg.method
        kws = {}
        if meth == 'floodfill':
            kws['tol'] = cfg.floodfill_tol

        fill_mask = segment_object_from_seeds(sub_img, seeds, sub_valid, None, meth, **kws)
        tgt_mask[y0:y1, x0:x1] |= fill_mask

    clean[tgt_mask] = True
    return clean, tgt_mask

def filter_holes(rprops, obj_mask):
    shape = obj_mask.shape
    kernel = disk(1)
    cand_rprops = []
    for rp in rprops:
        rp_mask, (x0, y0, x1, y1) = expand_rprop_mask(rp, 1, shape)
        ext_rp_mask = binary_dilation(rp_mask, kernel)
        if (ext_rp_mask & obj_mask[y0:y1, x0:x1]).any():
            obj_mask[y0:y1, x0:x1] |= rp_mask
        else:
            cand_rprops.append(rp)

    rprops = measure.regionprops(measure.label(obj_mask))
    return cand_rprops, rprops

def filter_small_area(rprops, foreign_map, max_area):
    cand_rprops = []
    out = []
    for rp in rprops:
        if is_obj_overlap_mask(rp, foreign_map):
            continue
        if rp.area <= max_area:
            out.append(rp)
        else:
            cand_rprops.append(rp)

    return cand_rprops, out

def filter_similar_color(rprops, imdic, roi, wl_mask, foreign_map, cfg):
    I = imdic['I']
    imh, imw = I.shape[:2]
    out = []
    cand_props = []

    radius = cfg.surr_radius

    kernel = disk(3)

    for idx, rp in enumerate(rprops):
        if is_obj_overlap_mask(rp, foreign_map):
            continue

        rp_min = rp.min_intensity[0]
        rp_max = rp.max_intensity[0]
        if rp_max - rp_min > cfg.max_intensity_range:
            cand_props.append(rp)
            continue

        yy0, xx0, yy1, xx1 = rp.bbox
        cy, cx = rp.centroid

        rp_mask, (x0, y0, x1, y1) = expand_rprop_mask(rp, radius, (imh, imw))
        sub_img = I[y0:y1, x0:x1]
        valid_mask = roi[y0:y1, x0:x1]

        meth = cfg.method
        kws = {}
        if meth == 'floodfill':
            kws['tol'] = cfg.floodfill_tol

        bg_hint = wl_mask[y0:y1, x0:x1]
        bg_hint = binary_erosion(bg_hint, kernel)

        seed_top_k = 1
        seeds = find_topk_points(sub_img, rp_mask, topk=seed_top_k, reverse=(cfg.seed_mode=='light'))

        fill_mask = segment_object_from_seeds(sub_img, seeds, bg_hint, valid_mask, meth, **kws)

        expand_area = ~bg_hint & valid_mask
        expand_mask = (fill_mask | rp_mask) & ~rp_mask

        if expand_mask.sum() > cfg.fill_max_factor * expand_area.sum():
            out.append(rp)
        else:
            cand_props.append(rp)

    return cand_props, out

def calc_cr_cb_dist(a, b):
    x0, y0 = map(int, a)
    x1, y1 = map(int, b)
    return ((x0 - x1)**2 + (y0 - y1)**2)**0.5

def filter_different_color(rprops, imdic, foreign_map, cfg):
    lab = imdic['lab']
    imh, imw = lab.shape[:2]
    tgt_rprops = []
    tgt_mask = np.zeros((imh, imw), dtype=bool)
    cand_props = []
    surr_kernel = disk(cfg.surr_radius)

    for rp in rprops:
        if is_obj_overlap_mask(rp, foreign_map):
            continue

        yy0, xx0, yy1, xx1 = rp.bbox
        rp_min = rp.min_intensity[0]
        rp_max = rp.max_intensity[0]
        rp_rb_min = rp.min_intensity[1:]
        rp_rb_max = rp.max_intensity[1:]

        if rp_max - rp_min > cfg.min_intensity_var and calc_cr_cb_dist(rp_rb_min, rp_rb_max) > cfg.min_rb_var:
            tgt_mask[yy0:yy1, xx0:xx1] |= rp.image
            tgt_rprops.append(rp)
            continue

        rp_mask, (x0, y0, x1, y1) = expand_rprop_mask(rp, cfg.surr_radius, (imh, imw))

        surr_mask = binary_dilation(rp_mask, surr_kernel)
        surr_mask = surr_mask ^ rp_mask
        surr_pix_mean = np.percentile(lab[y0:y1, x0:x1][surr_mask], 50, axis=0)

        if cfg.seed_mode == 'dark_gray':
            if surr_pix_mean[0] - rp_min > cfg.min_intensity_var and calc_cr_cb_dist(surr_pix_mean[1:], rp_rb_min) > cfg.min_rb_var: # judge intensity and gray
                tgt_mask[yy0:yy1, xx0:xx1] |= rp.image
                tgt_rprops.append(rp)
                continue

        elif cfg.seed_mode == 'dark':
            if surr_pix_mean[0] - rp_min > cfg.min_intensity_var: # judge intensity
                tgt_mask[yy0:yy1, xx0:xx1] |= rp.image
                tgt_rprops.append(rp)
                continue

        elif cfg.seed_mode == 'light':
            if rp_max - surr_pix_mean[0] > cfg.min_intensity_var:
                tgt_mask[yy0:yy1, xx0:xx1] |= rp.image
                tgt_rprops.append(rp)
                continue

        cand_props.append(rp)

    return cand_props, tgt_rprops

def get_foreign_img(imdic, with_lab=False, with_grad=False, with_gray=False):
    if not isinstance(imdic, dict):
        imdic = {
            'img': imdic
        }

    if with_grad:
        with_gray = True

    if with_lab and 'lab' not in imdic:
        img = imdic['img']
        imdic['lab'] = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    if with_gray and 'I' not in imdic:
        if 'lab' in imdic:
            imdic['I'] = imdic['lab'][..., 0]
        else:
            imdic['I'] = cv2.cvtColor(imdic['img'], cv2.COLOR_RGB2GRAY)

    if with_grad and 'grad' not in imdic:
        I = imdic['I']
        gx = cv2.Sobel(I, cv2.CV_8U, 0, 1, 3)
        gy = cv2.Sobel(I, cv2.CV_8U, 1, 0, 3)
        imdic['grad'] = (gx**2 + gy**2)**0.5

    if 'shape' not in imdic:
        for k in ['img', 'I', 'lab', 'grad']:
            if k in imdic:
                imdic['shape'] = imdic[k].shape[:2]
                break

    return imdic

def update_output(tp, out, objs, cfg):
    foreign_map = out['segmap']
    objs = [format_rprop(i, type=tp) for i in objs]
    update_foreign_map(foreign_map, objs, cfg[tp].label)
    out['objects'] += objs
    return out

def detect_foreigns_on_copper(continent, land, imdic, mask, out, cfg):
    imh, imw = continent.shape

    ### detect inland sea
    cfg.inland_sea.setdefault('crop_border', cfg.crop_border)
    continent, init_inland_sea = detect_holes(continent, land, imdic, cfg.inland_sea)

    ### clear coastline area, make basic structures
    continent = binary_erosion(continent, disk(cfg.copper_margin))
    coastline = continent ^ binary_erosion(continent, disk(1))
    land[~continent] = False # clear outside land

    if mask is not None:
        init_inland_sea &= mask
        continent &= mask
        coastline &= mask
        land &= mask

    diff = continent ^ land # calculate difference
    diff[init_inland_sea] = False

    lbl = measure.label(diff)
    rest = measure.regionprops(lbl, intensity_image=imdic['lab'])

    foreign_map = out['segmap']

    rest, objs = filter_holes(rest, init_inland_sea)
    update_output('inland_sea', out, objs, cfg)

    rest, objs = filter_small_area(rest, foreign_map, cfg.small_pond.max_area)
    update_output('small_pond', out, objs, cfg)

    rest, objs = filter_similar_color(rest, imdic, continent, land, foreign_map, cfg.shallow_water)
    update_output('shallow_water', out, objs, cfg)

    rest, objs = filter_different_color(rest, imdic, foreign_map, cfg.deep_water)
    update_output('deep_water', out, objs, cfg)
    update_output('copper_cand', out, rest, cfg)

    return edict({
        'imdic': imdic,
        'continent': continent,
        'coastline': coastline,
        'land': land,
        'diff': diff,
    })

def detect_foreigns_on_bg(ocean, sea, imdic, mask, out, cfg):
    imh, imw = ocean.shape

    ### clear coastline area, make basic structures
    ### erosion ocean first for thin structure
    ocean = binary_erosion(ocean, disk(cfg.bg_margin))
    coastline = ocean ^ binary_erosion(ocean, disk(1))
    sea[~ocean] = False # clear outside land

    cfg.insea_land.setdefault('crop_border', cfg.crop_border)
    ocean, init_insea_land = detect_holes(ocean, sea, imdic, cfg.insea_land)

    if mask is not None:
        init_insea_land &= mask
        ocean &= mask
        coastline &= mask
        sea &= mask

    diff = ocean ^ sea # calculate difference
    diff[init_insea_land] = False

    lbl = measure.label(diff)
    rest = measure.regionprops(lbl, intensity_image=imdic['lab'])

    foreign_map = out['segmap']

    rest, objs = filter_holes(rest, init_insea_land)
    update_output('insea_land', out, objs, cfg)

    rest, objs = filter_small_area(rest, foreign_map, cfg.small_reef.max_area)
    update_output('small_reef', out, objs, cfg)

    rest, objs = filter_similar_color(rest, imdic, ocean, sea, foreign_map, cfg.shallow_sand)
    update_output('shallow_sand', out, objs, cfg)

    rest, objs = filter_different_color(rest, imdic, foreign_map, cfg.high_sand)
    update_output('high_sand', out, objs, cfg)

    update_output('bg_cand', out, rest, cfg)

    return edict({
        'ocean': ocean,
        'sea': sea,
        'coastline': coastline,
        'diff': diff,
    })

def detect_foreigns(segmap, imdic, classes, cfg, mask=None):
    print('detect foreigns')
    imdic = get_foreign_img(
        imdic,
        with_gray=True,
        with_lab=True,
        with_grad=False,
    )

    foreign_map = np.zeros_like(segmap, dtype='u1')

    objects = []
    out = {
        'detector_config': cfg,
        'segmap': foreign_map,
        'objects': objects,
    }

    continent = segmap & classes.copper.label > 0
    land = segmap & classes.wl_copper.label > 0
    copper_ctx = detect_foreigns_on_copper(continent, land, imdic, mask, out, cfg)
    print('copper')
    foreign_map[copper_ctx.continent] |= classes.copper.label

    ocean = segmap & classes.bg.label > 0
    sea = segmap & classes.wl_bg.label > 0
    bg_ctx = detect_foreigns_on_bg(ocean, sea, imdic, mask, out, cfg)
    print('bg')
    foreign_map[bg_ctx.ocean] |= classes.bg.label

    objs = {}
    for idx, o in enumerate(out['objects']):
        c = cfg[o['type']]
        if o['type'].startswith('small_'):
            continue

        o.update({
            'id': idx,
            'level': c.level,
            'located': c.located,
        })
        objs[idx] = o

    if len(objs) > 0:
        print('start cluster')
        grp_labels = cluster_defects(objs, segmap.shape, **cfg.cluster)
        print('end cluster')
        for (k, o), g in zip(objs.items(), grp_labels):
            if g >= 0:
                o['group'] = g

    ### Add recall logic for massive light foreigns here

    out.update({
        'objects': objs,
        'groups': build_groups(objs),
    })

    return out, edict({
        'copper': copper_ctx,
        'bg': bg_ctx,
    })
