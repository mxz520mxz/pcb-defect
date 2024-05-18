import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
from skimage import measure
from .base import transform_img, estimate_transform, Image
from .align_edge import align_edge

def cut_cells(img, cells, pat_sizes, ref_width, order=None, border_value=None):
    if border_value is None:
        border_value = 0
    if order is None:
        order = 'nearest'

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if img.mode == 'RGB':
        img = img.convert('RGBA')

    imw, imh = img.size
    scale = imw / ref_width

    out = []
    for cell in cells:
        corners = np.array(cell['corners'])
        corners *= scale

        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        roi = x0, y0, x1, y1

        if x1 <= 0 or imw <= x0 or y1 <= 0 or imh <= y0:
            continue

        crop_img = np.asarray(img.crop(roi))
        if img.mode == 'RGBA':
            crop_mask = crop_img[..., 3] > 128
            crop_img = crop_img[..., :3]
        elif img.mode == 'L':
            crop_mask = crop_img > 0

        crop_img[~crop_mask] = border_value

        src_corners = corners.copy()
        src_corners[:, 0] -= x0
        src_corners[:, 1] -= y0

        pat = cell["pattern"]
        pw, ph = pat_sizes[pat]
        pw *= scale
        ph *= scale
        x0, y0, x1, y1 = 0, 0, pw, ph

        tgt_corners = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
        tgt_w, tgt_h = int(pw), int(ph)

        tform = estimate_transform(src_corners, tgt_corners)
        tgt_img = transform_img(crop_img, tform.params, (tgt_w, tgt_h), order=order, border_value=border_value)

        # tgt_img = warp(crop_img, tform.inverse, output_shape=(tgt_h, tgt_w), order=order)
        # tgt_img = img_as_ubyte(tgt_img)

        out.append(edict({
            'img': tgt_img,
            'pos': cell.pos,
            'pattern': pat,
            'T_10': tform.params,
            'size': [tgt_w, tgt_h],
            'roi': roi,
        }))

    return out

def apply_cell_cuttings(img, tforms, order=None, border_value=None):
    if border_value is None:
        border_value = 0
    if order is None:
        order = 'nearest'

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if img.mode == 'RGB':
        img = img.convert('RGBA')

    imw, imh = img.size

    out = []
    for tform in tforms:
        x0, y0, x1, y1 = roi = tform['roi']
        crop_img = np.asarray(img.crop(roi))
        if img.mode == 'RGBA':
            crop_mask = crop_img[..., 3] > 128
            crop_img = crop_img[..., :3]
        elif img.mode == 'L':
            crop_mask = crop_img > 0
        crop_img[~crop_mask] = border_value
        tgt_img = transform_img(crop_img, tform['T_10'], tform['size'], order=order, border_value=border_value)
        out.append(tgt_img)

    return out

def filter_cell_contours(conts, blank_mask, min_len):
    out = []
    for c in conts:
        if len(c) < min_len:
            continue

        c = c.astype('i4')
        cut_sel = blank_mask[c[:,1], c[:,0]] > 0

        tp = 'continuous'
        if cut_sel.any():
            tp = 'fragmental'
            c = c[~cut_sel]

        out.append({
            'type': tp,
            'contour': c,
        })
    return out

def extract_cell_contours(img, blank_mask, min_len):
    conts = measure.find_contours(img)
    conts = [i[:,::-1] for i in conts]
    conts = filter_cell_contours(conts, blank_mask, min_len=min_len)
    return {
        'shape': img.shape,
        'contours': conts,
    }

def align_cell(conts, tpl_distmap, **kws):
    points = []
    for c in conts['contours']:
        points += c['contour'].tolist()
    points = np.array(points)

    src_shape = conts['shape']
    ret = align_edge(
        {
            'shape': src_shape,
            'points': points,
        },
        {
            'distmap': tpl_distmap,
        },
        **kws,
    )
    return {
        **ret,
        'tgt_shape': src_shape,
        'tpl_shape': tpl_distmap.shape[:2],
    }
