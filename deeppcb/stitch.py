import os
import os.path as osp
import numpy as np
import numpy.linalg as npl
from collections import defaultdict
from skimage.transform import ProjectiveTransform
from easydict import EasyDict as edict
from .base import transform_img

def stitch_images(imgs, Ts, o_size, order=None, border_value=None, fusion='avg'):
    o_w, o_h, o_c = o_size
    if o_c == 1:
        canvas = np.zeros((o_h, o_w), dtype='f4')
    else:
        canvas = np.zeros((o_h, o_w, o_c), dtype='f4')

    canvas_mask = np.zeros((o_h, o_w), dtype='i4')

    out = {
        'parts': []
    }

    for k, T_tgt_i in Ts.items():
        T_tgt_i = np.asarray(T_tgt_i)
        img = np.asarray(imgs[k])

        trans_img = transform_img(img, T_tgt_i, (o_w, o_h), order=order, border_mode='REPLICATE', border_value=border_value)

        mask = np.ones(img.shape[:2], dtype='u1')
        trans_mask = transform_img(mask, T_tgt_i, (o_w, o_h), order='nearest')
        trans_mask = trans_mask > 0
        trans_img[~trans_mask] = 0

        if fusion == 'avg':
            canvas[trans_mask] += trans_img[trans_mask]
            canvas_mask += trans_mask.astype('i4')
        elif fusion == 'overlap':
            canvas[trans_mask] = trans_img[trans_mask]

        out['parts'].append({
            'name': k,
            'T': T_tgt_i,
            'img': trans_img,
            'mask': trans_mask,
        })

    if fusion == 'avg':
        m = canvas_mask.copy()
        m[m == 0] = 1
        if o_c == 3:
            m = m[...,None]

        canvas = canvas / m

    out.update({
        'img': canvas.astype('u1'),
        'acc_mask': canvas_mask.astype('u1'),
    })
    return edict(out)

def inverse_project_cells(cells, Ts, width, overlap):
    out = defaultdict(list)
    for tile_id, T in Ts.items():
        invT = ProjectiveTransform(npl.inv(T))
        for cell in cells:
            pos = cell['pos']
            box = cell['box']
            if isinstance(box, np.ndarray):
                box = box.tolist()

            x0, y0, x1, y1 = box
            corners = invT([[x0,y0], [x0, y1], [x1, y1], [x1, y0]])
            xs = corners[:,0]
            x0 = xs.min()
            x1 = xs.max()
            if x1 < overlap or x0 >= width - overlap:
                continue

            out[tile_id].append({
                'pos': pos,
                'pattern': cell['pattern'],
                'corners': corners.tolist(),
            })
    return out
