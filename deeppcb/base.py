import os
import os.path as osp
import cv2
from skimage import transform
from skimage.morphology import disk
import numpy as np
import numpy.linalg as npl
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def estimate_transform(src, dst, tp=None, **kws):
    if tp is None:
        tp = 'projective'

    src = np.asarray(src)
    dst = np.asarray(dst)
    return transform.estimate_transform(tp, src, dst, **kws)

def get_order(order):
    if order is None:
        order = 'linear'
    order = getattr(cv2, f'INTER_{order.upper()}')
    return order

sk_orders = {
    'nearest': 0,
    'linear': 1,
}

def transform_img(img, T_10, size, order=None, border_mode=None, border_value=None):
    if border_mode is None:
        border_mode = 'constant'

    if border_value is None:
        border_value = 0

    size = tuple(size)
    img = np.asarray(img)

    max_l = max(img.shape[:2])
    if max_l < 32767:
        order = get_order(order)

        border_mode = getattr(cv2, f'BORDER_{border_mode.upper()}')

        return cv2.warpPerspective(img, T_10, size, flags=order, borderMode=border_mode, borderValue=border_value)
    else:
        if order is not None:
            order = sk_orders[order]
        return transform.warp(img, npl.inv(T_10), order=order, mode=border_mode, cval=border_value, preserve_range=True).astype(img.dtype)

def pil_rescale(img, scale, order=None):
    w, h = img.size
    w1, h1 = int(w*scale), int(h*scale)
    resample = None
    if order:
        resample = getattr(Image, order.upper())
    return img.resize((w1, h1), resample=resample)

def rescale(img, scale, order=None):
    img = np.asarray(img)
    dt = img.dtype
    if dt == bool:
        img = img.astype('u1')
        order='nearest'

    order = get_order(order)
    h, w = img.shape[:2]
    w1, h1 = int(w*scale), int(h*scale)
    out = cv2.resize(img, (w1, h1), interpolation=order)
    if dt == bool:
        out = out.astype(dt)
    return out

def resize(img, size, order=None):
    assert img is not None
    img = np.asarray(img)
    dt = img.dtype
    if dt == bool:
        img = img.astype('u1')
        order = 'nearest'

    order = get_order(order)
    out = cv2.resize(img, size, interpolation=order)
    if dt == bool:
        out = out.astype(dt)
    return out

def imread(fname, to_gray=False):
    if not to_gray:
        return np.asarray(Image.open(fname))
    return cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

def imsave(fname, img, quality=98):
    if isinstance(img, Image.Image):
        img.save(fname, quality=quality)
        return

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(fname, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

def get_gray(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_bbox(box, shape):
    h, w = shape[:2]
    if box is None:
        return [[0, 0], [w, 0], [w, h], [0, h]]
    if not hasattr(box[0], '__iter__'):
        if len(box) == 2:
            x0, y0 = box
            x1, y1 = x0 + w, y0 + h
        else:
            x0, y0, x1, y1 = box

        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    return box

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def parse_name(name):
    name = file_name(name)
    parts = name.split('-')
    out = {
        'name': name
    }
    if ':' not in parts[0]:
        out['id'], parts = parts[0], parts[1:]

    for part in parts:
        assert ':' in part
        k, v = part.split(':')
        if k in ('pos', 'ppos'):
            v = list(map(int, v.split('_')))
        elif k in ('roi', 'proi'):
            v = list(map(float, v.split(',')))
        out[k] = v

    return out

DFLT_EDGE_KWS = {
    'low_th': 100,
    'high_th': 200,
}

def get_edge(img, **kws):
    kws = {**DFLT_EDGE_KWS, **kws}
    gray = get_gray(img)

    edge = cv2.Canny(gray, kws['low_th'], kws['high_th'])
    imsave("distmap_edge.jpg",edge,100);
    return edge

def scale_H(H, s):
    H = np.asarray(H)
    H[:2, 2] *= s
    H[2, :2] /= s
    return H

def binary_erosion(m, kernel):
    if not isinstance(kernel, np.ndarray):
        kernel = disk(kernel)
    return cv2.erode(m.astype('u1'), kernel).astype(bool)

def binary_dilation(m, kernel):
    if not isinstance(kernel, np.ndarray):
        kernel = disk(kernel)
    return cv2.dilate(m.astype('u1'), kernel).astype(bool)

def get_bbox_clockwise(bbox):
    (x0, y0), (x1, y1), (x2, y2) = bbox[:3]
    A = np.array([
        [x0, y0, 1],
        [x1, y1, 1],
        [x2, y2 ,1]
    ])
    return npl.det(A)

def get_bbox_H(shape0, bbox1, tform_tp='projective'):
    mh, mw = shape0
    cw = get_bbox_clockwise(bbox1)
    if cw > 0:
        pts0 = np.array([[0, 0], [mw, 0], [mw, mh], [0, mh]], dtype='f4')
    else:
        pts0 = np.array([[0, 0], [0, mh], [mw, mh], [mw, 0]], dtype='f4')

    assert (get_bbox_clockwise(pts0) > 0) == (cw > 0)

    bbox1 = np.asarray(bbox1, dtype='f4')
    if tform_tp in ('affine', 'similarity'):
        H_01 = cv2.getAffineTransform(bbox1[:3], pts0[:3])
        H_01 = np.vstack((H_01, [0,0,1]))
    else:
        assert tform_tp == 'projective'
        H_01 = cv2.getPerspectiveTransform(bbox1, pts0)

    return H_01
