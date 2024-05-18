import numpy as np
import numpy.linalg as npl
import cv2
from .base import get_gray, get_bbox, get_bbox_H
import pylab as plt

def ocv_find_ecc(moving_img, fixed_img, init_bbox, tform_tp, max_iters=50, **kws):
    # 0: moving inside
    # 1: fixed like extended moving
    # 2: fixed
    moving_img = get_gray(moving_img)
    fixed_img = get_gray(fixed_img)

    init_bbox = np.asarray(get_bbox(init_bbox, moving_img.shape))
    H_01 = get_bbox_H(moving_img.shape, init_bbox, tform_tp)

    if tform_tp == 'affine':
        warp_mode = cv2.MOTION_AFFINE
    elif tform_tp == 'projective':
        warp_mode = cv2.MOTION_HOMOGRAPHY
    else:
        raise

    HH = H_01
    if tform_tp == 'affine':
        HH = H_01[:2]
    HH = np.asarray(HH).astype('f4')
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters,  1e-6)
    (rms, HH) = cv2.findTransformECC(fixed_img, moving_img, HH, warp_mode, crit)
    if HH.shape[0] == 2:
        HH = np.vstack((H, [0, 0, 1]))

    H_02 = HH
    H_20 = npl.inv(H_02)
    if H_01.shape[0] == 2:
        H_01 = np.vstack((H_01, [0, 0, 1]))

    H_21 = H_20.dot(H_01)
    return {
        'H_20': H_20,
        'H_21': H_21,
        'err': rms,
    }
