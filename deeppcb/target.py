import imp
import os
import os.path as osp
import numpy as np
import numpy.random as npr
import cv2
from easydict import EasyDict as edict
import networkx as nx
import json
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from skimage.morphology import disk
from skimage.measure import find_contours
from .base import get_gray, Image, imread, rescale, get_edge, imsave, transform_img, binary_erosion, binary_dilation
from .align_edge import align_edge
from .draw import draw_match_image
from .constants import LOW_INTENSITY
import pylab as plt
from sklearn.decomposition import PCA
from loguru import logger
from pathlib import Path
# from lightglue import LightGlue, SuperPoint, DISK
# from lightglue.utils import load_image, rbd
# from lightglue import viz2d
import torch

import time

# task_graph = nx.DiGraph()
# task_graph.add_edges_from([
#     ('start', 'crop'),
#     ('crop', 'filter'),
#     ('filter', 'align_camera'),
# ])

# def get_pipeline(stage):
#     return nx.shortest_path(task_graph, 'start', stage)[1:][::-1]

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def read_list(lst, cam_mapping, return_dict=False):
    tsks = []
    for l in open(lst):
        l = l.strip()
        if l[0] == '#':
            continue

        img_name, board, cam = l.split()
        cam_key = cam_mapping[cam]
        face, cam_id = cam_key[0], cam_key[1:]
        tsks.append((img_name, board, face, cam_id))

    if return_dict:
        out = {}
        for img_name, tpl_name, face, cam_id in tsks:
            name = file_name(img_name)
            out[name] = {
                'name': name,
                'board': tpl_name,
                'face': face,
                'cam_id': cam_id,
            }
        return edict(out)
    else:
        return tsks

def process_crop(img, tpl_center, tpl_size, scale):
    print(type(img),img.size)
    imw, imh = img.size
    tpl_cx, tpl_cy = tpl_center
    tpl_w, tpl_h = tpl_size

    gray = get_gray(img)
    if scale < 1:
        gray = rescale(gray, scale)

    edge = get_edge(gray)
    # assert edge.shape == tpl_edge.shape
    ys, xs = np.where(edge)
    cx, cy = xs.mean(), ys.mean()
    if scale < 1:
        cx /= scale
        cy /= scale

    dx = cx - tpl_cx
    dy = cy - tpl_cy

    img = img.crop((dx, dy, dx + tpl_w, dy + tpl_h))
    # # for target gmm seg
    # v_gap = 10
    # img = img.crop((dx, dy-v_gap, dx + tpl_w, dy + tpl_h+v_gap))
    # img = np.asarray(img)
    # cfg = C.target.preproc.bilateral_filter
    # img = cv2.bilateralFilter(img, cfg.d, cfg.sigma_color, cfg.sigma_space)
    return img

def process_filter(img, d, sigma_color, sigma_space=None):
    if not sigma_space:
        sigma_space = d

    img = np.asarray(img)

    # yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # clahe = cv2.createCLAHE(2)
    # yuv_img[..., 0] = clahe.apply(yuv_img[..., 0])
    # img = cv2.cvtColor(yuv_img, cv2.COLOR_YCrCb2RGB)

    img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    return img

def process_align_camera(img, tpl_distmap, init_bbox, **kws):
    src_img = get_gray(img)
    tpl_w = tpl_distmap.shape[1]

    x0, y0, x1, y1 = init_bbox
    moving_img = src_img[y0:y1, x0:x1]

    ret = align_edge(
        moving_img,
        {'distmap': tpl_distmap,},
        init_bbox=init_bbox,
        **kws,
    )

    ret1 = {}
    for k, v in ret.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        ret1[k] = v

    out =  {
        'tgt_shape': src_img.shape[:2],
        'tpl_shape': tpl_distmap.shape[:2],
        **ret1,
    }
    return out
def process_spilt_align_camera(img, tpl_distmap, **kws):
    src_img = get_gray(img)
    tpl_w = tpl_distmap.shape[1]

    ret = align_edge(
        src_img,
        {'distmap': tpl_distmap,},
        **kws,
    )

    ret1 = {}
    for k, v in ret.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        ret1[k] = v

    out =  {
        'tgt_shape': src_img.shape[:2],
        'tpl_shape': tpl_distmap.shape[:2],
        **ret1,
    }
    return out

# def process_spilt_align_lightglue(img, tpl):
#     torch.set_grad_enabled(False)
#     img = get_gray(img)
#     _, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     cv2.imwrite('img.jpg',thresholded)
#     cv2.imwrite('tpl.jpg',tpl)
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
#     matcher = LightGlue(features='superpoint').eval().to(device)
    
        
#     image0 = load_image('img.jpg')
#     image1 = load_image('tpl.jpg')

#     print('img_tensor shape:',image0.shape)

#     print('tpl_tensor shape:',image1.shape)
    
#     feats0 = extractor.extract(image0.to(device))
#     feats1 = extractor.extract(image1.to(device))
    
#     matches01 = matcher({'image0': feats0, 'image1': feats1})
#     feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

#     kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
#     m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    
#     H,mask = cv2.findHomography(m_kpts0.cpu().numpy(),m_kpts1.cpu().numpy(),cv2.RANSAC,5.0)

#     return H
def sample_points_from_mask(mask, nr=None, seed=None):
    if seed is not None:
        npr.seed(seed)

    ys, xs = np.where(mask)
    idxs = npr.permutation(len(ys))
    if nr is not None:
        idxs = idxs[:nr]

    ys = ys[idxs]
    xs = xs[idxs]
    return list(zip(xs, ys))

def get_blank_mask(I):
    return I < LOW_INTENSITY

def get_gmm_img(imdic, features=[]):
    if isinstance(imdic, np.ndarray):
        imdic = {
            'rgb': imdic
        }

    need_yrb = 'yrb' not in imdic and any(i.startswith('yrb.') for i in features)
    if need_yrb:
        assert 'rgb' in imdic
        imdic['yrb'] = cv2.cvtColor(imdic['rgb'], cv2.COLOR_RGB2YCrCb)

    need_lab = 'lab' not in imdic and any(i.startswith('lab.') for i in features)
    if need_lab:
        assert 'rgb' in imdic
        imdic['lab'] = cv2.cvtColor(imdic['rgb'], cv2.COLOR_RGB2LAB)

    for f in features:
        trans, feats = f.split('.')
        if feats == '*':
            feats = trans

        cols = []
        idxs = [trans.index(i) for i in feats]
        imdic[f] = imdic[trans][..., idxs]

    return imdic

def get_feature_img(imdic, feature):
    return np.dstack([imdic[i] for i in feature])

def get_feature_samples(imdic, feature, coords=None):
    if coords is not None:
        coords = np.asarray(coords)
        return np.hstack([imdic[i][coords[:,1], coords[:,0]] for i in feature])

    img = get_feature_img(imdic, feature)
    return img.reshape(-1, img.shape[2])

def process_gmm_seg(img, classes, blank_mask, sample_nr, chull_scale=1/4, chull_erosion=1, random_seed=7, ys_init=10):
    imh, imw = img.shape[:2]

    coords = sample_points_from_mask(~blank_mask, sample_nr, seed=random_seed)
    coords = np.asarray(coords)
    samples = img[coords[:,1], coords[:,0]]

    cls_nr = 2
    gmm_s = time.time()
    gmm = GaussianMixture(n_components=cls_nr, covariance_type='tied').fit(samples)
    gmm_predict_s = time.time()
    label = gmm.predict(img.reshape(-1, 3)).reshape(imh, imw)
    gmm_e = time.time()
    print('gmm train',gmm_predict_s-gmm_s)
    print('gmm predict',gmm_e-gmm_predict_s)

    copper_label = None
    bg_label = None
    for i in range(cls_nr):
        mask = label == i
        ys, xs = np.where(mask)
        if ys.min() < ys_init:                       # ???zoom
            bg_label = i
        else:
            copper_label = i

    segmap = np.zeros((imh, imw), dtype='u1')
    copper_mask = label == copper_label
    segmap[copper_mask] = classes.copper.label
    bg_mask = label == bg_label
    segmap[bg_mask] = classes.bg.label

    valid_mask = get_valid_chull_mask(copper_mask, scale=chull_scale, erosion_radius=chull_erosion) # error?
    # valid_mask = get_valid_chull_mask(copper_mask, scale=1, erosion_radius=chull_erosion)
    blank_mask |= ~valid_mask
    segmap[blank_mask] = 0
    return segmap

def get_valid_chull_mask(mask, scale, erosion_radius):
    mask = mask.astype('u1')
    mask = rescale(mask, scale)
    # imsave('mask.png',mask)
    conts, _ = cv2.findContours(mask,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.concatenate(conts)
    hull = cv2.convexHull(pts)
    mask = cv2.fillConvexPoly(mask, hull, 255)
    kernel = disk(erosion_radius)
    mask = cv2.erode(mask, kernel)
    mask = rescale(mask, 1/scale) > 128
    return mask
