#!/usr/bin/env python
import numpy as np
import numpy.linalg as npl
from skimage.draw import rectangle
from scipy.ndimage.morphology import distance_transform_edt
import cv2
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.util import img_as_ubyte
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import kornia.geometry as KG
from pprint import pprint
from .base import get_bbox, get_edge, get_bbox_H, imsave
import pylab as plt



DFLT_DISTMAP_KWS = {
    'max_dist': 100,
}

class AffineModel(nn.Module):
    def __init__(self, sx=1, sy=1) -> None:
        super().__init__()
        self.model = nn.Parameter(torch.zeros((2, 3)))
        self.register_buffer('offset', torch.eye(3)[:2])
        self.register_buffer('scale', torch.tensor([[sx, 0], [0, sy]]))
        self.reset_model()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.model})'

    def reset_model(self):
        torch.nn.init.zeros_(self.model)

    def compose_matrix(self) -> torch.Tensor:
        return torch.mm(self.scale.float(), self.model.float()) + self.offset.float()

    def forward(self) -> torch.Tensor:
        return KG.convert_affinematrix_to_homography(self.compose_matrix().unsqueeze(dim=0))

    def forward_inverse(self) -> torch.Tensor:
        return torch.unsqueeze(torch.inverse(self.compose_matrix()), dim=0)

def get_edge_img(imdic, with_edge=False, with_distmap=False, with_points=False, edge_kws={}, distmap_kws={}):
    distmap_kws =  {**DFLT_DISTMAP_KWS, **distmap_kws}

    if isinstance(imdic, np.ndarray):
        imdic = {
            'img': imdic,
            'shape': imdic.shape[:2]
        }

    assert isinstance(imdic, dict)
    if with_distmap and 'distmap' not in imdic:
        with_edge = True

    if with_points and 'points' not in imdic:
        with_edge = True

    if 'edge' not in imdic and with_edge:
        assert 'img' in imdic
        img = np.asarray(imdic['img'])
        imdic['edge'] = get_edge(img, **edge_kws)

    if 'distmap' not in imdic and with_distmap:
        assert 'edge' in imdic
        edge = imdic['edge']
        max_dist = distmap_kws['max_dist']
        if max_dist < 1:
            max_dist = int(max_dist * edge.shape[1])

        distmap = distance_transform_edt(~edge)

        # distmap = cv2.distanceTransform(~edge, distanceType=cv2.DIST_L2, maskSize=5)
        distmap = distmap.clip(max=max_dist)
        imdic['distmap'] = distmap

    if 'points' not in imdic and with_points:
        assert 'edge' in imdic
        edge = imdic['edge']
        vs, us = np.where(edge)
        imdic['points'] = np.vstack((us, vs)).T

    if 'shape' not in imdic:
        for k in ['img', 'edge', 'distmap']:
            if k in imdic:
                imdic['shape'] = imdic[k].shape[:2]
                break

    assert 'shape' in imdic
    return imdic

def align_edge(moving, fixed, init_bbox=None, moving_mask=None, tform_tp='projective', sim_no_scale=False, optim='adam', lr=3e-3, H_sx=1, H_sy=1, max_iters=100, max_patience=5, lr_gamma=0.9, lr_sched_step=10, max_dist=200, abs_tol=1e-3, err_th=None, msg_prefix='', dev='cuda:0', verbose=True, **kws):
# def align_edge(moving, fixed, init_bbox=None, moving_mask=None, tform_tp='projective', sim_no_scale=False, optim='adam', lr=3e-3, H_sx=1, H_sy=1, max_iters=100, max_patience=5, lr_gamma=0.9, lr_sched_step=50, max_dist=200, abs_tol=1e-4, err_th=None, msg_prefix='', dev='cuda:0', verbose=True, **kws):
    # 0: moving_inside
    # 1: fixed like extended moving
    # 2: fixed
    params = {
        'tform_tp': tform_tp,
        'sim_no_scale': sim_no_scale,
        'lr': lr,
        'max_iters': max_iters,
        'max_patience': max_patience,
        'max_dist': max_dist,
        'H_scale': (H_sx, H_sy),
    }

    if verbose:
        print ("align params")
        pprint (params)

    fixed = get_edge_img(fixed, with_distmap=True, distmap_kws={
        'max_dist': max_dist,
    })
    fixed_distmap = np.asarray(fixed['distmap'])


    fixed_h, fixed_w = fixed_distmap.shape

    K = np.array([[fixed_w/2, 0, fixed_w/2], [0, fixed_h/2, fixed_h/2], [0, 0, 1]], dtype='f4')
    invK = npl.inv(K)


    moving = get_edge_img(moving, with_points=True)
    xs0 = moving['points']
    if moving_mask is not None:
        sel = np.where(moving_mask[xs0[:,1], xs0[:,0]])[0]
        xs0 = xs0[sel]

    init_bbox = get_bbox(init_bbox, moving['shape'])
    # raise
    H_01 = get_bbox_H(moving['shape'], init_bbox, tform_tp)
    H_10 = npl.inv(H_01)


    xs0 = np.hstack((xs0, np.ones((len(xs0), 1))))
    len_xs0_sample = len(xs0)
    xs1 = xs0.dot(H_10.T)
    xs1 = xs1.dot(invK.T)
    t_xs1 = torch.tensor(xs1.T.astype('f4')[None,...]).to(dev)

    t_fixed_distmap = torch.tensor(fixed_distmap[None, None, ...].astype('f4')).to(dev)
    if tform_tp == 'similarity':
        model = KG.Similarity(scale=not sim_no_scale)
    elif tform_tp == 'affine':
        model = AffineModel(sx=H_sx, sy=H_sy)
    elif tform_tp == 'projective':
        model = KG.Homography()
    else:
        raise

    model.to(dev)
    if optim == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optim == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_model = model().detach()
    best_iter = 0
    scheduler = ExponentialLR(optimizer, gamma=lr_gamma)

    import time
    align_start = time.time()
    for cur_iter in range(max_iters):
        optimizer.zero_grad()

        t_m = model()
        proj_points = torch.bmm(t_m, t_xs1)
        proj_us = proj_points[:, 0] / proj_points[:, 2]
        proj_vs = proj_points[:, 1] / proj_points[:, 2]
        grid = torch.stack((proj_us, proj_vs), dim=-1)[:,None,...]
        sampled_values = F.grid_sample(t_fixed_distmap, grid, padding_mode='border',align_corners=True).squeeze()

        loss = sampled_values.mean()

        cur_loss = loss.item()

        if cur_loss < best_loss - abs_tol:
            best_loss = cur_loss
            best_iter = cur_iter
            best_model = t_m.detach()
            if err_th is not None and best_loss < err_th:
                break

        elif cur_iter - best_iter > max_patience:
            break

        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        if cur_iter > 0 and cur_iter % lr_sched_step == 0:
            scheduler.step()

        if cur_iter % 2 and verbose:
            cur_lr = scheduler.get_last_lr()[0]
            print (msg_prefix+f"iter: {cur_iter}/{best_iter} loss: {cur_loss:.6f}/{best_loss:.6f} lr:{cur_lr:.6f}")


    # print('sample ratio',len_xs0_sample/len_xs0_before)
    # print('sample time', sample_end-sample_start)
    # print('1:and time', and_time_e-and_time_s)
    # print('draw time', draw_e-draw_s)
    # print('mask time', mask_e-draw_e)
    # print('where time', where_e-mask_e)
    # print('2:intersection time', intersection_e-intersection_s)
    # print('align time', align_end-align_start)

    m = best_model.cpu().numpy()[0]
    H_21 = K.dot(m).dot(invK)
    H_20 = H_21.dot(H_10)

    return {
        'H_21': H_21,
        'H_20': H_20,
        'err': float(best_loss),
        'params': params,
    }
