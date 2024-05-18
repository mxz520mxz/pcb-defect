import numpy as np
import cv2
from sklearn.covariance import MinCovDet
from .base import binary_erosion
from .target import get_gmm_img, sample_points_from_mask, get_feature_samples

def process_ood_seg(img, mask, cfg, shadow_mask=None):
    shrink = cfg.segmap_shrink
    edge_region_radius = cfg.edge_region_radius
    random_seed = cfg.random_seed

    mask = binary_erosion(mask, shrink)

    edge_mask = mask ^ binary_erosion(mask, edge_region_radius)

    sample_masks = {
        'all': mask,
        'edge': edge_mask,
    }

    feat = cfg.feature
    dist_th = cfg.dist_th
    sample_nr = cfg.sample_nr
    support_frac = cfg.support_frac

    if shadow_mask is not None:
        shadow_mask = mask & shadow_mask
        sample_masks['shadow'] = shadow_mask

    coords = []
    for k, v in sample_masks.items():
        coords += sample_points_from_mask(sample_masks[k], sample_nr[k], seed=random_seed)

    imdic = get_gmm_img(img, set(feat))

    train_xs = get_feature_samples(imdic, feat, coords)
    clf = MinCovDet(support_fraction=support_frac).fit(train_xs)

    test_xs = get_feature_samples(imdic, feat)
    dists = clf.mahalanobis(test_xs).reshape(mask.shape)

    lbl = dists < dist_th
    lbl[~mask] = 0

    return lbl == 1, {
        'clf': clf,
        'feature': feat,
        'sample_points': coords,
        'mask': mask,
    }

def verify_ood_seg(img, segmaps, sample_points):
    copper_mask = segmaps['copper']
    bg_mask = segmaps['bg']
    canvas = (img.astype('i4') // 2 + np.dstack([copper_mask]*3) * [80, 0, 0] + np.dstack([bg_mask]*3) * [0, 80, 0]).astype('u1')
    for x, y in sample_points['copper']:
        cv2.circle(canvas, (x, y), 2, (0,255,0), thickness=-1)
    for x, y in sample_points['bg']:
        cv2.circle(canvas, (x, y), 2, (255,0,0), thickness=-1)

    return canvas
