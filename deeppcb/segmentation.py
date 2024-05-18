import numpy as np
from skimage.segmentation.morphsnakes import _curvop
from skimage.util import img_as_float
import pylab as plt

def morphological_chan_vese(image, seeds, mask, num_iter,
                            smoothing=1, lambda1=1, lambda2=1):

    image = img_as_float(image)
    has_mask = mask is not None
    if not has_mask:
        mask = np.ones(image.shape[:2], dtype=bool)

    u = seeds

    # kmeans like method
    for it in range(num_iter):
        fg_mask = u & mask
        bg_mask = ~u & mask
        c_fg = (image[fg_mask]).sum() / float(fg_mask.sum() + 1e-8)
        c_bg = (image[bg_mask]).sum() / float(bg_mask.sum() + 1e-8)

        # Image attachment
        du = np.gradient(u.astype('i4'))
        abs_du = np.abs(du).sum(0)
        aux = abs_du * (lambda1 * (image - c_fg)**2 - lambda2 * (image - c_bg)**2)

        u[aux < 0] = True
        u[aux > 0] = False

        for _ in range(smoothing):
            u_ = _curvop(u)

    if has_mask:
        u[~mask] = False

    return u
