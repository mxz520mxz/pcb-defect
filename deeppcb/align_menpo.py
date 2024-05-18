import numpy as np
from menpo import image as mpimg
from menpo.shape import PointDirectedGraph
from menpofit.lk import LucasKanadeFitter, GradientCorrelation, GradientImages, ECC, SSD
from menpofit.transform import DifferentiableAlignmentSimilarity, DifferentiableAlignmentAffine

def menpofit_lk(moving_img, fixed_img, init_bbox, scales=[1/16, 1/8, 1/4, 1/2, 1], tform_tp='similarity', residual_cls=SSD, **kws):
    # TODO
    pass

    # np_init_bbox_yx = np.array(init_bbox)[:, ::-1]

    # np_moving_img = np.array(moving_img.convert('L'))[None].astype('f4') / 255.0
    # np_fixed_img = np.array(fixed_img.convert('L'))[None].astype('f4') / 255.0

    # moving_img = mpimg.Image(np_moving_img)
    # fixed_img = mpimg.Image(np_fixed_img)
    # h, w = moving_img.shape

    # bbox_edges = [[0,1], [1,2], [2,3], [3,0]]

    # moving_img.landmarks['bounding_box'] = PointDirectedGraph.init_from_edges(
    #     points=np.array([
    #         [0, 0],
    #         [h-1, 0],
    #         [h-1, w-1],
    #         [0, w-1],
    #     ]),
    #     edges=bbox_edges
    # )

    # init_bb = PointDirectedGraph.init_from_edges(
    #     points=np_init_bbox_yx,
    #     edges=bbox_edges
    # )

    # if tform_tp == 'similarity':
    #     transform = DifferentiableAlignmentSimilarity
    # else:
    #     transform = DifferentiableAlignmentAffine

    # fitter = LucasKanadeFitter(moving_img, scales=scales, residual_cls=residual_cls, transform=transform, **kws)
    # fr = fitter.fit_from_bb(fixed_img, init_bb)
    # refined_bbox = fr.final_shape.points[:, ::-1]

    # return {
    #     'refined_bbox': refined_bbox
    # }
