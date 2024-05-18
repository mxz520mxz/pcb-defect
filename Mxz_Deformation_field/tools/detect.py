#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch
import cv2
import time
import matplotlib.pyplot as plt

import sys
import os.path as osp
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import flow_vis
import numpy.random as npr
from sklearn.covariance import MinCovDet
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import neurite as ne 




from PIL import Image

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from networks import network

def detect(canvas,crop_box,moving_list,fixed_list,distmap_list,balck_mask_list,
           base_name_list,spilt_part,gpu='-1',save_moved = True,save_warp = False):
    
    start_time = time.time()
    # device handling
    if gpu and (gpu != '-1'):
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    model = base_dir+'/models/pcb/0140.pt'
    model = network.VxmDense.load(model, device)
    model.to(device)
    model.eval()

    [x0,y0,x1,y1] = crop_box
    
    for idx,base_name in enumerate(base_name_list):
      
        moving = moving_list[idx]
        fixed = fixed_list[idx]
        distmap = distmap_list[idx]
        balck_mask = balck_mask_list[idx]
        # print(base_name)
       
        # coords = []
        # coords = coords + sample_points_from_mask(fixed,1000)
        # coords = coords + sample_points_from_mask(255 - fixed,1000)
        
        mean = moving.mean()
    
        # cv2.imwrite(base_dir+'/results/'+base_name+'_moving.jpg',moving)
        # cv2.imwrite(base_dir+'/results/'+base_name+'_fixed.jpg',fixed)

        # cv2.imwrite(base_dir+'/results/'+base_name+'_distmap.jpg',distmap)
        # cv2.imwrite(base_dir+'/results/'+base_name+'_balck_mask.jpg',balck_mask.astype(np.uint8) * 255)
        
        # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # axes[0].imshow(moving,cmap='gray')
        # axes[0].set_title("Moving")
        # axes[0].axis('off')

        # axes[1].imshow(fixed, cmap='gray')
        # axes[1].set_title("Fixed")
        # axes[1].axis('off')

        
        moving = moving / 255
        fixed = fixed / 255
        moving = moving[np.newaxis, :, :, np.newaxis]
        fixed = fixed[np.newaxis, :, :, np.newaxis]

        input_moving = torch.from_numpy(moving).to(device).float().permute( 3, 0, 1, 2)
        input_fixed = torch.from_numpy(fixed).to(device).float().permute(3, 0, 1, 2)

        # predict
        moved, warp = model(input_moving, input_fixed, registration=True)
        moved = moved * 255
        warp = warp *255
        
        fixed = fixed.squeeze() * 255

        if save_moved:
            moved = moved.detach().cpu().numpy().squeeze()
            cv2.imwrite(base_dir+'/results/'+base_name+'_moved.jpg',moved)

        if save_warp:
            warp = warp.detach().cpu().numpy().squeeze()
            # flow_arrow,_ = ne.plot.flow([warp.transpose(1,2,0)[::5,::5]],width=15,scale=0.3)
            # print('warp shape:',warp.shape)
            # print('flow_arrow:',flow_arrow)
            # flow_arrow.savefig(base_dir+'/results/'+base_name+'_flow_arrow.jpg')
            # np.save(base_dir+'/results/'+base_name+'_warp.npy',warp)
           
            # new_min = -255
            # new_max = 255
            # scaled_array = np.interp(warp[0], (warp[0].min(), warp[0].max()), (new_min, new_max))
            # cv2.imwrite(base_dir+'/results/'+base_name+'_warp_x.jpg',scaled_array)
            
            # scaled_array = np.interp(warp[1], (warp[1].min(), warp[1].max()), (new_min, new_max))
            # cv2.imwrite(base_dir+'/results/'+base_name+'_warp_y.jpg',scaled_array)
            
            # # 模值
            # elementwise_modulus = np.sqrt(warp[0]**2 + warp[1]**2)
            # min_value = elementwise_modulus.min()
            # max_value = elementwise_modulus.max()

            # scaled_modulus = 255 * (elementwise_modulus - min_value) / (max_value - min_value)
            # scaled_modulus = scaled_modulus.astype(np.uint8)
            # cv2.imwrite(base_dir+'/results/'+base_name+'_warp_mod.jpg',scaled_modulus)
            
            #角度
            # angle_radians = np.arctan2(warp[0], warp[1])
            # angle_degrees = np.degrees(angle_radians)
            # cv2.imwrite(base_dir+'/results/'+base_name+'_warp_degree.jpg',angle_degrees)
            
            # #flow field arrow
            flow = warp.transpose(1,2,0)
            flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
            # plt.imsave(base_dir+'/results/'+base_name+'_flow.jpg',flow_color)

            # axes[2].imshow(flow_color)
            # axes[2].set_title("Flow")
            # axes[2].axis('off')

            # # 调整子图之间的间距
            # plt.tight_layout()
            # plt.savefig(base_dir+'/results/'+base_name+'_flow_check.jpg')
            
            min_value = flow.min()
            max_value = flow.max()
            
            scaled_data = (flow - min_value) / (max_value - min_value) 
            
            flow_color = flow_vis.flow_to_color(scaled_data, convert_to_bgr=False)

            # plt.imsave(base_dir+'/results/'+base_name+'_flow_scaled.jpg',flow_color)

            gray_img = get_gray(flow_color)
            # cv2.imwrite(base_dir+'/results/'+base_name+'_flow_gray.jpg',gray_img)
            
            sub = fixed.astype(np.uint8) ^ moved.astype(np.uint8)
            
            if mean > 20:   
                ys,xs = np.where(gray_img < 75)
                for y, x in zip(ys, xs):
                    sub[y,x] = 255
            
            scaled_data = 255 * scaled_data
            scaled_data = scaled_data.astype(np.uint8)
            flow_color = flow_vis.flow_to_color(scaled_data, convert_to_bgr=False)
            gray_img = get_gray(flow_color)
            
            
      
            ys,xs = np.where(gray_img > 110)
            for y, x in zip(ys, xs):
                distmap_value = distmap[y, x]         
                if distmap_value > 3:
                    sub[y,x] = 255
                else:
                    continue
            
               
            # if mean < 10:
            #     dist = np.zeros(flow_color.shape[:2])
            # else:
            #     coords = np.asarray(coords)
            #     train = np.hstack([flow_color[coords[:,1], coords[:,0]]])
            #     # print('train shape',train.shape)
            #     # print('train ',train)
            #     clf = MinCovDet(support_fraction=0.8).fit(train)
            #     test = flow_color.reshape(-1, flow_color.shape[2])
            #     dists = clf.mahalanobis(test)
                
            #     dist = dists.reshape(flow_color.shape[:2])
            #     pixels = np.ravel(dist)
            #     pixels_sorted = np.sort(pixels)
            #     percentile = np.percentile(pixels_sorted, 99.99)
            #     dist[dist <= percentile] = 0
                
                
            #     ys, xs = np.where(dist)
    
            #     for y, x in zip(ys, xs):
            #         distmap_value = distmap[y, x]         
            #         if distmap_value > 2:
            #             sub[y,x] = 255
            #         else:
            #             dist[y,x] = 0
                        
            # plt.imsave(base_dir+'/results/'+base_name+'_dist.jpg',dist,cmap = 'gray')
       
        ys, xs = np.where(sub)
    
        for y, x in zip(ys, xs):
            distmap_value = distmap[y, x]
            balck_value = balck_mask[y, x]
            
            if balck_value:
                sub[y,x] = 0
                
            if distmap_value > 1:
                continue
            else:
                sub[y,x] = 0
          
             
        # cv2.imwrite(base_dir+'/results/'+base_name+'_defect.jpg',sub)
        
        id_list = base_name.split('_')[:3]
     
        if 'row' in id_list:
            id1 = int(id_list[0])
            id2 = int(id_list[1]) 
         
            row = id2*512 + y0 + id1 * ((y1 - y0)/spilt_part) 
            col = canvas[y0:y1, x0:x1].shape[1]- 512 + x0
          
        elif 'col' in id_list:
            id1 = int(id_list[0])
            id2 = int(id_list[1]) 
            col = id2*512 + x0 
            body = lambda x: 2 if x == 0 else 1
            row = canvas[y0:y1, x0:x1].shape[0]/(body(id1)) - 512 + y0
           
        else:
            id1 = int(id_list[0])
            id2 = int(id_list[1]) # row
            id3 = int(id_list[2]) # col
            
            row = id2*512 + y0 + id1 * ((y1 - y0)/spilt_part) #x
            col = id3*512 + x0  #y
            
        row = int(row)
        col = int(col)
  
        sub = cv2.cvtColor(sub, cv2.COLOR_GRAY2RGBA)
        sub[:,:,3] = 0
        red_color = [255, 0, 0, 255]
        mask = (sub[:, :, 0] == 255) & (sub[:, :, 1] == 255) & (sub[:, :, 2] == 255)
        sub[mask] = red_color
        canvas[row:row + sub.shape[0], col:col + sub.shape[1]] = sub    
       
    print('over time is :',time.time()-start_time)
    
    
    
def get_gray(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def sample_points_from_mask(mask, nr=None, seed=None):
    if seed is not None:
        npr.seed(seed)

    ys,xs = np.where(mask)
    idxs = npr.permutation(len(ys))

    if nr is not None:
        idxs = idxs[:nr]

    ys = ys[idxs]
    xs = xs[idxs]
    return list(zip(xs, ys))
