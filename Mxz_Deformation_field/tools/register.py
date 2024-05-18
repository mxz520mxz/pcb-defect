#!/usr/bin/env python
import os
import argparse
import numpy as np
import nibabel as nib
import torch
import cv2
import time
import matplotlib.pyplot as plt

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
from networks import network

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
parser.add_argument('--warp', default=True,help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()
start_time = time.time()
# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

add_feat_axis = not args.multichannel

directory = '../DeepPCB/test_data'
file_list = os.listdir(directory)
moving_list = []
fixed_list = []
distmap_list = []
base_name_list = []
for file_name in file_list:
    if 'distmap' in file_name:
        base_name=file_name.split('crop_distmap_')[1].split('.jpg')[0]
        distmap_dir = 'crop_distmap_'+base_name+'.jpg'
        distmap_dir = 'crop_distmap_'+base_name+'.jpg'
        moving_dir = 'crop_img_'+base_name+'.jpg'
        fixed_dir = 'crop_tpl_'+base_name+'.jpg'
        
        moving_img = cv2.imread(os.path.join(directory,moving_dir),0)
        fixed_img = cv2.imread(os.path.join(directory,fixed_dir),0)
        distmap_img = cv2.imread(os.path.join(directory,distmap_dir),0)
        
        moving_list.append(moving_img)
        fixed_list.append(fixed_img)
        distmap_list.append(distmap_img)
        base_name_list.append(base_name)
        
model = network.VxmDense.load(args.model, device)
model.to(device)
model.eval()

for idx,base_name in enumerate(base_name_list):
    moving = moving_list[idx]
    fixed = fixed_list[idx]
    distmap = distmap_list[idx]
    
    cv2.imwrite('results/'+base_name+'_fixed.jpg',fixed)
    
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

    if args.moved:
        moved = moved.detach().cpu().numpy().squeeze()
        cv2.imwrite('results/'+base_name+'_moved.jpg',moved)

    if args.warp:
        warp = warp.detach().cpu().numpy().squeeze()
        print('warp shape',warp.shape)
    
        cv2.imwrite('results/'+base_name+'_warp.jpg',warp[0])
        
    sub = fixed.astype(np.uint8) ^ moved.astype(np.uint8)

    ys, xs = np.where(sub)

    for y, x in zip(ys, xs):
        distmap_value = distmap[y, x]
        if distmap_value > 1:
            continue
        else:
            sub[y,x] = 0
            
    # plt.imsave('results/'+base_name+'_defect.jpg',sub)
    cv2.imwrite('results/'+base_name+'_defect.jpg',sub)
print('over time is :',time.time()-start_time)

