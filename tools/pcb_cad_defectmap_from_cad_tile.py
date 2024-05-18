#!/usr/bin/env python
import numpy as np
import cv2
import os.path as osp
import time
import toml
import json
from easydict import EasyDict as edict
from skimage.morphology import disk
import click
import os

def defect_map_from_cad(zoom, root, global_dir, deviation_dir, foreign_dir, align_region, cad_2x):

    img_dict = {}
    for name in os.listdir(osp.join(root, global_dir, f'{zoom}x', deviation_dir)):
        cam_id = name.split('_')[1][3:]
        img_dict[f'{cam_id}'] = name

    img_num = len(img_dict)

    cad_2x_gray = cv2.cvtColor(cad_2x, cv2.COLOR_BGR2GRAY)
    cad_2x[cad_2x_gray==255] = [4, 63, 120]
    cad_2x[cad_2x_gray==128] = [50, 40, 30]
    cad_2x[cad_2x_gray==0] = [0,100,0]

    b_channel, g_channel, r_channel = cv2.split(cad_2x) 
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    cad_2x = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) 

    defect_channel_global = np.ones(cad_2x.shape, dtype=cad_2x.dtype) * 255
    
    for i,info in img_dict.items():
        i = int(i)
        i_new = (i-1)%5
        img_name = img_dict[f'{i}'][:-4]
        cam_d_p = osp.join(root, global_dir, f'{zoom}x', deviation_dir, img_name + '.png')
        cam_f_p = osp.join(root, global_dir, f'{zoom}x', foreign_dir, img_name + '.png')
        cam_d = cv2.imread(cam_d_p, cv2.IMREAD_UNCHANGED)
        cam_f = cv2.imread(cam_f_p, cv2.IMREAD_UNCHANGED)
        defect_channel = np.ones(cam_d.shape, dtype=cam_d.dtype) * 255 
        defect_channel = cv2.add(cam_d, cam_f)

        dx = int(align_region[f'{i_new}']['cut_bbox'][0])
        cam_width = cam_d.shape[1]

        if i_new == 0:
            defect_channel_global[:,:cam_width,:] = defect_channel
        if i_new != 0 and i_new < img_num: 
            defect_channel_global[:,dx:dx+cam_width,:] = defect_channel
            defect_channel_global[:,dx:dx+cam_width,:] = defect_channel
        if i_new == img_num:
            print(defect_channel_global.shape,defect_channel.shape)
            defect_channel_global[:,dx:dx+cam_width,:] = defect_channel
    
    map_cad = cv2.add(cad_2x, defect_channel_global)

    return map_cad

def defect_map_from_tile(zoom, root, global_dir, deviation_dir, foreign_dir, image_dir, align_region, cad):

    img_dict = {}
    for name in os.listdir(osp.join(root, global_dir, f'{zoom}x', deviation_dir)):
        cam_id = name.split('_')[1][3:]
        img_dict[f'{cam_id}'] = name

    img_num = len(img_dict)

    defect_channel_global = np.zeros((cad.shape[0], cad.shape[1], 4), np.uint8) 
    
    cam_check_dict = {}
    for i,info in img_dict.items():
        i = int(i)
        i_new = (i-1)%5
        img_name = img_dict[f'{i}'][:-4]
        cam_d_p = osp.join(root, global_dir, f'{zoom}x', deviation_dir, img_name + '.png')
        cam_f_p = osp.join(root, global_dir, f'{zoom}x', foreign_dir, img_name + '.png')
        cam_i_p = osp.join(root, global_dir, f'{zoom}x', image_dir, img_name + '.jpg')
        cam_d = cv2.imread(cam_d_p, cv2.IMREAD_UNCHANGED)
        cam_f = cv2.imread(cam_f_p, cv2.IMREAD_UNCHANGED)
        cam_i = cv2.imread(cam_i_p, cv2.IMREAD_UNCHANGED)

        b_channel, g_channel, r_channel = cv2.split(cam_i)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 
        cam_i = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) 
        defect_channel = np.ones(cam_d.shape, dtype=cam_d.dtype) * 255 
        defect_channel = cv2.add(cam_d, cam_f)

        cam_check = cv2.add(defect_channel, cam_i)
        cam_check_dict[img_dict[f'{i}']] = cam_check

        dx = int(align_region[f'{i_new}']['cut_bbox'][0])
        cam_width = cam_d.shape[1]

        if i_new == 0:
            defect_channel_global[:,:cam_width,:] = cam_check
        if i_new != 0 and i_new < img_num: 
            defect_channel_global[:,dx:dx+cam_width,:] = cam_check
            defect_channel_global[:,dx:dx+cam_width,:] = cam_check
        if i_new == img_num:
            print(defect_channel_global.shape,cam_check.shape)
            defect_channel_global[:,dx:dx+cam_width,:] = cam_check
    
    return defect_channel_global, cam_check_dict


if __name__ == '__main__':
    
    #   result = {'board_result': 
    #         {'B': {'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/8530/B', 
    #                 'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/BR0100P04W0070087A1_L3'}, 
    #         'A': {'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/BR0100P04W0070087A1_L2', 
    #                 'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/8530/A'}}, 
    #         }

    # result = {'board_result': 
    #         { 
    #         'A': {'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_A', 
    #                 'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245/A'}}, 
    #         }

    # 9245
    result = {'board_result': {'A': {'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_A', 'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245/A'}, 'B': {'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245/B', 'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_B'}}, 'cam_result': {'20230613T205849S499_cam4_9245_concatenate_pillow.jpg': {'deviation_statistic': 'concave:8,convex:0,group:1', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245', 'foreign_statistic': 'black:15,gray:1,white:0,group:2', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_A'}, '20230613T205906S357_cam3_9245_concatenate_pillow.jpg': {'deviation_statistic': 'concave:3,convex:0,group:0', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_A', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245', 'foreign_statistic': 'black:2,gray:0,white:0,group:0'}, '20230613T205801S975_cam1_9245_concatenate_pillow.jpg': {'deviation_statistic': 'concave:1,convex:0,group:0', 'foreign_statistic': 'black:3,gray:0,white:0,group:0', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_A', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245'}, '20230613T205818S632_cam2_9245_concatenate_pillow.jpg': {'foreign_statistic': 'black:3,gray:0,white:0,group:1', 'deviation_statistic': 'concave:2,convex:1,group:0', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_A', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245'}, '20230613T210035S817_cam10_9245_concatenate_pillow.jpg': {'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_B', 'foreign_statistic': 'black:15,gray:1,white:0,group:2', 'deviation_statistic': 'concave:65,convex:31,group:80'}, '20230613T205939S181_cam7_9245_concatenate_pillow.jpg': {'foreign_statistic': 'black:24,gray:1,white:0,group:0', 'deviation_statistic': 'concave:34,convex:10,group:45', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_B', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245'}, '20230613T205958S071_cam8_9245_concatenate_pillow.jpg': {'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_B', 'deviation_statistic': 'concave:40,convex:4,group:45', 'foreign_statistic': 'black:19,gray:0,white:0,group:5'}, '20230613T205922S083_cam6_9245_concatenate_pillow.jpg': {'deviation_statistic': 'concave:5,convex:6,group:7', 'foreign_statistic': 'black:4,gray:1,white:0,group:0', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_B', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245'}, '20230613T210016S466_cam9_9245_concatenate_pillow.jpg': {'foreign_statistic': 'black:19,gray:0,white:0,group:0', 'deviation_statistic': 'concave:35,convex:8,group:39', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_B'}, '20230613T205834S406_cam5_9245_concatenate_pillow.jpg': {'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9245_A', 'deviation_statistic': 'concave:3,convex:0,group:0', 'foreign_statistic': 'black:7,gray:0,white:0,group:0', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9245'}}}

    # 9242
    # result = {'board_result': {'B': {'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9242/B', 'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9242_B'}, 'A': {'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9242_A', 'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9242/A'}}, 'cam_result': {'20230613T205105S163_cam3_9242_concatenate_pillow.jpg': {'foreign_statistic': 'black:4,gray:0,white:0,group:0', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9242', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9242_A', 'deviation_statistic': 'concave:1,convex:0,group:1'}, '20230613T205205S496_cam8_9242_concatenate_pillow.jpg': {'deviation_statistic': 'concave:69,convex:46,group:54', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9242', 'foreign_statistic': 'black:45,gray:8,white:0,group:10', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9242_B'}, '20230613T205117S988_cam4_9242_concatenate_pillow.jpg': {'deviation_statistic': 'concave:3,convex:0,group:0', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9242', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9242_A', 'foreign_statistic': 'black:9,gray:0,white:0,group:1'}, '20230613T205051S879_cam2_9242_concatenate_pillow.jpg': {'deviation_statistic': 'concave:3,convex:0,group:0', 'foreign_statistic': 'black:9,gray:3,white:0,group:0', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9242_A', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9242'}, '20230613T205219S881_cam9_9242_concatenate_pillow.jpg': {'foreign_statistic': 'black:28,gray:0,white:0,group:6', 'deviation_statistic': 'concave:175,convex:83,group:73', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9242_B', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9242'}, '20230613T205234S237_cam10_9242_concatenate_pillow.jpg': {'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9242', 'foreign_statistic': 'black:11,gray:0,white:0,group:1', 'deviation_statistic': 'concave:61,convex:32,group:26', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9242_B'}, '20230613T205039S526_cam1_9242_concatenate_pillow.jpg': {'deviation_statistic': 'concave:5,convex:0,group:1', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9242', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9242_A', 'foreign_statistic': 'black:8,gray:0,white:0,group:2'}, '20230613T205151S782_cam7_9242_concatenate_pillow.jpg': {'deviation_statistic': 'concave:8,convex:4,group:0', 'foreign_statistic': 'black:23,gray:1,white:0,group:2', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9242', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9242_B'}}}
    
    # 9246
    # result = {'cam_result': {'20230613T210130S841_cam3_9246_concatenate_pillow.jpg': {'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9246_A', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9246', 'foreign_statistic': 'black:32,gray:0,white:0,group:38', 'deviation_statistic': 'concave:44,convex:21,group:82'}, '20230613T210050S685_cam1_9246_concatenate_pillow.jpg': {'foreign_statistic': 'black:67,gray:2,white:0,group:71', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9246', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9246_A', 'deviation_statistic': 'concave:44,convex:28,group:89'}, '20230613T210228S150_cam10_9246_concatenate_pillow.jpg': {'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9246_B', 'deviation_statistic': 'concave:82,convex:3,group:31', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9246', 'foreign_statistic': 'black:12,gray:0,white:0,group:0'}, '20230613T210202S276_cam8_9246_concatenate_pillow.jpg': {'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9246', 'foreign_statistic': 'black:28,gray:2,white:0,group:17', 'deviation_statistic': 'concave:100,convex:1,group:16', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9246_B'}, '20230613T210101S924_cam2_9246_concatenate_pillow.jpg': {'foreign_statistic': 'black:103,gray:2,white:0,group:115', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9246', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9246_A', 'deviation_statistic': 'concave:40,convex:29,group:9'}, '20230613T210215S017_cam9_9246_concatenate_pillow.jpg': {'deviation_statistic': 'concave:119,convex:3,group:36', 'foreign_statistic': 'black:7,gray:0,white:0,group:2', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9246', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9246_B'}}, 'board_result': {'B': {'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9246/B', 'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9246_B'}, 'A': {'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9246/A', 'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9246_A'}}}
    
    # 9248
    # result = {'cam_result': {'20230613T210523S030_cam3_9248_concatenate_pillow.jpg': {'foreign_statistic': 'black:11,gray:0,white:0,group:1', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9248', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9248_A', 'deviation_statistic': 'concave:12,convex:0,group:0'}, '20230613T210510S641_cam4_9248_concatenate_pillow.jpg': {'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9248_A', 'deviation_statistic': 'concave:13,convex:0,group:1', 'foreign_statistic': 'black:18,gray:0,white:0,group:1', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9248'}, '20230613T210551S190_cam7_9248_concatenate_pillow.jpg': {'foreign_statistic': 'black:25,gray:1,white:0,group:0', 'deviation_statistic': 'concave:26,convex:2,group:7', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9248_B', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9248'}, '20230613T210639S895_cam10_9248_concatenate_pillow.jpg': {'deviation_statistic': 'concave:27,convex:0,group:13', 'foreign_statistic': 'black:13,gray:1,white:0,group:3', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9248_B', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9248'}, '20230613T210623S239_cam9_9248_concatenate_pillow.jpg': {'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9248_B', 'deviation_statistic': 'concave:28,convex:1,group:8', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9248', 'foreign_statistic': 'black:29,gray:1,white:0,group:4'}, '20230613T210433S875_cam1_9248_concatenate_pillow.jpg': {'deviation_statistic': 'concave:5,convex:0,group:0', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9248', 'foreign_statistic': 'black:1,gray:0,white:0,group:1', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9248_A'}, '20230613T210606S916_cam8_9248_concatenate_pillow.jpg': {'foreign_statistic': 'black:69,gray:1,white:1,group:13', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9248', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9248_B', 'deviation_statistic': 'concave:25,convex:7,group:14'}, '20230613T210458S389_cam2_9248_concatenate_pillow.jpg': {'deviation_statistic': 'concave:8,convex:0,group:0', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9248', 'foreign_statistic': 'black:7,gray:1,white:0,group:0', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9248_A'}}, 'board_result': {'A': {'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9248/A', 'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9248_A'}, 'B': {'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9248_B', 'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9248/B'}}}

    # 9240
    # result = {'cam_result': {'20230613T204852S395_cam2_9240_concatenate_pillow.jpg': {'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9240', 'foreign_statistic': 'black:11,gray:9,white:7,group:8', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9240_A', 'deviation_statistic': 'concave:24,convex:2,group:9'}, '20230613T204905S665_cam3_9240_concatenate_pillow.jpg': {'deviation_statistic': 'concave:24,convex:4,group:3', 'foreign_statistic': 'black:10,gray:12,white:12,group:13', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9240', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9240_A'}, '20230613T204837S414_cam1_9240_concatenate_pillow.jpg': {'foreign_statistic': 'black:6,gray:8,white:0,group:2', 'deviation_statistic': 'concave:64,convex:8,group:31', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9240_A', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9240'}}, 'board_result': {'A': {'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9240/A', 'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9240_A'}}}
    
    # 9252
    # result = {'cam_result': {'20230613T211853S374_cam8_9252_concatenate_pillow.jpg': {'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_B', 'foreign_statistic': 'black:80,gray:3,white:1,group:29', 'deviation_statistic': 'concave:37,convex:0,group:25', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252'}, '20230613T211916S455_cam9_9252_concatenate_pillow.jpg': {'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_B', 'foreign_statistic': 'black:25,gray:0,white:0,group:0', 'deviation_statistic': 'concave:53,convex:5,group:24', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252'}, '20230613T211729S182_cam4_9252_concatenate_pillow.jpg': {'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_A', 'foreign_statistic': 'black:45,gray:3,white:1,group:6', 'deviation_statistic': 'concave:7,convex:0,group:0'}, '20230613T211649S923_cam5_9252_concatenate_pillow.jpg': {'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_A', 'foreign_statistic': 'black:7,gray:1,white:0,group:3', 'deviation_statistic': 'concave:2,convex:0,group:0'}, '20230613T211630S191_cam1_9252_concatenate_pillow.jpg': {'foreign_statistic': 'black:20,gray:14,white:0,group:9', 'deviation_statistic': 'concave:13,convex:5,group:0', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_A'}, '20230613T211810S194_cam6_9252_concatenate_pillow.jpg': {'deviation_statistic': 'concave:27,convex:0,group:5', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_B', 'foreign_statistic': 'black:4,gray:0,white:0,group:0'}, '20230613T211709S199_cam2_9252_concatenate_pillow.jpg': {'deviation_statistic': 'concave:6,convex:1,group:0', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_A', 'foreign_statistic': 'black:16,gray:9,white:0,group:5'}, '20230613T211831S303_cam7_9252_concatenate_pillow.jpg': {'foreign_statistic': 'black:18,gray:0,white:0,group:4', 'deviation_statistic': 'concave:57,convex:7,group:20', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_B'}, '20230613T211750S141_cam3_9252_concatenate_pillow.jpg': {'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_A', 'foreign_statistic': 'black:14,gray:1,white:0,group:9', 'deviation_statistic': 'concave:8,convex:1,group:1', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252'}, '20230613T211938S387_cam10_9252_concatenate_pillow.jpg': {'foreign_statistic': 'black:14,gray:1,white:0,group:5', 'template_cad_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_B', 'defect_directory': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252', 'deviation_statistic': 'concave:19,convex:4,group:3'}}, 'board_result': {'B': {'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252/B', 'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_B'}, 'A': {'result_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/result/9252/A', 'template_path': '/home/vision/users/dengsx/pcb_py/data_bench_server/templates/9252_A'}}}

    zoom = 2
    global_dir = 'defect_global'
    deviation_dir = 'verify_deviations'
    foreign_dir = 'verify_foreigns'
    image_dir = 'save_aligned_images'

    for face, face_info in result['board_result'].items():
        result_path = face_info['result_path']
        template_path = face_info['template_path']
        print('---------------------------------------------')
        print(face)
        print(result_path)
        print(template_path)
        print('---------------------------------------------')

        cad = cv2.imread(osp.join(template_path, f'cad_image_{zoom}x.png'))
        tgt_tpl_dir = osp.join(template_path, f'target_{zoom}x')
        align_region = json.load(open(osp.join(tgt_tpl_dir, 'cut_region.json')))

        # save path
        verify_global_map = osp.join(result_path, global_dir, f'{zoom}x', 'verify_map')
        verify_rgb_defects = osp.join(result_path, global_dir, f'{zoom}x', 'verify_rgb_defects')
        os.makedirs(verify_global_map, exist_ok=True) 
        os.makedirs(verify_rgb_defects, exist_ok=True)

        # cad
        map_cad_b = defect_map_from_cad(zoom, result_path, global_dir, deviation_dir, foreign_dir, align_region, cad)
        save_map_path = osp.join(verify_global_map, 'defectmap_cad.png')
        print(save_map_path)
        cv2.imwrite(save_map_path, map_cad_b)

        # tile
        map_tile_b, cam_check_dict = defect_map_from_tile(zoom, result_path, global_dir, deviation_dir, foreign_dir, image_dir, align_region, cad)
        save_map_path = osp.join(verify_global_map, 'defectmap_tile.png')
        print(save_map_path)
        cv2.imwrite(save_map_path, map_tile_b)
        for name, img in cam_check_dict.items():
            save_tile_path = osp.join(verify_rgb_defects, name)
            print(save_tile_path)
            cv2.imwrite(save_tile_path, img)
        
        result['board_result'][f'{face}']['defect_map_path'] = verify_global_map

    print(result)
