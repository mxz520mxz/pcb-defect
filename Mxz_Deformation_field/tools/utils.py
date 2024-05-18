import os
import numpy as np
import cv2
from skimage import measure


def default_unet_features():
    nb_features = [
        [16,32,32,32],
        [32,32,32,32,32,16,16]

    ]
    return nb_features


def load_data(directory):
    fixed_img_train_list = []
    moving_img_train_list = []
    fixed_img_val_list = []
    moving_img_val_list = []
    fixed_img_test_list = []
    moving_img_test_list = []
    file_list = os.listdir(directory)
    for file_name in file_list:
        if 'test' == file_name:
            test_path = os.path.join(directory, file_name)
            test_list = os.listdir(test_path)
            for test_name in test_list:
                if 'fixed' == test_name:
                    fixed_path = os.path.join(test_path,test_name)
                    fixed_test_img_path_list = os.listdir(fixed_path)
                    for fixed_test_img_path in fixed_test_img_path_list:
                        fixed_test_img_path = os.path.join(fixed_path,fixed_test_img_path)
                        fixed_test_img = cv2.imread(fixed_test_img_path,cv2.IMREAD_GRAYSCALE)
                        fixed_test_img = cv2.resize(fixed_test_img, (512, 512))
                        fixed_img_test_list.append(fixed_test_img)
                else:
                    moving_path = os.path.join(test_path,test_name)
                
                    moving_test_img_path_list = os.listdir(moving_path)
                    for moving_test_img_path in moving_test_img_path_list:
                        moving_test_img_path = os.path.join(moving_path,moving_test_img_path)
                        moving_test_img = cv2.imread(moving_test_img_path,cv2.IMREAD_GRAYSCALE)
                        moving_test_img = cv2.resize(moving_test_img, (512, 512))
                        moving_img_test_list.append(moving_test_img)
        elif 'train' == file_name:
            test_path = os.path.join(directory, file_name)
            test_list = os.listdir(test_path)
            for test_name in test_list:
                if 'fixed' == test_name:
                    fixed_path = os.path.join(test_path,test_name)
                    fixed_test_img_path_list = os.listdir(fixed_path)
                    for fixed_test_img_path in fixed_test_img_path_list:
                        fixed_test_img_path = os.path.join(fixed_path,fixed_test_img_path)
                        fixed_test_img = cv2.imread(fixed_test_img_path,cv2.IMREAD_GRAYSCALE)
                        fixed_test_img = cv2.resize(fixed_test_img, (512, 512))
                        fixed_img_train_list.append(fixed_test_img)
                else:
                    moving_path = os.path.join(test_path,test_name)
                
                    moving_test_img_path_list = os.listdir(moving_path)
                    for moving_test_img_path in moving_test_img_path_list:
                        moving_test_img_path = os.path.join(moving_path,moving_test_img_path)
                        moving_test_img = cv2.imread(moving_test_img_path,cv2.IMREAD_GRAYSCALE)
                        moving_test_img = cv2.resize(moving_test_img, (512, 512))
                        moving_img_train_list.append(moving_test_img)
                    
        elif 'val' == file_name:
            test_path = os.path.join(directory, file_name)
            test_list = os.listdir(test_path)
            for test_name in test_list:
                if 'fixed' == test_name:
                    fixed_path = os.path.join(test_path,test_name)
                    fixed_test_img_path_list = os.listdir(fixed_path)
                    for fixed_test_img_path in fixed_test_img_path_list:
                        fixed_test_img_path = os.path.join(fixed_path,fixed_test_img_path)
                        fixed_test_img = cv2.imread(fixed_test_img_path,cv2.IMREAD_GRAYSCALE)
                        fixed_test_img = cv2.resize(fixed_test_img, (512, 512))
                        fixed_img_val_list.append(fixed_test_img)
                else:
                    moving_path = os.path.join(test_path,test_name)
                
                    moving_test_img_path_list = os.listdir(moving_path)
                    for moving_test_img_path in moving_test_img_path_list:
                        moving_test_img_path = os.path.join(moving_path,moving_test_img_path)
                        moving_test_img = cv2.imread(moving_test_img_path,cv2.IMREAD_GRAYSCALE)
                        moving_test_img = cv2.resize(moving_test_img, (512, 512))
                        moving_img_val_list.append(moving_test_img)

    fixed_img_train = np.stack(fixed_img_train_list)
    moving_img_train = np.stack(moving_img_train_list)
    fixed_img_val = np.stack(fixed_img_val_list)
    moving_img_val = np.stack(moving_img_val_list)
    fixed_img_test = np.stack(fixed_img_test_list)
    moving_img_test = np.stack(moving_img_test_list)

    fixed_img_train  = fixed_img_train.astype('float')/255 
    moving_img_train = moving_img_train.astype('float')/255
    fixed_img_val    = fixed_img_val.astype('float')/255   
    moving_img_val   = moving_img_val.astype('float')/255  
    fixed_img_test   = fixed_img_test.astype('float')/255  
    moving_img_test  = moving_img_test.astype('float')/255 

    print('fixed train shape is :',fixed_img_train.shape)
    print('moving train shape is :',moving_img_train.shape)     
    print('fixed val shape is :',fixed_img_val.shape)
    print('moving val shape is :',moving_img_val.shape)    
    print('fixed test shape is :',fixed_img_test.shape)
    print('moving test shape is :',moving_img_test.shape)    

    return [fixed_img_train,moving_img_train,fixed_img_val,moving_img_val,fixed_img_test,moving_img_test]


def data_generator(fixed_data,moving_data,batch_size = 1):
    shape = fixed_data.shape[1:]
    ndims = len(shape)

    zero_field = np.zeros([batch_size,*shape,ndims])

    while True:
        idx = np.random.randint(0,moving_data.shape[0],size=batch_size)
        moving_images = moving_data[idx,...,np.newaxis]
        fixed_images = fixed_data[idx,...,np.newaxis]
        inputs = [moving_images,fixed_images]

        outputs = [fixed_images,zero_field]

        yield (inputs,outputs)
        
def get_gray(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

        