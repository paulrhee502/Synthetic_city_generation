'''
@Author: Varun Nair
Create on 11/07/2019

Binarizing the synthetic label
    1. Truncate them to square
    2. Resize all the synthetic images to a fixed size
    3. Rename all the files and make sure that they are compatible to Bohao's code

'''

import glob, os
import numpy as np
import re
import time
import cv2 as cv
from PIL import Image

final_res = (650, 650)
#needs to have a trailing "/"
file_path = '/Users/Varun/Documents/CityEngine/Default Workspace/c3/images/road_patches/test_env2/'
city_name = 'Austin'


def num_2_str(num):
    num_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']
    c_list = ['a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    num2c = dict(zip(num_list, c_list))
    return ''.join([num2c[e_n] for e_n in num])

def reject_patch(patch, threshold=0.1): #input threshold is amount of white in RGB file
    return np.mean(np.sum(patch,axis=-1) == 255*3) > threshold

def binarize(file_path, city_name, mode='buildings'):
    '''
    Function reads list of tile pairs in directory and creates GTs
    @param file_path: directory with RGB and GT tiles
    @param city_name: name of city imagery is for
    @param mode: takes either 'buildings' or 'roads' to read in GTs differently
    '''
    colTileNames = []

    for file in glob.glob(file_path + "*RGB.jpg"):
        img = np.array(Image.open(file))

        if mode == 'buildings':
            img_gt = np.array(Image.open(file.replace('RGB', 'GT')))
        elif mode == 'roads':
            img_gt = np.array(Image.open(file.replace('RGB', 'GT2')))

        t_shape = min(img.shape[0], img.shape[1])

        img_GT = np.zeros((t_shape, t_shape))
        img_RGB = np.zeros((t_shape, t_shape, img.shape[2]))

        file = file.replace(file_path, '')
        # find the number in the name of file
        num = re.findall(r"\d+\.?\d*", file)

        print("-"*10+'Now processing '+ file +'-'*10)

        img_gt = img_gt[:t_shape, :t_shape, :]
        # Binarization
        for i in range(t_shape):
            for j in range(t_shape):
                if mode == 'buildings':
                    if(img_gt[i, j, 0], img_gt[i, j, 1], img_gt[i, j, 2]) < (35, 35, 35):
                        img_GT[i, j] = 255
                    else:
                        img_GT[i, j] = 0
                elif mode == 'roads':
                    if(img_gt[i, j, 0], img_gt[i, j, 1], img_gt[i, j, 2]) > (235, 235, 235):
                        img_GT[i, j] = 0
                    else:
                        img_GT[i, j] = 255

        #fills in spots of unlabeled road/building in GTs
        kernel = np.ones((1,1), np.uint8)
        img_GT = cv.morphologyEx(img_GT, cv.MORPH_CLOSE, kernel)

        #img_GT = morphology.binary_closing(img_GT) #from skimage import morphology

        img_GT = np.array(Image.fromarray(img_GT).resize(final_res,
                            resample=Image.NEAREST))

        img_RGB = img[:t_shape, :t_shape, :]
        img_RGB = np.array(Image.fromarray(img_RGB).resize(final_res,
                            resample=Image.BILINEAR))

        if reject_patch(img_RGB): #if less than 90% of patch is city
            print('Too much of patch is on edge')
            continue
        else:
            img_RGB = Image.fromarray(img_RGB)
            img_GT = Image.fromarray(img_GT)

            img_RGB.save(city_name + '_' + num_2_str(num[0]) + '_' + num[1] + '_RGB.tif')
            if mode == 'buildings':
                img_GT.save(city_name + '_' + num_2_str(num[0]) + '_' + num[1] + '_GT.tif')
            elif mode == 'roads':
                img_GT.save(city_name + '_' + num_2_str(num[0]) + '_' + num[1] + '_GT2.tif')

        colTileNames.append(city_name + '_' + num_2_str(num[0]) + '_' + num[1])
    return colTileNames

start_time = time.time()
file_name_list = binarize(file_path, city_name, mode='roads')
# save as file for collection
print(file_name_list)
np.savetxt('colTileNames.txt', np.array(file_name_list), fmt='%s', delimiter='\n')
print('Duration: {}'.format(time.time()-start_time))
