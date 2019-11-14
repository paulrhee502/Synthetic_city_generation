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

def num_2_str(num):
    '''Eliminates numbers in names (not tile numbers) by changing to letter'''

    num_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']
    c_list = ['a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    num2c = dict(zip(num_list, c_list))
    return ''.join([num2c[e_n] for e_n in num])

def reject_patch(patch, threshold=0.1):
    '''Decides whether to keep a patch based on amount of white in tile
    @param patch: RGB image to be checked
    @param threshold: in [0,1], lower is stricter amount of white in RGB
    '''
    return np.mean(np.sum(patch,axis=-1) == 255*3) > threshold

def binarize(src_dir, city_name, imageSize=(650,650), mode='buildings'):
    '''
    Function reads list of tile pairs in directory and creates GTs
    @param src_dir: directory with RGB and GT tiles
    @param city_name: name of city imagery is for
    @param imageSize: tuple of tile dimensions
    @param mode: takes either 'buildings' or 'roads' to read in GTs differently
    '''
    colTileNames = []

    for file in glob.glob(src_dir + "*RGB.jpg"):
        img = np.array(Image.open(file))

        if mode == 'buildings':
            img_gt = np.array(Image.open(file.replace('RGB', 'GT')))
        elif mode == 'roads':
            img_gt = np.array(Image.open(file.replace('RGB', 'GT2')))

        t_shape = min(img.shape[0], img.shape[1])

        img_GT = np.zeros((t_shape, t_shape))
        img_RGB = np.zeros((t_shape, t_shape, img.shape[2]))

        file = file.replace(src_dir, '')
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

        img_GT = np.array(Image.fromarray(img_GT).resize(imageSize,
                            resample=Image.NEAREST))

        img_RGB = img[:t_shape, :t_shape, :]
        img_RGB = np.array(Image.fromarray(img_RGB).resize(imageSize,
                            resample=Image.BILINEAR))

        if reject_patch(img_RGB): #if less than 90% of patch is city
            print('Too much of patch is on edge')
            continue
        else:
            img_RGB = Image.fromarray(img_RGB)
            img_GT = Image.fromarray(img_GT)

            img_RGB.save(city_name + '_' + num_2_str(num[0]) + '_' + num[1] + '_RGB.tif')
            #different file names based on whether it's roads or buildings
            if mode == 'buildings':
                img_GT.save(city_name + '_' + num_2_str(num[0]) + '_' + num[1] + '_GT.tif')
            elif mode == 'roads':
                img_GT.save(city_name + '_' + num_2_str(num[0]) + '_' + num[1] + '_GT2.tif')

        colTileNames.append(city_name + '_' + num_2_str(num[0]) + '_' + num[1])
    return colTileNames

start_time = time.time()
final_res = (650, 650)
#needs to have a trailing "/"
dir = '/Users/Varun/Documents/CityEngine/Default Workspace/c3/images/road_patches/test_env2/'
city = 'Austin'

file_name_list = binarize(dir, city, imageSize = final_res, mode='roads')

print(file_name_list)
np.savetxt('colTileNames.txt', np.array(file_name_list), fmt='%s', delimiter='\n')
print('Duration: {}'.format(time.time()-start_time))
