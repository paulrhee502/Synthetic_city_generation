'''
@Author: Fanjie Kong
Create on 03/20/2019

Binarizing the synthetic label
    1. Truncate them to square
    2. Resize all the synthetic images to a fixed size
    3. Rename all the files and make sure that they are compatible to Bohao's code

'''

import glob, os
import scipy.misc as smc
import numpy as np
import re
import time

final_res = (572, 572)
file_name_list = []
file_path = '/Users/Varun/Documents/CityEngine/Default Workspace/c3/images/austin2.2/raw/'
city_name = 'Austin'


def num_2_str(num):
    num_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']
    c_list = ['a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    num2c = dict(zip(num_list, c_list))
    return ''.join([num2c[e_n] for e_n in num])

def reject_patch(patch, threshold=0.5):
    return np.mean(np.sum(patch,axis=-1) == 255*3) > threshold

start_time = time.time()
for file in glob.glob(file_path + "*RGB.jpg"):
    img = smc.imread(file)
    img_gt = smc.imread(file.replace('RGB', 'GT'))
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
            if(img_gt[i, j, 0], img_gt[i, j, 1], img_gt[i, j, 2]) < (35, 35, 35):
                img_GT[i, j] = 255
            else:
                img_GT[i, j] = 0
    img_GT = smc.imresize(img_GT, final_res, interp='nearest')

    img_RGB = img[:t_shape, :t_shape, :]
    img_RGB = smc.imresize(img_RGB, final_res, interp='bilinear')

    if reject_patch(img_RGB): #if less than 50% of patch is city
        print('Too much of patch is on edge')
        continue
    else:
        smc.imsave(city_name + '_' + num_2_str(num[0]) + '_' + num[1] + '_RGB.tif'
                   , img_RGB)
        smc.imsave(city_name + '_' + num_2_str(num[0]) + '_' + num[1] + '_GT.tif'
                   , img_GT)
    file_name_list.append(city_name + '_' + num_2_str(num[0]) + '_' + num[1])

# save as file for collection
print(file_name_list)
np.savetxt('colTileNames.txt', np.array(file_name_list), fmt='%s', delimiter='\n')
print('Duration: {}'.format(time.time()-start_time))
