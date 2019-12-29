"""
Author: Varun Nair
Date: 6/28/2019
"""
import os
import numpy as np
import imageio as imio
from skimage import exposure
from skimage.util import img_as_float, img_as_uint, img_as_ubyte
import cv2 as cv
import time

def gammaCorrect(image, s, gamma):
    """Scales the color channels of a single image tile
    @param: tile is the file you are editing
    @param: s is vector of color scalings for that city
    @param: gamma is exponent for scaling"""

    image[:,:,0] = s[0] * image[:,:,0]**gamma
    image[:,:,1] = s[1] * image[:,:,1]**gamma
    image[:,:,2] = s[2] * image[:,:,2]**gamma
    return image

def WB2(img, threshold=0.005):
    balanced_img = np.zeros_like(img) #Initialize final image

    for i in range(3): #i stands for the channel index
        hist, bins = np.histogram(img[..., i].ravel(), 256, (0, 255))
        cum_hist = np.cumsum(hist) / hist.sum()
        bmin = np.where(cum_hist>threshold)[0][0]
        bmax = np.where(cum_hist>1-threshold)[0][0]
        balanced_img[...,i] = np.clip(img[...,i], bmin, bmax)
        balanced_img[...,i] = (balanced_img[...,i]-bmin) / (bmax - bmin) * 255
    return balanced_img.round().astype('uint8')

def transform(src_dir, city, s, gamma, method='CLAHE'):
    """Scales all images of given city
    @param: src_dir is directory the tiles are in
    @param: city is the positive filter for the city you're looking for
    @param: s is vector of color scalings for that city
    @param: gamma is exponent for scaling
    @param: type of method to transform by"""

    imList = [f for f in os.listdir(src_dir) if ('RGB' in f
                                                and city in f
                                                and 'tif' in f)]
    path = src_dir + 'transformed_{}/'.format(method)
    try: os.mkdir(path)
    except: pass

    if method == 'WB2': #transformations considering entire city of data
        threshold = 0.005
        cum_hist = {}
        for i in range(3): #each color channel
            histSum = 0
            cumSum = 0
            for file in imList:
                image = imio.imread(src_dir + file)
                hist, bins = np.histogram(image[...,i].ravel(), 256, (0,255))
                histSum += hist.sum()
                cumSum += np.cumsum(hist)
            cum_hist[i] = cumSum / histSum

    for file in imList:
        name = file[0].upper() + file[1:]
        image = imio.imread(src_dir + file)

        if method == 'gamma_correction':
            new_image = gammaCorrect(image, s, gamma)
        elif method == 'scale_only':
            new_image = gamma(image, s, gamma=1.0)
        elif method == 'CLAHE':
            new_image = exposure.equalize_adapthist(image, clip_limit=0.01)
            new_image = img_as_uint(new_image)
        elif method == 'log':
            new_image = exposure.adjust_log(image)
        elif method == 'sigmoid':
            new_image = exposure.adjust_sigmoid(image, gain=4, cutoff=0.35)
        elif method == 'KyleWb2':
            new_image = KyleWB2(image)
        elif method == 'WB2':
            new_image = np.zeros_like(image) #Initialize final image

            for i in range(3): #each color channel
                bmin = np.where(cum_hist[i]>threshold)[0][0]
                bmax = np.where(cum_hist[i]>1-threshold)[0][0]
                new_image[...,i] = np.clip(image[...,i], bmin, bmax)
                new_image[...,i] = (new_image[...,i]-bmin) / (bmax - bmin) * 255
            new_image = new_image.round().astype('uint8')

        print(name)
        imio.imwrite(path + name, new_image)

if __name__ == '__main__':
    """For pure scaling and exponentiation"""
    scalingParis = np.array([1.2, 1., 0.85]) #with g=1.02
    scalingShanghai = np.array([1.2, 1.05, 0.85]) #with g=1.02
    scalingVegas = np.array([1.05, 1.05, 0.90]) #with g=1.0
    scalingKhartoum = np.array([1.05, 1., 0.85]) #with g=1.0

    '''
    start_time = time.time()
    transform('./inria_wb2_test_images/', 'austin',
        scalingParis, 1.02, method='WB2')
    print("--- %s seconds ---" % (time.time() - start_time))


    transform('./', 'kitsap',
        scalingParis, 1.02, method='WB2')
    transform('./', 'tyrol',
        scalingParis, 1.02, method='WB2')

    transform('./', 'Shanghai',
        scalingParis, 1.02, method='WB2')
    transform('./', 'Vegas',
        scalingParis, 1.02, method='WB2')
    transform('./', 'Paris',
        scalingParis, 1.02, method='WB2')
    transform('./', 'Khartoum',
        scalingParis, 1.02, method='WB2')
    '''

    transform('./', 'Paris', scalingParis, 1.0, method='gamma_correction')
    transform('./', 'Paris', scalingParis, 1.0, method='sigmoid')
    transform('./', 'Paris', scalingParis, 1.0, method='log')
    transform('./', 'Paris', scalingParis, 1.0, method='CLAHE')
    transform('./', 'Paris', scalingParis, 1.0, method='WB2')
