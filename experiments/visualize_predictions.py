'''
@Author: Varun Nair
Create on 11/29/2019

This code includes functions for making visual maps of model predctions vs. GTs
'''

import glob, os
import numpy as np
import time
from PIL import Image

def predictionMap(imagePath):
    """Given the path to a model's prediction png file, this function compares
    that prediction to a GT in the same directory and saves a new image showing
    true positives, false positives, false negatives, and true negatives in
    different colors.
    True positives in black
    False positives in blue
    False negatives in red
    True negatives in white
    """

    pred = np.array(Image.open(imagePath))
    gt = np.array(Image.open(imagePath.replace('pred.png','GT.tif')))

    gt2 = gt/255
    compare = pred - gt2

    visual = np.zeros((5000,5000, 3))

    for i in range(5000):
        for j in range(5000):
            if compare[i,j] == -1: #false negative
                visual[i,j,0] = 255 #red channel
            elif compare[i,j] == 1: #false positive
                visual[i,j,2] = 255 #blue channel
            elif compare[i,j] == 0 and gt[i,j] == 1: #true positive
                visual[i,j,1] = 255
            elif compare[i,j] == 0 and gt[i,j] == 0: #true negative
                visual[i,j,:] = (255,255,255)

            if (i!=0 and i%1000==0) and j==4999:
                print(i,j)

    im = Image.fromarray(visual.astype(np.uint8))
    im.save(imagePath.replace('pred.png','performance.png'))
    return True

def run_over(directory):
    """Uses function predictionMap() and iterates through all possible file
    pairs in a directory"""

    for file in glob.glob(directory + "*pred.png"):
        file = file.replace(directory, '')
        print("-"*10 + 'Now processing ' + file + '-'*10)

        predictionMap(file)
        print("-"*10 + 'Done processing ' + file + '-'*10)

    return None

if __name__ == '__main__':
    start_time = time.time()
    run_over('/Users/Varun/Documents/CityEngine/Default Workspace/c3/images/accuracy/R0+syn/')
    print(time.time()-start_time)
