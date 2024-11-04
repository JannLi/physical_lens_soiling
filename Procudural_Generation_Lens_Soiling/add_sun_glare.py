#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:37:06 2024

@author:

v0  zs:  add sun glare.


    

"""

import cv2
import numpy as np
import random
import os
import pylab as plt

def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)





def add_sunglare_rgb_jpg(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    blur = cv2.GaussianBlur(gray_image, (0,0), sigmaX = 10)
    thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
    color_glare = image.copy().astype(np.int32)
    color_glare = color_glare * thresh[:,:,np.newaxis] 
    color_glare = color_glare/ 64
    color_glare = cv2.GaussianBlur(color_glare, (0,0), sigmaX = 100)
    color_glare= color_glare.astype(np.uint8)
    
    
    image2 = image.copy().astype(np.int32)

    image2 = np.minimum(255, image2 + color_glare  )
    return image2.astype(np.uint8)





image_file = '2023-07-25-17-20-50-509_CAM_SURROUND_BACK_17089172278569992.jpg'


mod = 'sun_glare'

image = cv2.imread(image_file)

    
if mod == 'sun_glare':
    image_rgb = add_sunglare_rgb_jpg(image)
    

    image_name = save_dir + mod + image_file
    cv2.imwrite(image_name,image_rgb)

