#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:54:55 2024

@author: sczone
"""

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import os

image_files = 'image'
modes = ['mud','rain']
for mod in modes:
    image_save_dir = './' + 'image_' + mod + r'/' 
   
    os.makedirs(image_save_dir,exist_ok = True )
 


image_file_list = os.listdir(image_files)
for item in image_file_list:
    image = cv2.imread(os.path.join(image_files,item) , cv2.IMREAD_UNCHANGED  )
    for mod in modes:
        transform = A.Spatter(always_apply=False, p=1.0,mean=(0.65,0.65),std=(0.3,0.3),gauss_sigma=(2,2),intensity=(0.6,0.6),cutout_threshold=(0.68,0.68),mode=[mod])

        transformed = transform(image = image)
        transformed_image = transformed['image']
        
        image_save_dir = './' + 'image_' + mod + r'/' 
            
        image_name = image_save_dir + item
        cv2.imwrite(image_name,transformed_image)