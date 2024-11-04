#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:36:01 2024

@author: sczone

Created on Wed Apr 10 16:37:06 2024
@author: 

v1  zs:  fist version. add mud or water droplet.
    need texture from  generate_texture_paper.py 
    
    
    
"""

import cv2
import numpy as np
import random
import os








def get_texture_random(texture_file_list):
    random_item = random.choice(texture_file_list)
    return random_item
    


def add_mudByTxture(image,dir_texture):
    dir_texture = cv2.resize(dir_texture , (image.shape[1],image.shape[0]) )

    dir_texture = (dir_texture  - dir_texture.min() )/ (dir_texture.max() - dir_texture.min() )
    dir_mask = dir_texture.copy() 
    dir_mask[dir_mask>0.5] = 1

    dir_mask = cv2.GaussianBlur(dir_mask, ksize=(0, 0),sigmaX = 6)
   
    base_color = np.array([20, 42, 63]) *  ( np.random.random()  *2 )
    

    dir_mask = dir_mask[:,:,np.newaxis]
    mean_color = image.mean()
    alpha = 0.1
    mix_color = (base_color* (1 - alpha) + mean_color * alpha ) 

    image_blur =  cv2.GaussianBlur(image, ksize=(0, 0),sigmaX = 15)
    
     
    image_blur = image * (1 - dir_mask) + image_blur * dir_mask
    image_out = image_blur * (1 - dir_mask) + mix_color * dir_mask
 
    image_out = image_out.astype(np.uint8)
    return image_out , dir_mask * 255 


def add_dirtwaterByTxture(image,dir_texture):
    dir_texture = cv2.resize(dir_texture , (image.shape[1],image.shape[0]) )


    dir_texture = (dir_texture  - dir_texture.min() )/ (dir_texture.max() - dir_texture.min() )
    blur_mask = dir_texture.copy()
    
    blur_mask[blur_mask>0.2] = 1
    
    
    blur_mask = cv2.GaussianBlur(blur_mask, ksize=(0, 0),sigmaX = 10)
    dir_mask = dir_texture.copy()
    dir_mask = (dir_mask  - dir_mask.min() )/ (dir_mask.max() - dir_mask.min() )

    dir_mask[dir_mask>0.7] = 1
  
    
    dir_mask = cv2.GaussianBlur(dir_mask, ksize=(0, 0),sigmaX = 8)
    

    water_base_color = np.array([255, 255, 255])  *  ( np.random.random()* 0.5 + 0.5 )
    
    if dir_texture.ndim < 3 :
        
        dir_texture = dir_texture[:,:,np.newaxis]
        dir_mask = dir_mask[:,:,np.newaxis]
        blur_mask = blur_mask[:,:,np.newaxis]
    mean_color_ratio = image.mean()  /255
    alpha = mean_color_ratio
    mix_color = (water_base_color* (1 - alpha)  ) 

    
    image_blur =  cv2.GaussianBlur(image, ksize=(0, 0),sigmaX = 32)
    
    image = cv2.GaussianBlur(image, ksize=(0, 0),sigmaX = 2)
    image_blur = image * (1 - blur_mask) + image_blur * blur_mask
    image_out = image_blur * (1 - dir_mask) + mix_color * dir_mask
    

    image_out = image_out.astype(np.uint8)
    return image_out, dir_mask *255


def add_dirtwaterByTxture_slight(image,dir_texture):
    dir_texture = cv2.resize(dir_texture , (image.shape[1],image.shape[0]) )

    dir_texture = dir_texture/255 
    blur_mask = dir_texture.copy()
    blur_mask = (blur_mask  - blur_mask.min() )/ (blur_mask.max() - blur_mask.min() )
    blur_mask[blur_mask>0.2] = 1
    
    
    blur_mask = cv2.GaussianBlur(blur_mask, ksize=(0, 0),sigmaX = 10)
    dir_mask = dir_texture.copy()
    dir_mask = np.minimum(1,(  (dir_mask  - dir_mask.min() )/ (dir_mask.max() - dir_mask.min() ) + 0.2 ) )


    dir_mask = cv2.GaussianBlur(dir_mask, ksize=(0, 0),sigmaX = 8)
    

    water_base_color = np.array([255, 255, 255])  *  ( np.random.random()* 0.5 + 0.5 )
    
    if dir_texture.ndim < 3 :
        
        dir_texture = dir_texture[:,:,np.newaxis]
        dir_mask = dir_mask[:,:,np.newaxis]
        blur_mask = blur_mask[:,:,np.newaxis]
    mean_color_ratio = image.mean()  /255
    alpha = mean_color_ratio
    mix_color = (water_base_color* (1 - alpha)  ) 

    
    image_blur =  cv2.GaussianBlur(image, ksize=(0, 0),sigmaX = 32)
    
    image = cv2.GaussianBlur(image, ksize=(0, 0),sigmaX = 2)
    image_blur = image * (1 - blur_mask) + image_blur * blur_mask
    image_out = image_blur * (1 - dir_mask) + mix_color * dir_mask
    
    

    image_out = image_out.astype(np.uint8)
    return image_out , dir_mask *255








texture_mod = 'big_rain_drop'
#r_fog     r_water_mud  f_water_mud  big_rain_drop  many_rain_drop

texture_mods = ['r_fog', 'thick_fog','r_water_mud',  'f_water_mud',  'big_rain_drop', 'little_rain_drop', 'many_rain_drop','many_dust_drop']

mod = 'water_thick'
'''
"mud", "water_thick", "water_thin"
for water_thick mask,  
       255 * (0~ 0.02) = clearn    
       255 * (0.02~ 0.7 )  = semi_transparent
       255 * (0.7~ 1)  = Opaque

for mud mask,
           255 * (0~ 0.02) = clearn    
           255 * (0.02~ 0.6 )  = semi_transparent
           255 * (0.6~ 1)  = Opaque


'''
image_files = 'image'


from generate_texture_paper import generate_texture

modes = ["mud", "water_thick"]
rands = 10

for rand in range(rands):
    for mod in modes:
        image_save_dir = './' + 'image_' + mod + str(rand) +r'/' 
        mask_save_dir = './' + 'mask_' + mod + str(rand) + r'/' 
        os.makedirs(image_save_dir,exist_ok = True )
        os.makedirs(mask_save_dir,exist_ok = True )
    
    image_file_list = os.listdir(image_files)
    for item in image_file_list:
        image = cv2.imread(os.path.join(image_files,item) , cv2.IMREAD_UNCHANGED  )
        texture_mod = random.choice(texture_mods)
        dir_texture = generate_texture(mod = texture_mod)
        dir_texture = dir_texture[:,:,np.newaxis]
        
        for mod in modes:
            if mod == 'mud':
                image_rgb,mask_rgb = add_mudByTxture(image,dir_texture)
            elif mod == 'water_thick':
                image_rgb,mask_rgb = add_dirtwaterByTxture(image,dir_texture)
            elif mod == 'water_thin':
                image_rgb,mask_rgb = add_dirtwaterByTxture_slight(image,dir_texture)
            image_save_dir = './' + 'image_' + mod + str(rand)  + r'/' 
            mask_save_dir = './' + 'mask_' + mod + str(rand)  + r'/' 
            
            image_name = image_save_dir + item
            cv2.imwrite(image_name,image_rgb)
            mask_name = mask_save_dir + item
            cv2.imwrite(mask_name,mask_rgb)


