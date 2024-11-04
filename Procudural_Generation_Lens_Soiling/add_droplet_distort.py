#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:24:01 2024

@author: 

v1  zs:  add two raindrop distorion effect to mimic light refraction pass through water droplet.
       distort_onedirection : enlarge image in one direction
       enlarege_droplet  : enlarge image in radiation direction 
    
"""

import numpy as np
import pylab as plt

from scipy import ndimage
import cv2
import random
import os

import skimage.transform

def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)








def rand_point(n=3, jitter = 0.05, if_random = 0,random_seed = 5):
    rand_seed = random_seed
    #np.random.seed(rand_seed)
    if if_random == 0:
        
        x = np.linspace(0.2, 0.8,n) 
        y = np.linspace(0.2, 0.8,n)

        X, Y = np.meshgrid(x,y)
        out = np.array([X.reshape((-1)) , Y.reshape((-1))]).T 
        out = out + np.random.uniform(-jitter, jitter,(n*n,2))
    elif if_random == 1:
        out = np.random.random((n*n,2))
    
    return out


def generate_single_droplet_mask(shape=(1000,2000),n_shape = 8, if_round = 2, center=(0.5,0.5), scale = (0.6,0.6),random_seed = 5):
    rand_seed = random_seed
    #np.random.seed(rand_seed)
    

    r = np.random.normal(0.5,0.05,n_shape)

    if if_round == 1:
        r = ndimage.median_filter(r,6)
        #r = ndimage.median_filter(r,6)
    elif if_round == 2:
        r = ndimage.median_filter(r,size = max(2,int(n_shape / 4)  ))
        #r = ndimage.median_filter(r,size = max(2,int(n_shape / 4)  ))
    
    phi = 2 * np.pi * ( np.linspace(0, 1, len(r))  + np.random.random() * 0.5  )

    z =  r * np.exp(1j * phi)
    
    h =  shape[0] 
    w =  shape[1]   
    
    center_x, center_y = center


        
    scale_x , scale_y = scale

    
    center_p = np.array([[center_x * w],[center_y * h]])
    
    scale_droplet  = np.array([[scale_x * w /2 ],[scale_y * h / 2]])
    
    scale_droplet  = np.array([[scale_x * w /2 ],[scale_y * w / 2]])
    
    points = center_p + ( np.array([z.real, z.imag])  ) * scale_droplet
    points = points.astype(np.int64).T


    points_small = center_p + ( np.array([z.real, z.imag])  ) * scale_droplet * 0.8
    points_small = points_small.astype(np.int64).T
    return points , points_small

def droplet_distort_onedirection(image, shape=(1000,2000),center=(0.7,0.7),scale = (0.6,0.6) ,random_seed = 5):
    rand_seed = random_seed
   
    angle =   90 * np.random.random()
    
    
    theta = np.radians(angle)
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([
        [cos,-sin],
        [sin,cos]
        ])
    
    h =  shape[0] 
    w =  shape[1]   
    
    center_x, center_y = center        

    scale_x,scale_y = scale
   
    center_p = np.array([[center_x * w],[center_y * h]]).T 
    
    scale_p  = np.array([[scale_x * w /1.5 ],[scale_y * h / 1.5]])
    
    scale_roi = scale_p[:,0]
    scale_from_point = scale_p[:,0] / 5
    scale_to_point = scale_p[:,0] / 1.5


    roi_points = np.array(  [(-scale_roi[0],-scale_roi[1]),(scale_roi[0],-scale_roi[1]),(scale_roi[0],scale_roi[1]),(-scale_roi[0],scale_roi[1])] ).dot(R) + center_p

    
    from_points = np.array( [(scale_from_point[0],scale_from_point[1]),(-scale_from_point[0],-scale_from_point[1])]   ).dot(R)  + center_p
    to_points = np.array( [(scale_to_point[0],scale_to_point[1]),(-scale_to_point[0],-scale_to_point[1])]  ).dot(R)   + center_p  
    
    
    
    
    from_points = np.concatenate((roi_points, from_points))
    to_points = np.concatenate((roi_points, to_points))


    affin = skimage.transform.PiecewiseAffineTransform()

    affin.estimate(to_points,from_points)
    
    im_array = skimage.transform.warp(image, affin)
    im_array = np.array(im_array * 255., dtype=np.uint8)
    
    
    mask,mask_small = generate_single_droplet_mask(shape=shape ,n_shape = 20, if_round = 2, center= center, scale = scale,random_seed = rand_seed)
    mask = mask[:-1]
    mask_small = mask_small[:-1]
    new_r = np.zeros(image.shape )
    new_r = cv2.fillPoly(new_r,[mask.astype(np.int32)] ,color=(255,255,255)) 
    new_r = cv2.GaussianBlur(new_r, ksize=(0, 0),sigmaX = 20)
    
    new_mask_small = np.zeros(image.shape )
    new_mask_small = cv2.fillPoly(new_mask_small,[mask_small.astype(np.int32)] ,color=(255,255,255)) 
    
    
    new_mask_small_blur = cv2.GaussianBlur(new_mask_small, ksize=(0, 0),sigmaX = np.random.randint(4,10))
    
    out_to_blur = im_array * (new_mask_small_blur / 255)    + image * (1- new_mask_small_blur / 255)      
    
    out_blur = cv2.GaussianBlur(out_to_blur, ksize=(0, 0),sigmaX = np.random.randint(8,20)) 
    
    out3 = out_blur * (new_r / 255)    + image * (1- new_r / 255)
    


    tt = out3.astype(np.uint8)
    return tt, new_r
    
def generate_distort_enlarege(image, shape=(1000,2000),n_shape = 8, if_round = 2, center=(0.5,0.5), scale = (0.6,0.6),random_seed = 5):
    rand_seed = random_seed
  
    r = np.random.normal(0.5,0.05,n_shape)

    if if_round == 1:
        r = ndimage.median_filter(r,6)
    elif if_round == 2:
        r = ndimage.median_filter(r,size = max(2,int(n_shape / 4)  ))

    h =  shape[0] 
    w =  shape[1]   
    
    center_x, center_y = center

    scale_x , scale_y = scale

 
    scale_p  = np.array([[scale_x * w /2 ],[scale_y * h / 2]])
    scale_p  = np.array([[scale_x * w /2 ],[scale_y * w / 2]])
    
    center_p = np.array([[center_x * w],[center_y * h]])    
    phi = 2 * np.pi * ( np.linspace(0, 1, len(r))  + np.random.random()    )

    distor_step = [0.1,0.4,0.6,0.8,1] 

    z_base =  r * np.exp(1j * phi)   
    
    
    z = z_base * distor_step[-1]
    points = center_p + ( np.array([z.real, z.imag])  ) * scale_p
    points = points.astype(np.int64).T

    zo = z_base  * distor_step[0]
    ou_point = center_p + ( np.array([zo.real, zo.imag])  ) * scale_p
    ou_point = ou_point.astype(np.int64).T

    zo1 = z_base * distor_step[2]
    ou_point1 = center_p + ( np.array([zo1.real, zo1.imag])  ) * scale_p
    ou_point1 = ou_point1.astype(np.int64).T


    zroi = z_base * 1.2
    roi_points = center_p + ( np.array([zroi.real, zroi.imag])  ) * scale_p
    roi_points = roi_points.astype(np.int64).T
    
    #  1.2   1   0.6
    #  1.2  0.6  0.1
    from_points = np.concatenate((roi_points, points[:-1], ou_point1[:-1]))

    to_points = np.concatenate((roi_points, ou_point1[:-1], ou_point[:-1]))



    affin = skimage.transform.PiecewiseAffineTransform()

    affin.estimate(from_points,to_points)
    
    im_array = skimage.transform.warp(image, affin)
    im_array = np.array(im_array * 255., dtype=np.uint8)

    mask,mask_small = generate_single_droplet_mask(shape=shape ,n_shape = 20, if_round = 2, center= center, scale = scale,random_seed = rand_seed)
    mask = mask[:-1]
    

    mask_small = mask_small[:-1]
    new_r = np.zeros(image.shape )
    new_r = cv2.fillPoly(new_r,[mask] ,color=(255,255,255)) 
    new_r = cv2.GaussianBlur(new_r, ksize=(0, 0),sigmaX = 20)
    
    new_mask_small = np.zeros(image.shape )
    new_mask_small = cv2.fillPoly(new_mask_small,[mask_small.astype(np.int32)] ,color=(255,255,255)) 
    
    
    new_mask_small_blur = cv2.GaussianBlur(new_mask_small, ksize=(0, 0),sigmaX = np.random.randint(4,10))
    
    out_to_blur = im_array * (new_mask_small_blur / 255)    + image * (1- new_mask_small_blur / 255)      
    
    out_blur = cv2.GaussianBlur(out_to_blur, ksize=(0, 0),sigmaX = np.random.randint(8,20)) 
    
    out3 = out_blur * (new_r / 255)  * 0.9  + image * (1- new_r / 255)
    # 0.9 is for light absortion and decay
    
    tt = out3.astype(np.uint8)

    return tt,new_r
def generate_manydroplet_dis(image,shape=(1000,1000), n_point=5, n_shape = 12, scale_drop = 0.8,mode = 'enlarge',random_seed = 5):
    npoints = rand_point(n=n_point,jitter = 0.2, if_random = 0,random_seed = random_seed) 

    image_out = image.copy()
    mask_out = np.zeros(image.shape )
    i = 0
    
    num_points = np.random.randint(6,20)
    for point in npoints:
        

        max_scale_x = scale_drop 
        max_scale_y = max_scale_x
        scale_x = np.random.uniform(0.2,max_scale_x)
        scale_y = scale_x
        
 
        image_new = image_out.copy()
        
        if mode == 'enlarge' :

            image_out,new_mask_small = generate_distort_enlarege(image_new, shape=shape,center=(point),scale = (scale_x,scale_y) ,random_seed = random_seed)
        elif mode == 'one_direction':
            image_out,new_mask_small = droplet_distort_onedirection(image_new, shape=shape,center=(point),scale = (scale_x,scale_y),random_seed = random_seed )
        elif mode == 'mix_direction':
            if i % 2 == 0:
                image_out,new_mask_small = generate_distort_enlarege(image_new, shape=shape,center=(point),scale = (scale_x,scale_y) ,random_seed = random_seed)
            else:
                image_out,new_mask_small = droplet_distort_onedirection(image_new, shape=shape,center=(point),scale = (scale_x,scale_y),random_seed = random_seed )
        else:
            pass
        mask_out += new_mask_small
        i+=1
        if i == num_points:
            #control number of points. as npoints equals n^2, not random.
            break
    mask_out[mask_out > 250] = 255
    mask_out = mask_out.astype(np.uint8)
    # if you need a mask with blur margins, output 'new_mask_small_blur' in generate_distort_enlarege() or droplet_distort_onedirection()
    return image_out , mask_out

def add_distort(image):
    #image = cv2.imread(image_file)
    mod_list = [0,1,2,3]
    i = random.choice(mod_list)
    if i % 3 == 0:
        mode = 'enlarge' 
    elif i % 3 == 1:
        mode = 'one_direction'
    else:
        mode = 'mix_direction'
    i +=1
    
    rand_seed = np.random.randint(2,10000)
    #np.random.seed(rand_seed)
    #try to give fix droplet size by random seed, but seems not work. will update later.
    num_point = np.random.randint(2,4)
    max_scal = 0.5 + (1 / num_point )
    scale_drop = np.random.uniform(0.2,max_scal)
    n_shape = np.random.randint(12,16)

    image_out , mask_out = generate_manydroplet_dis(image,shape= image.shape, n_point=num_point, n_shape = n_shape, scale_drop = max_scal,mode =mode,random_seed = rand_seed)

    return image_out , mask_out 
mod = 'distort'

image_files = 'image'
image_save_dir = './' + 'image_' + mod + r'/' 
mask_save_dir = './' + 'mask_' + mod + r'/' 
os.makedirs(image_save_dir,exist_ok = False )
os.makedirs(mask_save_dir,exist_ok = False )

#image_file = '2023-07-25-17-20-50-509_CAM_SURROUND_BACK_17089172278569992.jpg'
image_file_list = os.listdir(image_files)


'''

for generate distort mask, wew define that: 
       0~100   = clearn    
       100 ~ 219  = semi_transparent
       220~ 255  = Opaque


'''

for item in image_file_list:
    image = cv2.imread(os.path.join(image_files,item) , cv2.IMREAD_UNCHANGED  )


    image_rgb,mask_rgb = add_distort(image)

    image_name = image_save_dir + item
    cv2.imwrite(image_name,image_rgb)
    mask_name = mask_save_dir + item
    cv2.imwrite(mask_name,mask_rgb)

