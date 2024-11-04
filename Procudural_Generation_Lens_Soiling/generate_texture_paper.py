#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:24:01 2024

@author: zs

zs:  add big rain texture one by one.
    add little_rain_drop
    add big_rain_drop
    add many_rain_drop
    
"""

import numpy as np
import pylab as plt
from pythonperlin import perlin, extend2d
import imageio
from scipy.ndimage import gaussian_filter
from scipy import signal,ndimage
import cv2

def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)



def generate_single_droplet(shape=(1000,2000),n_shape = 8, if_round = 0, center=(0.5,0.5), scale = (0.6,0.6)):

    

    r = np.random.normal(0.5,0.2,n_shape)
    
    #b, a = signal.butter(3,0.05)
    #r = signal.filtfilt(b, a, r)
    if if_round == 1:
        r = ndimage.median_filter(r,2)
    elif if_round == 2:
        r = ndimage.median_filter(r,size = max(2,int(n_shape / 4)  ))

    
    phi = 2 * np.pi * ( np.linspace(0, 1, len(r))  + np.random.random()   )

    z =  r * np.exp(1j * phi)
    
    h =  shape[0] 
    w =  shape[1]   
    
    center_x, center_y = center


        
    scale_x , scale_y = scale

    
    center = np.array([[center_x * w],[center_y * h]])
    
    scale  = np.array([[scale_x * w /2 ],[scale_y * h / 2]])
    #scale  = np.array([[scale_x * w  ],[scale_y * w ]])
    #points = center + ( np.array([z.real, z.imag]) -  np.array(  [ [z.min().real, z.min().imag]]  ).T   / abs(z.max() -z.min()  ) ) * scale 
    #points = center + ( np.array([z.real, z.imag]) -  np.array(  [ [(z.max() -z.min()).real, (z.max() -z.min()).imag]]  ).T   / abs(z.max() -z.min()  ) ) * scale 
    points = center + ( np.array([z.real, z.imag])  ) * scale
    points = points.astype(np.int64).T

    return points
#tt,points = generate_single_droplet(shape=(1000,1000),n_shape =12, if_round = 2, center=(0.5,0.5), scale = (0.6,0.6))


def rand_point(n=3, jitter = 0.05, if_random = 0):
    
    if if_random == 0:
        
        x = np.linspace(0.2, 0.7,n)
        y = np.linspace(0.2, 0.7,n)

        X, Y = np.meshgrid(x,y)
        out = np.array([X.reshape((-1)) , Y.reshape((-1))]).T 
        out = out + np.random.uniform(-jitter, jitter,(n*n,2))
    elif if_random == 1:
        out = np.random.random((n*n,2))
    
    return out

def generate_manydroplet(shape=(1000,1000), n_point=5, n_shape = 12, scale_min = 0.8):
    #the function could also be used as lens dust mask
    npoints = rand_point(n=n_point,jitter = 0.2, if_random = 0) 
    h =  shape[0] 
    w =  shape[1]  
    tt = np.zeros((h,w)) 
    for point in npoints:
        
        #scale_x = min(scale_min, max(0.02, np.random.random()  ) )
        scale_x = np.random.uniform(0.02,scale_min )
        scale_y = scale_x
        poly_point = generate_single_droplet(shape=shape,n_shape = n_shape, if_round = 2, center=(point), scale = (scale_x,scale_y))
        color = int(255 * np.random.random())
        tt = cv2.fillPoly(tt,[poly_point[:-1]] ,color=(color,color,color))
    
    detail_texture =  np.random.normal(0,1.5,((h,w)))
    tt = tt * detail_texture
    tt = tt.astype(np.uint8)
    

    rgb8 = cv2.GaussianBlur(tt, ksize=(0, 0),sigmaX = 10)
    
    rgb8 = rgb8.astype(np.uint8)
    
    #plt.imshow( rgb8)
    return rgb8
    

def generate_runing_droplet(shape=(1000,1000), n_point=5, n_shape = 12, scale_min = 0.8):
    npoints = rand_point(n=n_point,jitter = 0.2, if_random = 0) 
    
    
    h =  shape[0] 
    w =  shape[1]  
    tt = np.zeros((h,w)) 
    for point in npoints:
        if n_point < 4:
            scale_x = min(1.4, max(scale_min, np.random.random() * 1.5  ) )
        elif n_point < 8:
            scale_x = min(0.6, max(scale_min, np.random.random()  ) )
        else:            
            scale_x = min(0.2, max(scale_min, np.random.random()  ) )
            
        scale_y = scale_x
        poly_point = generate_single_droplet(shape=shape,n_shape = n_shape, if_round = 2, center=(point), scale = (scale_x,scale_y))
        tt = cv2.fillPoly(tt,[poly_point] ,color=(255,255,255))
    tt = tt.astype(np.uint8)
    detail_texture =  np.random.normal(0,1,((h,w)))
    
    
    
    dist = cv2.distanceTransform(tt, cv2.DIST_L2, 5) 
    
    dist = dist * detail_texture
    
    dist = dist * 255 /  dist.max()
    rgb8 = cv2.GaussianBlur(dist, ksize=(0, 0),sigmaX = 10)
    rgb8 = rgb8 * 255 /  rgb8.max()
    rgb8 = rgb8.astype(np.uint8)
    
    #plt.imshow( tt)
    return rgb8

def generate_droplet(shape=(1000,1000), n_point=5, n_shape = 12, scale_min = 0.8):
    npoints = rand_point(n=n_point,jitter = 0.2, if_random = 0) 
    
    
    h =  shape[0] 
    w =  shape[1]  
    tt = np.zeros((h,w)) 
    for point in npoints:
        if n_point < 4:
            scale_x = min(1.4, max(scale_min, np.random.random() * 1.5  ) )
        elif n_point < 8:
            scale_x = min(0.6, max(scale_min, np.random.random()  ) )
        else:            
            scale_x = min(0.2, max(scale_min, np.random.random()  ) )
            
        scale_y = scale_x
        poly_point = generate_single_droplet(shape=shape,n_shape = n_shape, if_round = 2, center=(point), scale = (scale_x,scale_y))
        tt = cv2.fillPoly(tt,[poly_point] ,color=(255,255,255))
    tt = tt.astype(np.uint8)
    detail_texture =  np.random.normal(0.8,1,((h,w)))
    
    
    
    dist = cv2.distanceTransform(tt, cv2.DIST_L2, 5) 
    
    dist = dist * detail_texture
    
    dist = dist * 255 /  dist.max()
    rgb8 = cv2.GaussianBlur(dist, ksize=(0, 0),sigmaX = 15)
    rgb8 = rgb8 * 255 /  rgb8.max()
    rgb8 = rgb8.astype(np.uint8)
    
    #plt.imshow( tt)
    return rgb8
    


def generate_round(shape=(4,8),dens=8,extend_n=32,scale=1.5,if_big_round = 0):

    
    mul = 4
    shape_noise = (shape[0] * mul,shape[1] * mul)
    #x = perlin(shape_noise, dens=dens,octaves=8)
    x = perlin(shape_noise, dens=dens)
    delta = 2
    
    r = x[delta] + 1
    r = np.concatenate([r, (r[0],)])
    #r = np.random.normal(0.5,0.1,128)
    
    #b, a = signal.butter(3,0.05)
    #r = signal.filtfilt(b, a, r)
    #r = ndimage.median_filter(r,size = 8)
    
    if if_big_round == 0:
        r = np.convolve(r, np.ones((4))/4,mode='full')
    else:
        r = np.convolve(r, np.ones((20))/20,mode='full')
    
    phi = 2 * np.pi * ( np.linspace(0, 1, len(r))  + np.random.random()   )

    z =  r * np.exp(1j * phi)
    
    h =  shape[0] * dens * extend_n
    w =  shape[1] * dens * extend_n  
    
    off_x = 0.5
    off_y = 0.5
    
    if if_big_round  == 0:
        
        scale_x = 0.6
        scale_y = 0.6
    else:
        scale_x = 0.8
        scale_y = 0.8
    
    offset = np.array([[off_x * w],[off_y * h]])
    
    scale  = np.array([[scale_x * w /2 ],[scale_y * h / 2]])
    
    #points = offset + ( np.array([z.real, z.imag]) -z.real.min() / (z.real.max() -z.real.min()  ) ) * scale 
    
    #points =  ( np.array([z.real, z.imag]) -z.real.min() / (z.real.max() -z.real.min()  ) ) * scale 
    points =   np.array([z.real, z.imag]) * scale + offset 
    points = points.astype(np.int64).T
    

    #t = np.zeros((h,w,3))
    t = np.zeros((h,w))
    tt = cv2.fillPoly(t,[points[:-1]] ,color=(255,255,255))
    
    tt = cv2.GaussianBlur(tt, ksize=(0, 0),sigmaX = 20)
    
    #plt.imshow( tt)
    return tt


def generate_flower(shape=(4,8),dens=8,extend_n=32,scale=1.5):
    
    mul = 2
    shape_noise = (shape[0] * mul,shape[1] * mul)
    x = perlin(shape_noise, dens=dens,octaves=2)
    
    delta = dens
    
    r = x[delta] + 1
    r = np.concatenate([r, (r[0],)])
    phi = 2 * np.pi * np.linspace(0, 1, len(r))

    z =  r * np.exp(1j * phi)
    
    h =  shape[0] * dens * extend_n
    w =  shape[1] * dens * extend_n  
    
    offset = np.array([w / 2,h / 2])[:,np.newaxis]
    
    scale  = 1.5
    
    points = offset + ( np.array([z.real, z.imag]) * scale * offset / (z.real.max() -z.real.min()  ) )
    points = points.astype(np.int64).T
    

    #t = np.zeros((h,w,3))
    t = np.zeros((h,w))
    tt = cv2.fillPoly(t,[points] ,color=(255,255,255))
    
    
    
    
    return tt


def generate_rain(thick_fog=0):
    pass

def generate_fog(shape=(4,8),dens=8,extend_n=32,thick_fog=0,mud_level = 1,if_round = 1,scale=1.5,if_big_round=0):

    mul = 2
    shape = (shape[0] * mul,shape[1] * mul)
    extend_n = int(extend_n / mul)
    x = perlin(shape, dens=dens,  octaves=4)
   
    if thick_fog:
        x += abs(perlin(shape, dens=dens,  octaves=4))

    h =  shape[0] * dens * extend_n
    w =  shape[1] * dens * extend_n  
    
    
    x_e = extend2d(x,n=extend_n,kind='cubic',mode = 'full')
    x_e = x_e[:h,:w]
    
    clip = max(-0.2,min(0.2,mud_level/20)  )
    
    rgb8 = to8b(x_e*0.5 + clip)
    
    
    rgb8 = to8b(x_e*0.5+0.5)
    if( if_round == 1):
        round_mask = generate_flower(shape,dens,extend_n,scale=scale)
        
        rgb8= rgb8 * round_mask[:,:] / 255
        rgb8 = rgb8.astype(np.uint8)
    elif (if_round == 2):
        round_mask = generate_round(shape,dens,extend_n,scale=scale,if_big_round=1)
        
        rgb8= rgb8 * round_mask[:,:] / 255
        rgb8 = rgb8.astype(np.uint8)
        
        
    return rgb8
#rgb8 = generate_fog()

def generate_mud_waterfog_texture(shape=(4,8),dens=8,extend_n=32,mud_level = 1,if_round = 1,scale=1.5):
    
    x = perlin(shape, dens=dens, octaves=1)

    x_e = extend2d(x,n =extend_n,kind='cubic',mode = 'full')
    
    
    h =  shape[0] * dens * extend_n
    w =  shape[1] * dens * extend_n  
    
    x_e = x_e[:h,:w]
    
    
    
    clip = max(-0.2,min(0.2,mud_level/20)  )
    
    rgb8 = to8b(x_e*0.5 + clip)
    
    if( if_round == 1):
        round_mask = generate_flower(shape,dens,extend_n,scale=scale)
        
        rgb8= rgb8 * round_mask[:,:] / 255
        rgb8 = rgb8.astype(np.uint8)
    elif (if_round == 2):
        round_mask = generate_round(shape,dens,extend_n,scale=scale)
        
        rgb8= rgb8 * round_mask[:,:] / 255
        rgb8 = rgb8.astype(np.uint8)
    
    
    rgb8_mask = rgb8.copy()
    rgb8_mask[rgb8_mask>0] = 255
    rgb8_mask = rgb8_mask.astype(np.uint8)
    return rgb8,rgb8_mask

    
mod = 'big_rain_drop'
# r_water_mud:  for water and mud
# x_fog: for different fog

mod = 'many_dust_drop'
save_dir = './fog_r_mask/'



def generate_texture(mod = 'f_fog'):
    if mod == 'r_fog':
        rgb8 = generate_fog(mud_level=10,if_round=2)

    elif mod == 'f_fog':
        rgb8 = generate_fog(mud_level=10,if_round=0)

    elif mod == 'thick_fog':
        rgb8 = generate_fog(thick_fog=1,mud_level=10,if_round=2,if_big_round=1)

    elif mod == 'r_water_mud':
        rgb8,rgb8_mask = generate_mud_waterfog_texture(mud_level=1,if_round=1)

    elif mod == 'f_water_mud':
        rgb8,rgb8_mask = generate_mud_waterfog_texture(mud_level=1,if_round=0)

    elif mod == 'little_rain_drop':
        rgb8 = generate_droplet(shape=(768,1024), n_point=np.random.randint(4,12),n_shape =np.random.randint(5,10),scale_min = 0.4)

    elif mod == 'big_rain_drop':
        rgb8 = generate_droplet(shape=(768,1024), n_point=np.random.randint(2,4),n_shape =np.random.randint(10,16),scale_min = 0.8)

    elif mod == 'many_rain_drop':
        rgb8 = generate_manydroplet(shape=(768,1024), n_point=np.random.randint(8,10),n_shape =np.random.randint(10,32),scale_min = 0.1)

    
    elif mod == 'many_dust_drop':
        rgb8 = generate_manydroplet(shape=(768,1024), n_point=np.random.randint(4,6),n_shape =np.random.randint(10,32),scale_min = 0.05)
    
    else:
        rgb8 = None
    return rgb8

if __name__ == "__main__":
    n = 10
    for i in range(n):
        if mod == 'r_fog':
            rgb8 = generate_fog(mud_level=10,if_round=2)
            save_dir = './r_fog/'
            cv2.imwrite(save_dir + str(i) +'r_fog.jpg',rgb8)
        elif mod == 'f_fog':
            rgb8 = generate_fog(mud_level=10,if_round=0)
            save_dir = './f_fog/'
            cv2.imwrite(save_dir + str(i) +'f_fog.jpg',rgb8)
        elif mod == 'thick_fog':
            rgb8 = generate_fog(thick_fog=1,mud_level=10,if_round=2,if_big_round=1)
            save_dir = './thick_fog/'
            cv2.imwrite(save_dir + str(i) +'thick_fog.jpg',rgb8)
        elif mod == 'r_water_mud':
            rgb8,rgb8_mask = generate_mud_waterfog_texture(mud_level=1,if_round=1)
            save_dir = './r_water_mud/'
            cv2.imwrite(save_dir + str(i)+'r_water_mud.jpg',rgb8)
        elif mod == 'f_water_mud':
            rgb8,rgb8_mask = generate_mud_waterfog_texture(mud_level=1,if_round=0)
            save_dir = './f_water_mud/'
            cv2.imwrite(save_dir + str(i)+'f_water_mud.jpg',rgb8)
        elif mod == 'little_rain_drop':
            rgb8 = generate_droplet(shape=(768,1024), n_point=np.random.randint(4,12),n_shape =np.random.randint(5,10),scale_min = 0.4)
            save_dir = './little_rain_drop/'
            cv2.imwrite(save_dir + str(i) +'little_rain_drop.jpg',rgb8)
        elif mod == 'big_rain_drop':
            rgb8 = generate_droplet(shape=(768,1024), n_point=np.random.randint(2,4),n_shape =np.random.randint(10,16),scale_min = 0.8)
            save_dir = './big_rain_drop/'
            cv2.imwrite(save_dir + str(i) +'big_rain_drop.jpg',rgb8)
        elif mod == 'many_rain_drop':
            rgb8 = generate_manydroplet(shape=(768,1024), n_point=np.random.randint(8,10),n_shape =np.random.randint(10,32),scale_min = 0.1)
            save_dir = './many_rain_drop/'
            cv2.imwrite(save_dir + str(i) +'many_rain_drop.jpg',rgb8)
        
        elif mod == 'many_dust_drop':
            rgb8 = generate_manydroplet(shape=(768,1024), n_point=np.random.randint(4,6),n_shape =np.random.randint(10,32),scale_min = 0.05)
            save_dir = './many_dust_drop/'
            cv2.imwrite(save_dir + str(i) +'many_dust_drop.jpg',rgb8)

