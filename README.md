These codes correspond to the ones we use in the paper “Procedural Generation of Lens Soiling Data via Physics-based Simulation”.

Among them, code in folder “Procedural_Generation_Lens_Soiling” is our own code, and codes in folder “lens_dirt_by_augmentation” is the one we use to do comparative experiments with public data augmentation package


    1、 Procedural_Generation_Lens_Soiling 
1.0. requrirement: 
pip install pythonperlin
pip install skimage

1.1.  how to generate lens soiling images and masks

put images in 'image' folder and run python script
    • Adds mud and water soiling.
python add_mud.py
The script adds mud and water soiling. They are kept in same code as they share same imaging process and same shape, just with different parameters.
Default code generate 2 dirt type ”mud, water_thick” and 4 folder
		Mud in folders:  image_mud, mask_mud
		Water_thick in folder:  image_water_thick, mask_water_thick	   
Users can modify the code to generate their own dirty examples, such as:
texture_mods = ['r_fog', 'thick_fog','r_water_mud',  'f_water_mud',  'big_rain_drop', 'little_rain_drop', 'many_rain_drop','many_dust_drop']。 In the default code, we randomly selected these textures.
Besides mud \water_thick, user could also select “water_thin”

    • add water droplet		
pythron add_droplet_distort.py
	generate distor type and 2 folder
		folders:  image_distort, mask_distort

The code generates lens water droplet effect on images with image distortion function. By default, it generates 3 types of distortion in function add_distort(). Each droplet distortion is generated separately, and the distortion function takes up a lot of time, so this feature runs slowly. 
mode = 'enlarge'  #Overall zoom。
mode = 'one_direction'   #one directional zoom
mode = 'mix_direction' # mix zoom. Some water droplet with “Overall zoom” and some with one directional zoom
    • add sun glare
python add_sun_glare.py
	add sun glare effect
This function is to enhance the light source (the part with strong light intensity) in the image, first select the light source by the light intensity threshold, increase its intensity, and then do Gaussian blur, and then mix it with a random texture and add it to the original image.
    • add dust on lens
python add_lensdust.py
	output folder: result_lens_dirt
This feature blurs the scattering of dust, forming a dust mask and then brightening.

1.2. how to define mask with transparent thresholds
We provide the original transparency value of mask and users can choose their own threshold for further application according to their needs.
for example, we could define as follow. They are not defined in the codes here, but defined in the image segmentation codes.
for distort mask,  
       0~100   = clean    
       100 ~ 219  = semi_transparent
       220~ 255  = Opaque

"mud", "water_thick"
for water_thick mask,  
       255 * (0~ 0.02) = clean    
       255 * (0.02~ 0.7 )  = semi_transparent
       255 * (0.7~ 1)  = Opaque

for mud mask,
	       255 * (0~ 0.02) = clean    
       255 * (0.02~ 0.1 )  = semi_transparent
           255 * (0.1~ 1)  = Opaque

    2、 In folder lens_dirt_by_augmentation，we use open-source data augmentation packages to do comparative experiments in In image classification experiments.

2.0.  requrirement: 

pip install albumentations
2.1. how to use
put images in 'image' folder and run python script
	python add_mud_paper.py

or users can modify the code by themselves. as key parts of the code is only one line:

transform = A.Spatter(always_apply=False, p=1.0,mean=(0.65,0.65),std=(0.3,0.3),gauss_sigma=(2,2),intensity=(0.6,0.6),cutout_threshold=(0.68,0.68),mode=[mod])



2.1. citation:
The method uses public data augmentation code to generate mud or rain.

@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
