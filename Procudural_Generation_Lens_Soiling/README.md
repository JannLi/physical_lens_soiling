## 0. requrirement: 
	pip install pythonperlin
	pip install skimage


## 1. how to use

put images in 'image' folder and run python script
#add mud and water_thick
	python add_mud.py
	generate 2 dirt type and 4 folder
		mud in folders:  image_mud, mask_mud 
		water_thick in folders: image_water_thick, mask_water_thick, 	   

### add water droplet		
	python add_droplet_distort.py
generate distor type and 2 folder
	folders:  image_distort, mask_distort

### add sun glare
	python add_sun_glare.py
add sun glare effect

### add dust on lens
	python add_lensdust.py
output folder: result_lens_dirt

## 2. how to define mask with transparent thresholds
We provide the raw transparency value of mask and users can choose their own threshold for further application according to their needs
for example, we could define:
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

