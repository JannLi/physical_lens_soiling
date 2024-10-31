0. requrirement: 

pip install albumentations

1. how to use

put images in 'image' folder and run python script
python add_mud_paper.py
	generate 2 dirt type and 4 folder
		folders:  image_mud, mask_mud
		folders: image_water_thick, mask_water_thick	   
		
pythron droplet_distort_paper.py
	generate distor type and 2 folder
		folders:  image_distort, mask_distort


2. how to use mask
for distort mask,  
       0~100   = clearn    
       100 ~ 219  = semi_transparent
       220~ 255  = Opaque


"mud", "water_thick"
for water_thick mask,  
       255 * (0~ 0.02) = clearn    
       255 * (0.02~ 0.7 )  = semi_transparent
       255 * (0.7~ 1)  = Opaque

for mud mask,
           255 * (0~ 0.02) = clearn    
           255 * (0.02~ 0.1 )  = semi_transparent
           255 * (0.1~ 1)  = Opaque

