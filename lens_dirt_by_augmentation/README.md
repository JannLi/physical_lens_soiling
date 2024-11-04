## 0. requrirement

pip install albumentations

## 1. how to use

put images in 'image' folder and run python script
	python add_mud_paper.py

or users can modify the code by themself.
as key parts of the code is only one line:

transform = A.Spatter(always_apply=False, p=1.0,mean=(0.65,0.65),std=(0.3,0.3),gauss_sigma=(2,2),intensity=(0.6,0.6),cutout_threshold=(0.68,0.68),mode=[mod])

## 2. citation
The method use open source data augmentation code to generate mud or rain.

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
