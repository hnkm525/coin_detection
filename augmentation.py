import os
import numpy as np
import cv2
from PIL import Image

def gen_aug_img(img, method):
    gen_images = []
    if method == "rotate":
        for i in range(1, 366):
            tmp = img.rotate(i)
            gen_images.append(tmp)
    else:
        pass
    return gen_images

dir_name = "./images"
new_dir_name = "./auged_data"
files = os.listdir(dir_name)
for file in files:
    photo = Image.open(os.path.join(dir_name, file))
    photo_resize = photo.resize((224,224))
    rotate_images = gen_aug_img(photo_resize, "rotate")
    for i, image in enumerate(rotate_images):
        fn = file.split('.')[0] + "_" + str(i+1).zfill(3) + '.jpg'
        image.save(os.path.join(new_dir_name, fn))
    photo_resize.save(os.path.join(new_dir_name, file))