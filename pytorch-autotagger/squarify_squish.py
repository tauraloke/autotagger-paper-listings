from fastbook import *
from fastai import *
from fastai.vision import *
from os import listdir
from os.path import isfile, join

dirpath_for_original = '../dataset-for-tagging/original/'
dirpath_for_squared = '../dataset-for-tagging/squish/'
filenames = [f for f in listdir(dirpath_for_original) if isfile(
    join(dirpath_for_original, f))]
for filename in filenames:
    image = Image.open(dirpath_for_original + filename)
    old_size = image.size
    new_x = min(old_size)
    image = PILImage.create(
        dirpath_for_original + filename)
    rsz = Resize(new_x, method=ResizeMethod.Squish)
    rsz(image, split_idx=0).save(dirpath_for_squared + filename+'.bmp')
