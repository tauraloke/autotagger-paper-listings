from fastbook import *
from fastai import *
from fastai.vision import *
from os import listdir
from os.path import isfile, join

dirpath_for_original = '../dataset-for-tagging/original/'
dirpath_for_squared = '../dataset-for-tagging/squish224/'
filenames = [f for f in listdir(dirpath_for_original) if isfile(
    join(dirpath_for_original, f))]
for filename in filenames:
    image = PILImage.create(
        dirpath_for_original + filename)
    rsz = Resize(224, method=ResizeMethod.Squish)
    rsz(image, split_idx=0).save(dirpath_for_squared + filename+'.bmp')
