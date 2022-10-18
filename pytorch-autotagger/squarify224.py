from PIL import Image
from os import listdir
from os.path import isfile, join

dirpath_for_original = '../dataset-for-tagging/original/'
dirpath_for_squared = '../dataset-for-tagging/squared224/'
filenames = [f for f in listdir(dirpath_for_original) if isfile(
    join(dirpath_for_original, f))]
for filename in filenames:
    image = Image.open(dirpath_for_original +
                       filename).resize(tuple([224, 224]), Image.Resampling.LANCZOS)
    image.save(dirpath_for_squared + filename)
