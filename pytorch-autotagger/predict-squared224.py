# -*- mode: python ; coding: utf-8 -*-

from fastbook import *
from pandas import DataFrame, read_csv
from fastai.imports import noop
from fastai.callback.progress import ProgressCallback
from os import listdir
from os.path import isfile, join

model_path = "models/model.pth"
data_path = "test/tags.csv.gz"
tags_path = "data/tags.json"
df = read_csv(data_path)
vocab = json.load(open(tags_path))
threshold = 0.01
limit = 50
bs = 64
dirpath = '../dataset-for-tagging/squared224/'

dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock(vocab=vocab)),
    get_x=lambda df: Path("test") / df["filename"],
    get_y=lambda df: df["tags"].split(" "),
    item_tfms=Resize(224, method=ResizeMethod.Squish),
    batch_tfms=[RandomErasing()]
)

dls = dblock.dataloaders(df)
learn = vision_learner(dls, "resnet152", pretrained=False)
model_file = open(model_path, "rb")
learn.load(model_file, with_opt=False)
learn.remove_cb(ProgressCallback)

filepaths = [dirpath+f for f in listdir(dirpath) if isfile(join(dirpath, f))]
tags = {}
for filepath in filepaths:
    dl = learn.dls.test_dl([PILImage.create(filepath)], bs=bs)
    batch, _ = learn.get_preds(dl=dl)
    for scores in batch:
        df = DataFrame({"tag": learn.dls.vocab, "score": scores})
        df = df[df.score >= threshold].sort_values(
            "score", ascending=False).head(limit)
        tags[filepath] = dict(zip(df.tag, df.score))

print(tags)

# torch.onnx.export(learn.eval().to('cuda'), torch.randn(
#    1, 3, 512, 512).to('cuda'), "danbooru-512.onnx")
