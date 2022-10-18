## Summary
This repository contains attempts to adapt the [PyTorch autotagger model](https://github.com/danbooru/autotagger) in TensorflowJS + Node environment.

## Converting a weight file from PyTorch to Tensorflow
- Exporting *.pth file to ONNX (`torch.onnx.export(learn.eval().to('cuda'), torch.randn(1, 3, 224, 224).to('cuda'), "danbooru.onnx")`)
- Using [onnx-tf](https://github.com/onnx/onnx-tensorflow): `$ onnx-tf convert -i danbooru.onnx -o danbooru`
- Next stage is converting *.pb to a package for browsers, but I failed it. :( BTW *.pb model is good for my Electron pet project so I don't care about that.

## Requirements
- Node 16
- Python 3.9

