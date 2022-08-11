# PointPillars Inference with TensorRT

This repository contains sources and model for [PointPillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.
The model is created with [mmdetection3d](https://github.com/Tartisan/mmdetection3d).

Overall inference has five phases:

- Convert points cloud into 4-channle voxels
- Extend 4-channel voxels to 10-channel voxel features
- Run pfe TensorRT engine to get 64-channel voxel features
- Run rpn backbone TensorRT engine to get 3D-detection raw data
- Parse bounding box, class type and direction

## Model && Data

The demo use the waymo data from Waymo Open Dataset.
The onnx file can be converted by [onnx_tools](https://github.com/Tartisan/mmdetection3d/tree/master/tools/onnx_tools/pointpillars)

### Prerequisites

To build the pointpillars inference, **TensorRT** and **CUDA** are needed.

## Environments

- NVIDIA RTX A4000 Laptop GPU
- CUDA 11.1 + cuDNN 8.2.1 + TensorRT 8.2.3

### Compile && Run

```shell
$ mkdir build && cd build
$ cmake .. && make -j$(nproc)
$ ./demo
```

### Visualization

You should install `open3d` in python environment.

```shell
$ cd tools
$ python viewer.py
```

| trt fp16 | pytorch |
| -------- | ------- |
| ![trt fp16](https://tvax2.sinaimg.cn/large/0080fUsgly1h534cyivy0j31e70qrh62.jpg) | ![pytorch](https://tva3.sinaimg.cn/large/0080fUsgly1h534bzakywj31eg0qn1bm.jpg) |



#### Performance in FP16

```
| Function(unit:ms) | NVIDIA RTX A4000 Laptop GPU |
| ----------------- | --------------------------- |
| Preprocess        | 0.43786  ms                 |
| Pfe               | 3.27231  ms                 |
| Scatter           | 0.085242 ms                 |
| Backbone          | 71.0085  ms                 |
| Postprocess       | 1.79278  ms                 |
| Summary           | 76.601   ms                 |
```

## Note

- The waymo pretrained model in this project is trained only using 4-channel (x, y, z, i), which is different from the mmdetection3d pretrained_model.
- The demo will cache the onnx file to improve performance. If a new onnx will be used, please remove the cache file in "./model".

## References

- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
- [mmdetection3d](https://github.com/Tartisan/mmdetection3d)
- [mmdet_pp](https://github.com/perhapswo/mmdet_pp)
- [CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)
