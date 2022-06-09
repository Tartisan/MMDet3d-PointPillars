# PointPillars Inference with TensorRT

This repository contains sources and model for [pointpillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.
The model is created with [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).

Overall inference has five phases:

- Convert points cloud into 4-channle voxels
- Extend 4-channel voxels to 10-channel voxel features
- Run pfe TensorRT engine to get 64-channel voxel features
- Run rpn backbone TensorRT engine to get 3D-detection raw data
- Parse bounding box, class type and direction

## Model && Data

The demo use the waymo data from Waymo Open Dataset.
The onnx file can be converted by [mmdet3d_onnx_tools](https://github.com/speshowBUAA/mmdet3d_onnx_tools)

### Prerequisites

To build the pointpillars inference, **TensorRT** with PillarScatter layer and **CUDA** are needed. PillarScatter layer plugin is already implemented as a plugin for TRT in the demo.

## Environments

- NVIDIA A100-SXM4-40GB
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
$ python tools/viewer.py
```

<center><img src="https://images.weserv.nl/?url=https://article.biliimg.com/bfs/article/690fe472c1dfdb32fbb644487115790cb82f8060.png" width=60%></center>

#### Performance in FP16

```
| Function(unit:ms) | A100-SXM4-40GB |
| ----------------- | -------------- |
| Preprocess        | 0.270888 ms    |
| Pfe               | 2.0598   ms    |
| Scatter           | 0.078447 ms    |
| Backbone          | 11.1322  ms    |
| Postprocess       | 4.00043  ms    |
| Summary           | 17.5418  ms    |
```

## Note

- The waymo pretrained model in this project is trained only using 4-channel (x, y, z, i), which is different from the mmdetection3d pretrained_model.
- The demo will cache the onnx file to improve performance. If a new onnx will be used, please remove the cache file in "./model".

## References

- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [mmdet_pp](https://github.com/perhapswo/mmdet_pp)
- [CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)
