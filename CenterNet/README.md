# [CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/abs/1904.08189)

## Preparation
GCC 4.9.4 is required. See [this](https://github.com/princeton-vl/CornerNet/issues/47) if segmenation fault occurs when corner pool is called.

Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.
```
conda create --name CenterNet --file conda_packagelist.txt
```

After you create the environment, activate it.
```
source activate CenterNet
```

## Compiling Corner Pooling Layers
```
cd <CenterNet dir>/models/py_utils/_cpools/
python setup.py install --user
```

## Compiling NMS
```
cd <CenterNet dir>/external
make
```

## Installing MS COCO APIs
```
cd <CenterNet dir>/data/coco/PythonAPI
make
```

## Downloading MS COCO Data
- Download the training/validation split we use in our paper from [here](https://drive.google.com/file/d/1dop4188xo5lXDkGtOZUzy2SHOD_COXz4/view?usp=sharing) (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data))
- Unzip the file and place `annotations` under `<CenterNet dir>/data/coco`
- Download the images (2014 Train, 2014 Val, 2017 Test) from [here](http://cocodataset.org/#download)
- Create 3 directories, `trainval2014`, `minival2014` and `testdev2017`, under `<CenterNet dir>/data/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

## Training and Evaluation
We provide the configuration file (`CenterNet-52.json`) and the model file (`CenterNet-52.py`) for CenterNet in this repo.
To train CenterNet-52-aircraft:
```
python -u train.py CenterNet-52 | tee logs/<file name>
```
To test CenterNet-52:
```
python test.py CenterNet-52 --testiter <iter> --split <split>  (--debug)
```
To test CenterNet-52 in multi-scale:
```
python test.py CenterNet-52 --testiter <iter> --split <split>  --suffix multi_scale (--debug)
```
