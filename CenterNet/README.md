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

Also install bert with
```
pip install pytorch-pretrained-bert
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
