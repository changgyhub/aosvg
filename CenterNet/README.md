# [CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/abs/1904.08189)

## Preparation

GCC 4.9.4 is required. See [this](https://github.com/princeton-vl/CornerNet/issues/47) if segmenation fault occurs when corner pool is called.

Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.

```bash
conda create --name CenterNet --file conda_packagelist.txt
```

After you create the environment, activate it.

```bash
source activate CenterNet
```

Also install bert with

```bash
pip install pytorch-pretrained-bert
```

## Compiling Corner Pooling Layers

```bash
cd <CenterNet dir>/models/py_utils/_cpools/
python setup.py install --user
```

## Compiling NMS

```bash
cd <CenterNet dir>/external
make
```

## Preprocessing

Run the following command to preprocess phrases into bert feature vectors

```bash
python preprocess.py
```

The default saved files are at `data/flickr/` with names as `flickr-{bert_model}-{max_query_len}_{split}.pth`.

## Training and Evaluation

We provide the configuration file (`CenterNet-52.json`) and the model file (`CenterNet-52.py`) for CenterNet in this repo.
To train CenterNet-52:

```bash
python -u train.py CenterNet-52 | tee logs/<file name>
```

To test CenterNet-52:

```bash
python test.py CenterNet-52 --testiter <iter> --split <split>  (--debug)
```

To test CenterNet-52 in multi-scale:

```bash
python test.py CenterNet-52 --testiter <iter> --split <split>  --suffix multi_scale (--debug)
```
