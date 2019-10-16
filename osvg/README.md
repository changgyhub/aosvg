# Attended One-Stage Visual Grounding

Chang Gao, Yongxin Wang, Zhiyu Min, Yujia Chen

## Installation

1. Setup python environment

```bash
conda create -n aosvg python=3.5 anaconda
source activate aosvg

conda install pytorch=0.4.1 -c pytorch
conda install torchvision
pip install -r requirements.txt
```

2. Prepare the submodules and associated data

We temporally use data and model from "A Fast and Accurate One-Stage Approach to Visual Grounding".

* Flickr30K Entities Dataset: place the data or the soft link of dataset folder under ``./ln_data/``. The formated Flickr data is availble at [[Gdrive]](https://drive.google.com/open?id=1A1iWUWgRg7wV5qwOP_QVujOO4B8U-UYB), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/Eqgejwkq-hZIjCkhrgWbdIkB_yi3K4uqQyRCwf9CSe_zpQ?e=dtu8qF).

* Data index: download the generated index files and place them as the ``./data`` folder. Availble at [[Gdrive]](https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/Epw5WQ_mJ-tOlAbK5LxsnrsBElWwvNdU7aus0UIzWtwgKQ?e=XHQm7F).

* Model weights: download the pretrained model of [Yolov3](https://pjreddie.com/media/files/yolov3.weights) and [language models](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/ErrXDnw1igFGghwbH5daoKwBX4vtE_erXbOo1JGnraCE4Q?e=tQUCk7). Place the files in ``./saved_models``.

### Training
3. Train the model, run the code under main folder. 
Using flag ``--lstm`` to access lstm encoder, Bert is used as the default. 
Using flag ``--light`` to access the light model.

```bash
python train_yolo.py --data_root ./ln_data/ --dataset flickr \
  --gpu gpu_id --batch_size 32 --resume saved_models/lstm_flickr_model.pth.tar \
  --lr 1e-4 --nb_epoch 100 --lstm
```

4. Evaluate the model, run the code under main folder. 
Using flag ``--test`` to access test mode.

```bash
python train_yolo.py --data_root ./ln_data/ --dataset flickr \
  --gpu gpu_id --resume saved_models/lstm_flickr_model.pth.tar \
  --lstm --test
```

5. Visulizations. Flag ``--save_plot`` will save visulizations.


### Credits

Based on

    @inproceedings{yang2019fast,
      title={A Fast and Accurate One-Stage Approach to Visual Grounding},
      author={Yang, Zhengyuan and Gong, Boqing and Wang, Liwei and Huang
        , Wenbing and Yu, Dong and Luo, Jiebo},
      booktitle={ICCV},
      year={2019}
    }

Part of the code or models are from 
[One-Stage Visual Grounding](https://github.com/zyang-ur/onestage_grounding),
[DMS](https://github.com/BCV-Uniandes/DMS),
[MAttNet](https://github.com/lichengunc/MAttNet),
[Yolov3](https://pjreddie.com/darknet/yolo/) and
[Pytorch-yolov3](https://github.com/eriklindernoren/PyTorch-YOLOv3).
