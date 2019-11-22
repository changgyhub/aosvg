import sys
import os
import json
import numpy as np
import pickle
import random
import torch
import cv2

from tqdm import tqdm
from db.grounding import *
from config import system_configs


class FLICKR(GROUNDING):
    def __init__(self, db_config, split):
        super(FLICKR, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir  = system_configs.cache_dir

        self._split = split
        self._dataset = self._split

        self._label_dir  = os.path.join(data_dir, "flickr")
        self._label_file = os.path.join(self._label_dir, "flickr-" + self._configs["bert_model"] + "-" + str(self._configs["max_query_len"]) + "_{}.pth")
        self._label_file = self._label_file.format(self._dataset)

        if not os.path.exists(self._label_file):
            cache_bert_feature(dataset_name="flickr", label_dir=self._label_dir, split=split)

        self._image_dir  = os.path.join(data_dir, "flickr", "flickr30k_images")
        self._image_file = os.path.join(self._image_dir, "{}")

        self.images = torch.load(self._label_file)

        self._db_inds = np.arange(len(self.images))


class REFCOCO(DETECTION):
    def __init__(self, db_config, split):
        super(REFCOCO, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir  = system_configs.cache_dir

        self._split = split
        self._dataset = self._split

        self._label_dir  = os.path.join(data_dir, "refcoco")
        self._label_file = os.path.join(self._label_dir, "refcoco-" + self._configs["bert_model"] + "-" + str(self._configs["max_query_len"]) + "_{}.pth")
        self._label_file = self._label_file.format(self._dataset)

        if not os.path.exists(self._label_file):
            cache_bert_feature(dataset_name="refcoco", label_dir=self._label_dir, split=split)

        self._image_dir  = os.path.join(data_dir, "refcoco", "images", "mscoco", "images", "train2014")
        self._image_file = os.path.join(self._image_dir, "{}")

        self.images = torch.load(self._label_file)

        self._db_inds = np.arange(len(self.images))


class REFCOCOP(DETECTION):
    def __init__(self, db_config, split):
        super(REFCOCOP, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir  = system_configs.cache_dir

        self._split = split
        self._dataset = self._split

        self._label_dir  = os.path.join(data_dir, "refcoco")
        self._label_file = os.path.join(self._label_dir, "refcoco+-" + self._configs["bert_model"] + "-" + str(self._configs["max_query_len"]) + "_{}.pth")
        self._label_file = self._label_file.format(self._dataset)

        if not os.path.exists(self._label_file):
            cache_bert_feature(dataset_name="refcoco+", label_dir=self._label_dir, split=split)

        self._image_dir  = os.path.join(data_dir, "refcoco", "images", "mscoco", "images", "train2014")
        self._image_file = os.path.join(self._image_dir, "{}")

        self.images = torch.load(self._label_file)

        self._db_inds = np.arange(len(self.images))


class REFCOCOG(DETECTION):
    def __init__(self, db_config, split):
        super(REFCOCOG, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir  = system_configs.cache_dir

        self._split = split
        self._dataset = self._split

        self._label_dir  = os.path.join(data_dir, "refcoco")
        self._label_file = os.path.join(self._label_dir, "refcocog-" + self._configs["bert_model"] + "-" + str(self._configs["max_query_len"]) + "_{}.pth")
        self._label_file = self._label_file.format(self._dataset)

        if not os.path.exists(self._label_file):
            cache_bert_feature(dataset_name="refcocog", label_dir=self._label_dir, split=split)

        self._image_dir  = os.path.join(data_dir, "refcoco", "images", "mscoco", "images", "train2014")
        self._image_file = os.path.join(self._image_dir, "{}")

        self.images = torch.load(self._label_file)

        self._db_inds = np.arange(len(self.images))


class REFERIT(DETECTION):
    def __init__(self, db_config, split):
        super(REFERIT, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir  = system_configs.cache_dir

        self._split = split
        self._dataset = self._split

        self._label_dir  = os.path.join(data_dir, "referit")
        self._label_file = os.path.join(self._label_dir, "referit-" + self._configs["bert_model"] + "-" + str(self._configs["max_query_len"]) + "_{}.pth")
        self._label_file = self._label_file.format(self._dataset)

        if not os.path.exists(self._label_file):
            cache_bert_feature(dataset_name="referit", label_dir=self._label_dir, splits=["train", "val", "test"])

        self._image_dir  = os.path.join(data_dir, "referit", "images")
        self._image_file = os.path.join(self._image_dir, "{}")

        self.images = torch.load(self._label_file)

        self._db_inds = np.arange(len(self.images))


datasets = {
    "FLICKR": FLICKR,
    "REFCOCO": REFCOCO,
    "REFCOCOP": REFCOCOP,
    "REFCOCOG": REFCOCOG,
    "REFERIT": REFERIT
}
