import sys
sys.path.insert(0, "data/coco/PythonAPI/")

import os
import json
import numpy as np
import pickle
import random
import torch
import cv2

from tqdm import tqdm
from db.detection import DETECTION
from config import system_configs


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes, in x1y1x2y2 format
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = max(inter_rect_x2 - inter_rect_x1, 0) * max(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


class FLICKR(DETECTION):
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

        self._image_dir  = os.path.join(data_dir, "flickr", "flickr30k_images")
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "flickr"

        self.images = torch.load(self._label_file)

        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [1]
        self._classes = {1: 1}
        self._coco_to_class_map = {1: 1}

        self._db_inds = np.arange(len(self.images))

    def detections(self, idx):
        image_file, bbox, _, bert_feature = self.images[idx]  # phrase is not used
        ## box format: x1y1x2y2
        bbox = np.array(bbox, dtype=int)
        detections = np.ones((1, 5), dtype=np.float32)
        detections[:, :4] = bbox

        image_path = os.path.join(self._image_dir, image_file)
        image = cv2.imread(image_path)
        ## duplicate channel if gray image
        if image.shape[-1] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.stack([image] * 3)
        return image, bert_feature, detections

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_json(self, all_bboxes):
        detections = []
        for db_ind in all_bboxes:
            for bbox in all_bboxes[db_ind]:
                score = bbox[4]
                gt_bbox = list(map(self._to_float, self.images[db_ind][1]))
                pred_bbox  = list(map(self._to_float, bbox[0:4]))
                detection = {
                    "image_file": self.images[db_ind][0],
                    "phrase": self.images[db_ind][2],
                    "gt_bbox": gt_bbox,
                    "pred_bbox": pred_bbox,
                    "score": float("{:.2f}".format(score))
                }
                detections.append(detection)
        return detections

    def evaluate(self, best_bboxes):
        ious = []
        for db_ind in best_bboxes:
            iou = 0
            if best_bboxes[db_ind] is not None:
                gt_bbox = self.images[db_ind][1]
                pred_bbox = best_bboxes[db_ind][:4]
                iou = bbox_iou(pred_bbox, gt_bbox)
            ious.append(iou)
        ious = np.array(ious)
        print("Evaluation results:")
        print("BBox over 10% = {:f}%".format(100 * sum(ious > 0.1) / len(ious)))
        print("BBox over 20% = {:f}%".format(100 * sum(ious > 0.2) / len(ious)))
        print("BBox over 30% = {:f}%".format(100 * sum(ious > 0.3) / len(ious)))
        print("BBox over 40% = {:f}%".format(100 * sum(ious > 0.4) / len(ious)))
        print("BBox over 50% = {:f}%".format(100 * sum(ious > 0.5) / len(ious)))
        print("BBox over 60% = {:f}%".format(100 * sum(ious > 0.6) / len(ious)))
        print("BBox over 70% = {:f}%".format(100 * sum(ious > 0.7) / len(ious)))
        print("BBox over 80% = {:f}%".format(100 * sum(ious > 0.8) / len(ious)))
        print("BBox over 90% = {:f}%".format(100 * sum(ious > 0.9) / len(ious)))
        return sum(ious > 0.5) / len(ious)
