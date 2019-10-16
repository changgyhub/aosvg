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

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes, in x1y1x2y2 format
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = np.max(b1_x1, b2_x1)
    inter_rect_y1 = np.max(b1_y1, b2_y1)
    inter_rect_x2 = np.min(b1_x2, b2_x2)
    inter_rect_y2 = np.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.max(inter_rect_x2 - inter_rect_x1, 0) * np.max(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def letterbox(image, mask, height, color=(123.7, 116.3, 103.5)):  # resize a rectangular image to a padded square
    shape = image.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    if mask is not None:
        mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)  # resized, no border
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)  # padded square
    return image, mask, ratio, dw, dh


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class FLICKR(DETECTION):
    def __init__(self, db_config, split):
        super(FLICKR, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir  = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "train": "flickr_train",
            "val": "flickr_val",
            "test": "flickr_test"
        }[self._split]
        
        self._label_dir  = os.path.join(data_dir, "flickr")
        self._label_file = os.path.join(self._label_dir, "{}.pth")
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir  = os.path.join(data_dir, "flickr", "flickr30k_images")

        self._data = "flickr"

        self.images = []
        self.images += torch.load(self._label_file)
        
        self.seq_length = self._configs["max_query_len"]
        self.bert_model = self._configs["bert_model"]

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)

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
    
    def pull_item(self, idx):
        image_file, bbox, phrase = self.images[idx]
        ## box format: x1y1x2y2
        bbox = np.array(bbox, dtype=int)
        image_path = os.path.join(self._image_dir, image_file)
        print("======", image_path)
        image = cv2.imread(image_path)
        ## duplicate channel if gray image
        if image.shape[-1] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.stack([image] * 3)
        return image, phrase, bbox

    def detections(self, idx):
        image, phrase, bbox = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()

        ## encode phrase to bert input
        examples = read_examples(phrase, idx)
        features = convert_examples_to_features(examples=examples, seq_length=seq_length, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        
        detections = np.ones((1, 5), dtype=np.float32)
        detections[:, :4] = bbox
        return image, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), detections

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_json(self, all_bboxes):
        detections = []
        for db_ind in all_bboxes:
            for bbox in all_bboxes[db_ind]:
                score = bbox[4]
                bbox  = list(map(self._to_float, bbox[0:4]))
                detection = {
                    "image_file": self.images[db_ind][0],
                    "phrase": self.images[db_ind][2],
                    "gt_bbox": self.images[db_ind][1],
                    "pred_bbox": bbox,
                    "score": float("{:.2f}".format(score))
                }
                detections.append(detection)
        return detections

    def evaluate(self, best_bboxes):
        acc = 0.0
        for db_ind in best_bboxes:
            gt_bbox = self.images[db_ind][1]
            pred_bbox = best_bboxes[db_ind][:4]
            iou = bbox_iou(pred_bbox, gt_bbox, x1y1x2y2=True)
            if iou > 0.5:
                acc += 1.0
        acc = acc / len(self._db_inds)
        print("BBox accuracy = {:f}%".format(100 * acc))
        return acc
