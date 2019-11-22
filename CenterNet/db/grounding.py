import sys
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
import re
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


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


def cache_bert_feature(bert_model="bert-base-uncased", max_query_len=128, dataset_name=None, label_dir=None, split=None):
    with torch.no_grad():
        textmodel = BertModel.from_pretrained(bert_model).cuda()
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        
        label_file_path = os.path.join(label_dir, dataset_name + "_{}.pth")
        new_label_file_path = os.path.join(label_dir, dataset_name + "-" + bert_model + "-" + str(max_query_len) + "_{}.pth")

        label_file = label_file_path.format(split)
        new_label_file = new_label_file_path.format(split)
        images = torch.load(label_file)
        new_images = []
        for i in range(len(images)):
            image_file, bbox, phrase = images[i]

            ## encode phrase to bert input
            examples = read_examples(phrase.lower(), i)
            features = convert_examples_to_features(examples=examples, seq_length=max_query_len, tokenizer=tokenizer)
            word_id = np.array(features[0].input_ids, dtype=int).reshape((1, -1))
            word_mask = np.array(features[0].input_mask, dtype=int).reshape((1, -1))

            word_id = torch.from_numpy(word_id).cuda()
            word_mask = torch.from_numpy(word_mask).cuda()

            all_encoder_layers, _ = textmodel(word_id, token_type_ids=None, attention_mask=word_mask)
            bert_feature = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:] + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
            
            bert_feature = bert_feature.data.cpu().numpy().flatten()
            new_images.append((image_file, bbox, phrase, bert_feature))

            if (i % (len(images) // 1000)) == 0:
                print("Split {} {}/{}".format(split, i, len(images)))
                print("image_file:", image_file)
                print("bbox:", bbox)
                print("phrase:", phrase)
                print("bert_feature:", bert_feature[:5], "... total of size", len(bert_feature))

        torch.save(new_images, new_label_file)


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


class GROUNDING(DETECTION):
    def __init__(self, db_config, split):
        super(GROUNDING, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir  = system_configs.cache_dir

        self._split = None
        self._dataset = None

        self._label_dir  = None
        self._label_file = None
        self._label_file = None

        self._image_dir  = None
        self._image_file = None

        self._data = "default"

        self.images = None

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

        self._db_inds = None

    def detections_with_phrase(self, idx):
        image_file, bbox, phrase, bert_feature = self.images[idx]  # phrase is not used
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
        return image, bert_feature, detections, phrase

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
