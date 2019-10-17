import os
import cv2
import pdb
import json
import copy
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms, soft_nms_merge

bbox_color = np.random.rand(3)

def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    tx_inds = xs[:,:,0] <= -5
    bx_inds = xs[:,:,1] >= sizes[0,1]+5
    ty_inds = ys[:,:,0] <= -5
    by_inds = ys[:,:,1] >= sizes[0,0]+5
    
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)
    detections[:,tx_inds[0,:],4] = -1
    detections[:,bx_inds[0,:],4] = -1
    detections[:,ty_inds[0,:],4] = -1
    detections[:,by_inds[0,:],4] = -1

def save_image(data, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi = height)
    plt.close()

def kp_decode(nnet, inputs,  K, ae_threshold=0.5, kernel=3):
    detections, center = nnet.test(inputs, ae_threshold=ae_threshold, K=K, kernel=kernel)
    detections = detections.data.cpu().numpy()
    center = center.data.cpu().numpy()
    return detections, center

def kp_detection(db, nnet, result_dir, debug=False, decode_func=kp_decode, partial=False):
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    db_inds = db.db_inds[:100] if partial else db.db_inds

    K             = db.configs["top_k"]
    ae_threshold  = db.configs["ae_threshold"]
    nms_kernel    = db.configs["nms_kernel"]
    
    scales        = db.configs["test_scales"]
    weight_exp    = db.configs["weight_exp"]
    merge_bbox    = db.configs["merge_bbox"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]

    seq_length  = db.configs["max_query_len"]
    bert_model  = db.configs["bert_model"]
    textdim = 768 if bert_model == 'bert-base-uncased' else 1024

    top_bboxes = {}
    best_bboxes = {}
    for ind in tqdm(range(db_inds.size), ncols=80, desc="locating kps"):

        db_ind = db_inds[ind]
        image_file = db.images[db_ind][0]

        image, bert_feature, _ = db.detections(db_ind)

        height, width = image.shape[0:2]

        detections = []
        center_points = []

        for scale in scales:
            new_height = int(height * scale)
            new_width  = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            inp_height = new_height | 127
            inp_width  = new_width  | 127

            images           = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            bert_features    = np.zeros((1, textdim), dtype=np.float32)
            ratios           = np.zeros((1, 2), dtype=np.float32)
            borders          = np.zeros((1, 4), dtype=np.float32)
            sizes            = np.zeros((1, 2), dtype=np.float32)

            bert_features[0] = bert_feature

            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio  = out_width  / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.
            normalize_(resized_image, db.mean, db.std)

            images[0]  = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0]   = [int(height * scale), int(width * scale)]
            ratios[0]  = [height_ratio, width_ratio]       

            # Flip to perform detection twice
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            bert_features = np.concatenate((bert_features, bert_features), axis=0)

            images = torch.from_numpy(images)
            bert_features = torch.from_numpy(bert_features)
            dets, center = decode_func(nnet, [images, bert_features], K, ae_threshold=ae_threshold, kernel=nms_kernel)
            dets   = dets.reshape(2, -1, 8)
            center = center.reshape(2, -1, 4)
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            center[1, :, [0]] = out_width - center[1, :, [0]]
            dets   = dets.reshape(1, -1, 8)
            center   = center.reshape(1, -1, 4)
            
            _rescale_dets(dets, ratios, borders, sizes)
            center [...,[0]] /= ratios[:, 1][:, None, None]
            center [...,[1]] /= ratios[:, 0][:, None, None] 
            center [...,[0]] -= borders[:, 2][:, None, None]
            center [...,[1]] -= borders[:, 0][:, None, None]
            np.clip(center [...,[0]], 0, sizes[:, 1][:, None, None], out=center [...,[0]])
            np.clip(center [...,[1]], 0, sizes[:, 0][:, None, None], out=center [...,[1]])
            dets[:, :, 0:4] /= scale
            center[:, :, 0:2] /= scale

            if scale == 1:
              center_points.append(center)
            detections.append(dets)

        detections = np.concatenate(detections, axis=1)
        center_points = np.concatenate(center_points, axis=1)

        classes    = detections[..., -1]
        classes    = classes[0]
        detections = detections[0]
        center_points = center_points[0]
        
        valid_ind = detections[:,4]> -1
        valid_detections = detections[valid_ind]
        
        box_width = valid_detections[:,2] - valid_detections[:,0]
        box_height = valid_detections[:,3] - valid_detections[:,1]
        
        s_ind = (box_width*box_height <= 22500)
        l_ind = (box_width*box_height > 22500)
        
        s_detections = valid_detections[s_ind]
        l_detections = valid_detections[l_ind]
        
        s_left_x = (2*s_detections[:,0] + s_detections[:,2])/3
        s_right_x = (s_detections[:,0] + 2*s_detections[:,2])/3
        s_top_y = (2*s_detections[:,1] + s_detections[:,3])/3
        s_bottom_y = (s_detections[:,1]+2*s_detections[:,3])/3
        
        s_temp_score = copy.copy(s_detections[:,4])
        s_detections[:,4] = -1
        
        center_x = center_points[:,0][:, np.newaxis]
        center_y = center_points[:,1][:, np.newaxis]
        s_left_x = s_left_x[np.newaxis, :]
        s_right_x = s_right_x[np.newaxis, :]
        s_top_y = s_top_y[np.newaxis, :]
        s_bottom_y = s_bottom_y[np.newaxis, :]
        
        ind_lx = (center_x - s_left_x) > 0
        ind_rx = (center_x - s_right_x) < 0
        ind_ty = (center_y - s_top_y) > 0
        ind_by = (center_y - s_bottom_y) < 0
        ind_cls = (center_points[:,2][:, np.newaxis] - s_detections[:,-1][np.newaxis, :]) == 0
        ind_s_new_score = np.max(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0)), axis = 0) == 1
        index_s_new_score = np.argmax(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0))[:,ind_s_new_score], axis = 0)
        s_detections[:,4][ind_s_new_score] = (s_temp_score[ind_s_new_score]*2 + center_points[index_s_new_score,3])/3
       
        l_left_x = (3*l_detections[:,0] + 2*l_detections[:,2])/5
        l_right_x = (2*l_detections[:,0] + 3*l_detections[:,2])/5
        l_top_y = (3*l_detections[:,1] + 2*l_detections[:,3])/5
        l_bottom_y = (2*l_detections[:,1]+3*l_detections[:,3])/5
        
        l_temp_score = copy.copy(l_detections[:,4])
        l_detections[:,4] = -1
        
        center_x = center_points[:,0][:, np.newaxis]
        center_y = center_points[:,1][:, np.newaxis]
        l_left_x = l_left_x[np.newaxis, :]
        l_right_x = l_right_x[np.newaxis, :]
        l_top_y = l_top_y[np.newaxis, :]
        l_bottom_y = l_bottom_y[np.newaxis, :]
        
        ind_lx = (center_x - l_left_x) > 0
        ind_rx = (center_x - l_right_x) < 0
        ind_ty = (center_y - l_top_y) > 0
        ind_by = (center_y - l_bottom_y) < 0
        ind_cls = (center_points[:,2][:, np.newaxis] - l_detections[:,-1][np.newaxis, :]) == 0
        ind_l_new_score = np.max(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0)), axis = 0) == 1
        index_l_new_score = np.argmax(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0))[:,ind_l_new_score], axis = 0)
        l_detections[:,4][ind_l_new_score] = (l_temp_score[ind_l_new_score]*2 + center_points[index_l_new_score,3])/3
        
        detections = np.concatenate([l_detections,s_detections],axis = 0)
        detections = detections[np.argsort(-detections[:,4])] 
        classes   = detections[..., -1]

        # reject detections with negative scores
        keep_inds  = (detections[:, 4] > -1)
        detections = detections[keep_inds]
        classes    = classes[keep_inds]

        top_bboxes[db_ind] = {}

        top_bboxes[db_ind] = detections[:, 0:7].astype(np.float32)
        if merge_bbox:
            soft_nms_merge(top_bboxes[db_ind], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
        else:
            soft_nms(top_bboxes[db_ind], Nt=nms_threshold, method=nms_algorithm)
        top_bboxes[db_ind] = top_bboxes[db_ind][:, 0:5]

        scores = top_bboxes[db_ind][:, -1]
        if scores is not None and len(scores) > 0:
            best_bboxes[db_ind] = top_bboxes[db_ind][np.argmax(scores)]
        else:
            best_bboxes[db_ind] = None

        if len(scores) > max_per_image:
            kth    = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            keep_inds = (top_bboxes[db_ind][:, -1] >= thresh)
            top_bboxes[db_ind] = top_bboxes[db_ind][keep_inds]

        if debug:
            image_file = db.image_file(db_ind)
            image      = cv2.imread(image_file)
            im         = image[:, :, (2, 1, 0)]
            fig, ax    = plt.subplots(figsize=(12, 12)) 
            fig        = ax.imshow(im, aspect='equal')
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            keep_inds = (top_bboxes[db_ind][:, -1] >= 0.4)
            for bbox in top_bboxes[db_ind][keep_inds]:
                bbox  = bbox[0:4].astype(np.int32)
                xmin     = bbox[0]
                ymin     = bbox[1]
                xmax     = bbox[2]
                ymax     = bbox[3]
                ax.add_patch(plt.Rectangle((xmin, ymin),xmax - xmin, ymax - ymin, fill=False, edgecolor=bbox_color, linewidth=4.0))
                ax.text(xmin+1, ymin-3, 'object', bbox=dict(facecolor=bbox_color, ec='black', lw=2,alpha=0.5), fontsize=15, color='white', weight='bold')

            # debug_file1 = os.path.join(debug_dir, "{}.pdf".format(db_ind))
            debug_file2 = os.path.join(debug_dir, "{}.jpg".format(db_ind))
            # plt.savefig(debug_file1)
            plt.savefig(debug_file2)
            plt.close()

    result_json = os.path.join(result_dir, "results.json")
    detections  = db.convert_to_json(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)

    db.evaluate(best_bboxes)
    return 0

def testing(db, nnet, result_dir, debug=False, partial=False):
    return globals()[system_configs.sampling_function](db, nnet, result_dir, debug=debug, partial=partial)
