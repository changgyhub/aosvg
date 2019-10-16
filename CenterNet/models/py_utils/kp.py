import pdb
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr

from .kp_utils import _tranpose_and_gather_feat, _decode
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss
from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer, make_ct_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer


def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord


class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class kp(nn.Module):
    def __init__(
        self, db, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer, make_ct_layer=make_ct_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(kp, self).__init__()

        self.nstack             = nstack
        self._decode            = _decode
        self._db                = db
        self.K                  = self._db.configs["top_k"]
        self.ae_threshold       = self._db.configs["ae_threshold"]
        self.kernel             = self._db.configs["nms_kernel"]
        self.input_size         = self._db.configs["input_size"][0]
        self.output_size        = self._db.configs["output_sizes"][0][0]

        self.fix_visual         = self._db.configs["fix_visual"]

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        # ============================================
        # Language Attention module
        self.coordmap = self._db.configs["coordmap"]
        self.bert_model = self._db.configs["bert_model"]
        self.textmodel = lambda x: x
        self.mapping_lang = lambda x: x
        self.fusion_layers = lambda x: x
        # ============================================

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        self.ct_cnvs = nn.ModuleList([
            make_ct_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_hms = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_hms = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.ct_hms = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        ## tags
        self.tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        for tl_hm, br_hm, ct_hm in zip(self.tl_hms, self.br_hms, self.ct_hms):
            tl_hm[-1].bias.data.fill_(-2.19)
            br_hm[-1].bias.data.fill_(-2.19)
            ct_hm[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.ct_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image      = xs[0]
        word_id    = xs[1]
        word_mask  = xs[2]
        tl_inds    = xs[3]
        br_inds    = xs[4]
        ct_inds    = xs[5]

        inter      = self.pre(image)
        outs       = []

        layers = zip(
            self.kps,      self.cnvs,
            self.tl_cnvs,  self.br_cnvs, 
            self.ct_cnvs,  self.tl_hms, 
            self.br_hms, self.ct_hms,
            self.tl_tags,  self.br_tags,
            self.tl_regrs, self.br_regrs,
            self.ct_regrs,
            self.fusion_layers
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_,  br_cnv_  = layer[2:4]
            ct_cnv_,  tl_hm_ = layer[4:6]
            br_hm_, ct_hm_ = layer[6:8]
            tl_tag_,  br_tag_  = layer[8:10]
            tl_regr_,  br_regr_ = layer[10:12]
            ct_regr_         = layer[12]
            fusion_layer = layer[13]

            # # We fix the feature extracter with pre-trained model.
            # with torch.no_grad():
            kp  = kp_(inter)
            if self.fix_visual:
                kp = kp.detach()
            cnv = cnv_(kp)

            # ============================================
            # Language Attention module
            all_encoder_layers, _ = self.textmodel(word_id, token_type_ids=None, attention_mask=word_mask)
            raw_flang = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:] + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
            raw_flang = raw_flang.detach()
            flang = self.mapping_lang(raw_flang)
            flang = F.normalize(flang, p=2, dim=1)
            flang_tile = flang.view(flang.size(0), flang.size(1), 1, 1).repeat(1, 1, cnv.shape[2], cnv.shape[3])
            if self.coordmap:
                coord = generate_coord(cnv.shape[0], cnv.shape[2], cnv.shape[3])
                cnv = torch.cat([cnv, flang_tile, coord], dim=1)
            else:
                cnv = torch.cat([cnv, flang_tile], dim=1)
            cnv = fusion_layer(cnv)
            # ============================================     

            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)
            ct_cnv = ct_cnv_(cnv)

            tl_hm, br_hm, ct_hm = tl_hm_(tl_cnv), br_hm_(br_cnv), ct_hm_(ct_cnv)
            tl_tag, br_tag        = tl_tag_(tl_cnv),  br_tag_(br_cnv)
            tl_regr, br_regr, ct_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), ct_regr_(ct_cnv)

            tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
            br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)

            outs += [tl_hm, br_hm, ct_hm, tl_tag, br_tag, tl_regr, br_regr, ct_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return outs

    def _test(self, *xs, **kwargs):
        image      = xs[0]
        print(image.shape)
        word_id    = xs[1]
        print(word_id.shape)
        word_mask  = xs[2]

        inter = self.pre(image)

        outs          = []

        layers = zip(
            self.kps,      self.cnvs,
            self.tl_cnvs,  self.br_cnvs,
            self.ct_cnvs,  self.tl_hms,
            self.br_hms, self.ct_hms,
            self.tl_tags,  self.br_tags,
            self.tl_regrs, self.br_regrs,
            self.ct_regrs,
            self.fusion_layers
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_,  br_cnv_  = layer[2:4]
            ct_cnv_,  tl_hm_ = layer[4:6]
            br_hm_, ct_hm_ = layer[6:8]
            tl_tag_,  br_tag_  = layer[8:10]
            tl_regr_,  br_regr_ = layer[10:12]
            ct_regr_         = layer[12]
            fusion_layer = layer[13]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            # ============================================
            # Language Attention module
            all_encoder_layers, _ = self.textmodel(word_id, token_type_ids=None, attention_mask=word_mask)
            raw_flang = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:] + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
            flang = self.mapping_lang(raw_flang)
            flang = F.normalize(flang, p=2, dim=1)
            flang_tile = flang.view(flang.size(0), flang.size(1), 1, 1).repeat(1, 1, cnv.shape[2], cnv.shape[3])
            if self.coordmap:
                coord = generate_coord(cnv.shape[0], cnv.shape[2], cnv.shape[3])
                cnv = torch.cat([cnv, flang_tile, coord], dim=1)
            else:
                cnv = torch.cat([cnv, flang_tile], dim=1)
            cnv = fusion_layer(cnv)
            # ============================================     

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                ct_cnv = ct_cnv_(cnv)

                tl_hm, br_hm, ct_hm = tl_hm_(tl_cnv), br_hm_(br_cnv), ct_hm_(ct_cnv)
                tl_tag, br_tag        = tl_tag_(tl_cnv),  br_tag_(br_cnv)
                tl_regr, br_regr, ct_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), ct_regr_(ct_cnv)

                outs += [tl_hm, br_hm, tl_tag, br_tag, tl_regr, br_regr,
                         ct_hm, ct_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
                
        return self._decode(*outs[-8:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 3:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        stride = 8

        tl_hms = outs[0::stride]
        br_hms = outs[1::stride]
        ct_hms = outs[2::stride]
        tl_tags  = outs[3::stride]
        br_tags  = outs[4::stride]
        tl_regrs = outs[5::stride]
        br_regrs = outs[6::stride]
        ct_regrs = outs[7::stride]

        gt_tl_hm = targets[0]
        gt_br_hm = targets[1]
        gt_ct_hm = targets[2]
        gt_mask    = targets[3]
        gt_tl_regr = targets[4]
        gt_br_regr = targets[5]
        gt_ct_regr = targets[6]
        
        # focal loss
        focal_loss = 0

        tl_hms = [_sigmoid(t) for t in tl_hms]
        br_hms = [_sigmoid(b) for b in br_hms]
        ct_hms = [_sigmoid(c) for c in ct_hms]

        focal_loss += self.focal_loss(tl_hms, gt_tl_hm)
        focal_loss += self.focal_loss(br_hms, gt_br_hm)
        focal_loss += self.focal_loss(ct_hms, gt_ct_hm)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr, ct_regr in zip(tl_regrs, br_regrs, ct_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
            regr_loss += self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(tl_hms)
        return loss.unsqueeze(0), (focal_loss / len(tl_hms)).unsqueeze(0), (pull_loss / len(tl_hms)).unsqueeze(0), (push_loss / len(tl_hms)).unsqueeze(0), (regr_loss / len(tl_hms)).unsqueeze(0)
