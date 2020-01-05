# --------------------------------------------------------
# OIM
# Copyright (c) 2019 SenseTime Research
# Licensed under The MIT License [see LICENSE for details]
# Written by SenseTime Research (OICRLayer for reference)
# --------------------------------------------------------


"""The layer used during training for object instance mining.

OIMLayer implements a Caffe Python layer.
"""

from __future__ import division
import caffe
import numpy as np
import os
import scipy.sparse
import yaml

from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps

# print all elements in a numpy nd-array
np.set_printoptions(threshold=np.inf)

DEBUG = False

class OIMLayer(caffe.Layer):
    """get proposal labels used for online instance classifier refinement."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        if len(bottom) != 4:
            raise Exception("The number of bottoms should be 4!")

        if len(top) != 6:
            raise Exception("The number of tops should be 6!")

        if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
            raise Exception("bottom[0].data.shape[0] must equal to bottom[1].data.shape[0]")

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        top[3].reshape(1)
        top[4].reshape(1)
        top[5].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        boxes = bottom[0].data[:, 1:]
        cls_prob = bottom[1].data
        if cls_prob.shape[1] == self._num_classes:
            cls_prob = cls_prob[:, 1:]
        im_labels = bottom[2].data

        feature = bottom[3].data
        prob = cls_prob.copy()
        proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels, feature)
        labels, rois, cls_loss_weights, prob_ratio, fg_num = _sample_rois(boxes, proposals, self._num_classes, prob)

        assert rois.shape[0] == boxes.shape[0]

        # classification labels
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        top[1].reshape(*cls_loss_weights.shape)
        top[1].data[...] = cls_loss_weights

        top[2].reshape(*prob_ratio.shape)
        top[2].data[...] = prob_ratio

        # hyper-parameter for instance mining
        alpha = np.array([float(os.getenv('alpha', None))])
        top[3].reshape(*alpha.shape)
        top[3].data[...] = alpha

        top[4].reshape(*fg_num.shape)
        top[4].data[...] = fg_num

        beta = np.array([float(os.getenv('beta', None))])
        top[5].reshape(*beta.shape)
        top[5].data[...] = beta

        if DEBUG:
            print 'alpha:', alpha
            print 'beta:', beta

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _get_highest_score_proposals(boxes, cls_prob, im_labels, feature):
    """Get proposals with highest score."""

    dis_ratio = float(os.getenv('ratio', None))

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    gt_indices = np.zeros((0, 1), dtype=np.int32)
    gt_max = set()
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            max_index = np.argmax(cls_prob_tmp)

            # calculate distances among bbox and maximum
            dis_all = []
            for f in range(feature.shape[0]):
                distance = np.linalg.norm(feature[max_index][:] - feature[f][:])
                dis_all.append(distance)
            dis_all = np.array(dis_all)

            if DEBUG:
                print 'max_index:', max_index, 'cls_prob_tmp:', cls_prob_tmp[max_index]

            gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1)))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
            gt_scores = np.vstack((gt_scores,
                                   cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))
            gt_indices = np.vstack((gt_indices, max_index * np.ones((1, 1), dtype=np.int32)))
            gt_max.add(max_index)

            # multi-instance
            overlaplist = []
            overlaps_ins = bbox_overlaps(
                np.ascontiguousarray(boxes, dtype=np.float),
                np.ascontiguousarray(np.expand_dims(boxes[max_index], axis=0), dtype=np.float))
            overlaplist.append(overlaps_ins)

            max_overlaps = overlaps_ins.max(axis=1)
            fg_dis = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
            soft_dis = np.mean(dis_all[fg_dis])

            # debug
            if DEBUG:
                print soft_dis, np.std(dis_all[fg_dis]), \
                    np.min(dis_all[fg_dis]), np.max(dis_all[fg_dis]), np.linalg.norm(feature[max_index][:])

            # (key, value): (index, distance)
            k_dis = np.argsort(dis_all)
            v_dis = np.sort(dis_all)
            assert len(k_dis) == len(overlaps_ins) == len(v_dis)

            score_ins = cls_prob_tmp[max_index] * 0.01
            for idx in range(len(k_dis)):
                # search from the smallest instance
                flag = 0
                key_ins = k_dis[idx]
                # within 5 times of distances
                if v_dis[idx] < soft_dis * dis_ratio:
                    for found_bbox_idx in range(len(overlaplist)):
                        if overlaplist[found_bbox_idx][key_ins] >= 0.000001:
                            # skip connected bbox
                            flag = 1
                            break
                    if flag == 0:
                        overlaps_ins_tmp = bbox_overlaps(
                            np.ascontiguousarray(boxes, dtype=np.float),
                            np.ascontiguousarray(np.expand_dims(boxes[key_ins], axis=0), dtype=np.float))
                        overlaplist.append(overlaps_ins_tmp)
                        cls_prob[key_ins, :] = 0
                        gt_boxes = np.vstack((gt_boxes, boxes[key_ins, :].reshape(1, -1)))
                        gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
                        gt_scores = np.vstack((gt_scores,
                                               cls_prob_tmp[key_ins] * np.ones((1, 1), dtype=np.float32)))
                        gt_indices = np.vstack((gt_indices, key_ins * np.ones((1, 1), dtype=np.int32)))
                else:
                    break
            if DEBUG:
                print 'len(overlaplist):', len(overlaplist)
            cls_prob[max_index, :] = 0

    proposals = {'gt_boxes': gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores,
                 'gt_indices': gt_indices,
                 'gt_max': gt_max}

    return proposals

def _sample_rois(all_rois, proposals, num_classes, cls_prob):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    gt_indices = proposals['gt_indices']
    gt_max = proposals['gt_max']
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    indices = gt_indices[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds_first = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    fg_inds = []
    for n in fg_inds_first:
        if indices[n] in gt_max or n in gt_indices[:, 0]:
            fg_inds.append(n)
        else:
            labels[n] = 0
    fg_inds = np.array(fg_inds, dtype=np.int32)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    if DEBUG:
        print "number of fg:", len(fg_inds), 'number of bg:', len(bg_inds)

    labels[bg_inds] = 0

    cls_prob = np.insert(cls_prob, 0, values=cls_loss_weights, axis=1)
    prob = cls_prob[np.arange(len(labels)), labels]
    prob_ratio = prob / (cls_loss_weights + 1e-5)

    fg_num = np.zeros(max_overlaps.shape, dtype=np.float)
    for index in fg_inds:
        fg_num[gt_indices[gt_assignment[index]]] += 1

    rois = all_rois

    return labels, rois, cls_loss_weights, prob_ratio, fg_num
