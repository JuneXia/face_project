"""Functions for building the face recognition network.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile
import math
from six import iteritems
import cv2


RELEASE = True

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss



def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets



def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


import tensorflow.contrib.slim as slim
def center_loss1(features, label, alfa, nrof_classes, scope=None, name=None):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    weight_decay = 5e-4
    nrof_features = features.get_shape()[1]
    if (scope is not None) and (name is not None):
        with tf.variable_scope(scope) as center_scope:
            # logits_w = tf.get_variable("weights")  # [128, n_cls]   >>>   transpose
            # centers = tf.get_variable(name, [nrof_features, nrof_classes], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=True)
            centers = tf.get_variable(name, [nrof_features, nrof_classes], dtype=tf.float32, initializer=slim.initializers.xavier_initializer(), weights_regularizer=slim.l2_regularizer(weight_decay), trainable=True)
    else:
        centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)

    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch), name=name)
    return loss, centers


def center_loss2(normalized_pred, logits_w, label, alfa, nrof_classes, name='center_loss'):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    # nrof_features = normalized_pred.get_shape()[1]
    # centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)

    logits_w = tf.nn.l2_normalize(logits_w, 0, 1e-10)
    centers = tf.transpose(logits_w)  # n_cls x 512
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)  # -1 x 512
    diff = (1 - alfa) * (centers_batch - normalized_pred)  # -1 x 512
    # centers = tf.scatter_sub(tf.Variable(centers, trainable=False), label, diff)
    centers = tf.tensor_scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(normalized_pred - centers_batch), name=name)
    return loss, centers


"""


# OK
centers
<tf.Variable 'centers:0' shape=(35, 512) dtype=float32_ref>
label
<tf.Tensor 'Reshape:0' shape=(?,) dtype=int32>
diff
<tf.Tensor 'mul_1:0' shape=(?, 512) dtype=float32>
"""


def arcface_loss_src(embedding, labels, out_num, w_init=None, s=64., m=0.5):
    '''
    Reference: https://github.com/auroua/InsightFace_TF/losses/face_losses.py
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output



def arcface_loss_损失值较大(embedding, labels, out_num, w_init=None, s=64., m=0.5, name='arc_loss'):
    '''
    Reference: https://github.com/auroua/InsightFace_TF/losses/face_losses.py
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('Logits', reuse=True):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')

    ross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
    loss = tf.reduce_mean(ross_entropy, name=name)
    return loss


def arcface_loss_参考mx用tf半改半就写的_应该还没调试过(embedding, labels, out_num, w_init=None, s=64., m=0.5, name='arc_loss'):
    '''
    Reference: https://github.com/auroua/InsightFace_TF/losses/face_losses.py
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('Logits', reuse=True):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')

        labs = tf.reshape(tf.argmax(labels, axis=1), (-1, 1))
        cos_t_flatten = tf.gather_nd(cos_t, labs, batch_dims=1)

        cos_t2 = tf.square(cos_t_flatten, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t_flatten, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t_flatten - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t_flatten - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        diff = cos_mt_temp - s * cos_t_flatten
        diff = tf.expand_dims(diff, axis=1)
        onehot_labels = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        diff = tf.reshape(diff, (-1, 1))
        body = tf.multiply(onehot_labels, diff)

        output = body + cos_t * s

    ross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output)
    loss = tf.reduce_mean(ross_entropy, name=name)
    return loss



def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.5, name='arc_loss', summary=True):
    '''
    Reference: https://github.com/auroua/InsightFace_TF/losses/face_losses.py
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    shape_list = labels.shape.as_list()
    if len(shape_list) == 2 and shape_list[-1] > 1:
        onehot = True
    else:
        onehot = False

    with tf.variable_scope('Logits', reuse=True):
        weights = tf.get_variable(name='weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)

    # embedding = tf.nn.l2_normalize(embedding, 0, 1e-10)
    weights = tf.nn.l2_normalize(weights, 0, 1e-10)

    # cos(theta+m)
    fc7 = tf.matmul(embedding, weights, name='cos_t')  # cos_t

    if onehot:
        labs = tf.reshape(tf.argmax(labels, axis=1), (-1, 1))
    else:
        labs = tf.reshape(labels, (-1, 1))
    zy = tf.gather_nd(fc7, labs, batch_dims=1)  # cos_t_flatten

    # theta = tf.acos(0.707106781)*180/np.pi  == 45
    theta = tf.acos(zy, name='theta')
    zy_margin = tf.cos(theta + m, name='theta_add_margin')

    if summary:
        theta_mean, theta_var = tf.nn.moments(theta, axes=[0])
        tf.summary.scalar(theta.op.name + '_mean', theta_mean)
        tf.summary.scalar(theta.op.name + '_var', theta_var)

    if not onehot:
        onehot_labels = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
    else:
        onehot_labels = labels
    diff = zy_margin - zy
    diff = tf.expand_dims(diff, 1)
    fc7 = fc7 + tf.multiply(onehot_labels, diff)
    output = fc7 * s

    ross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
    loss = tf.reduce_mean(ross_entropy, name=name)
    return loss


def cosineface_losses1(embedding, labels, num_cls, w_init=None, s=30., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], num_cls),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=num_cls, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
    return output


# 参考liyule，但还是重新定义Weight
def cosineface_losses2(normalized_pred, labels, num_cls, w_init=None, reuse=False, margin=0.25, scale=64, name='cos_loss'):
    '''
    x: B x D - features
    y: B x 1 - labels
    num_cls: 1 - total class number
    alpah: 1 - margin
    scale: 1 - scaling paramter
    '''
    # define the classifier weights
    xs = normalized_pred.get_shape()
    with tf.variable_scope('centers_var', reuse=reuse) as center_scope:
        w = tf.get_variable("centers", [xs[1], num_cls], dtype=tf.float32,
                            initializer=w_init, trainable=True)  # tf.contrib.layers.xavier_initializer()

    # normalize the feature and weight
    # (N,D)
    # normalized_pred = tf.nn.l2_normalize(embedding, 1, 1e-10)
    # (D,C)
    w_feat_norm = tf.nn.l2_normalize(w, 0, 1e-10)

    # get the scores after normalization
    # (N,C)
    xw_norm = tf.matmul(normalized_pred, w_feat_norm)
    # implemented by py_func
    # value = tf.identity(xw)
    # substract the marigin and scale it
    # value = coco_func(xw_norm, y, margin) * scale

    # implemented by tf api
    margin_xw_norm = xw_norm - margin
    label_onehot = tf.one_hot(labels, num_cls)
    value = scale*tf.where(tf.equal(label_onehot, 1), margin_xw_norm, xw_norm)

    # compute the loss as softmax loss
    # cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=value))

    return value


def cosineface_losses(normalized_pred, logits_w, labels, num_cls, margin=0.25, scale=64, use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0, name='cos_loss'):
    '''
    x: B x D - features
    y: B x 1 - labels
    num_cls: 1 - total class number
    alpah: 1 - margin
    scale: 1 - scaling paramter
    '''
    # define the classifier weights
    xs = normalized_pred.get_shape()
    # with tf.variable_scope('Logits', reuse=reuse) as center_scope:
    #      w = tf.get_variable("weights", [xs[1], num_cls], dtype=tf.float32,
    #                          initializer=w_init, trainable=True)  # tf.contrib.layers.xavier_initializer()

    # normalize the feature and weight
    # (N,D)
    # normalized_pred = tf.nn.l2_normalize(embedding, 1, 1e-10)
    # (D,C)
    w_feat_norm = tf.nn.l2_normalize(logits_w, 0, 1e-10)

    # get the scores after normalization
    # (N,C)
    xw_norm = tf.matmul(normalized_pred, w_feat_norm)
    # implemented by py_func
    # value = tf.identity(xw)
    # substract the marigin and scale it
    # value = coco_func(xw_norm, y, margin) * scale

    # implemented by tf api
    margin_xw_norm = xw_norm - margin
    label_onehot = tf.one_hot(labels, num_cls)
    logits = scale*tf.where(tf.equal(label_onehot, 1), margin_xw_norm, xw_norm)

    # compute the loss as softmax loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    if use_focal_loss:
        # 从logits计算softmax
        reduce_max = tf.reduce_max(normalized_pred, axis=1, keepdims=True)
        prob = tf.nn.softmax(normalized_pred - reduce_max)

        # 计算交叉熵
        # clip_prob = tf.clip_by_value(prob, 1e-10, 1.0)
        # cross_entropy = -tf.reduce_sum(y_true * tf.log(clip_prob), 1)

        # 计算focal_loss
        prob = tf.reduce_max(prob, axis=1)
        weight = tf.pow(tf.subtract(1., prob), focal_gamma)
        fl = tf.multiply(tf.multiply(weight, cross_entropy), focal_alpha)
        loss = tf.reduce_mean(fl, name=name)
    else:
        loss = tf.reduce_mean(cross_entropy, name=name)

    return loss





from tensorflow.python.ops import array_ops
def focal_loss1(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2, name='focal_loss'):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    # target_tensor = tf.cast(target_tensor, dtype=tf.float32)
    target_tensor = tf.one_hot(target_tensor, depth=prediction_tensor.get_shape().as_list()[-1])
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent, name=name)

# _打算用tf内置交叉熵实现
def focal_loss_CustomTfCrossEntropy(y_pred, y_true, weights=None, alpha=0.25, gamma=2., name='focal_loss'):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    # zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    #
    # # For poitive prediction, only need consider front part loss, back part is 0;
    # # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    # # target_tensor = tf.cast(target_tensor, dtype=tf.float32)
    # target_tensor = tf.one_hot(target_tensor, depth=prediction_tensor.get_shape().as_list()[-1])
    # pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    #
    # # For negative prediction, only need consider back part loss, front part is 0;
    # # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    # neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    # per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
    #                       - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    y_true = tf.one_hot(y_true, depth=y_pred.get_shape().as_list()[-1], dtype=tf.float32)

    reduce_max = tf.reduce_max(y_pred, axis=1, keepdims=True)
    y_pred = tf.nn.softmax(y_pred - reduce_max)

    clip_preb = tf.clip_by_value(y_pred, 1e-10, 1.0)
    cross_entropy = -tf.reduce_sum(y_true * tf.log(clip_preb), 1)

    prob = tf.reduce_max(clip_preb, axis=1)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    weight = tf.clip_by_value(weight, 1e-10, 1.0)
    fl = tf.multiply(tf.multiply(weight, cross_entropy), alpha)
    fl = tf.clip_by_value(fl, 1e-10, 1.0)
    loss = tf.reduce_sum(fl, name=name)


    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # per_entry_cross_ent = alpha * tf.pow(tf.subtract(1., prediction_tensor), gamma) * cross_entropy
    # weight = tf.pow(tf.subtract(1., prediction_tensor), gamma)
    # per_entry_cross_ent = tf.multiply(tf.multiply(weight, cross_entropy), alpha)
    # loss = tf.reduce_sum(per_entry_cross_ent, name=name)

    # pred = tf.reduce_max(y_pred, axis=1)
    # weight = tf.pow(tf.subtract(1., pred), gamma)
    # fl = tf.multiply(tf.multiply(weight, cross_entropy), alpha)
    # loss = tf.reduce_sum(fl, name=name)

    return loss


# 交叉熵使用tf.nn.sparse_softmax_cross_entropy_with_logits实现
# 权重自己使用tf接口实现。
def focal_loss(y_pred, y_true, weights=None, alpha=0.25, gamma=2., name='focal_loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # 从logits计算softmax
    reduce_max = tf.reduce_max(y_pred, axis=1, keepdims=True)
    prob = tf.nn.softmax(y_pred - reduce_max)

    # 计算交叉熵
    # clip_prob = tf.clip_by_value(prob, 1e-10, 1.0)
    # cross_entropy = -tf.reduce_sum(y_true * tf.log(clip_prob), 1)

    # 计算focal_loss
    prob = tf.reduce_max(prob, axis=1)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    fl = tf.multiply(tf.multiply(weight, cross_entropy), alpha)
    loss = tf.reduce_mean(fl, name=name)

    return loss


def focal_loss_ok_损失波动较大(y_pred, y_true, weights=None, alpha=0.25, gamma=2., name='focal_loss'):
    y_true = tf.one_hot(y_true, depth=y_pred.get_shape().as_list()[-1], dtype=tf.float32)

    reduce_max = tf.reduce_max(y_pred, axis=1, keepdims=True)
    y_pred = tf.nn.softmax(tf.subtract(y_pred, reduce_max))

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0)
    # cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred), 1)
    cross_entropy = -tf.reduce_sum(tf.multiply(y_true, tf.log(y_pred)), axis=1)

    # 计算focal_loss
    prob = tf.reduce_max(y_pred, axis=1)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    # weight = tf.multiply(tf.multiply(weight, y_true), alpha)
    # weight = tf.reduce_max(weight, axis=1)

    fl = tf.multiply(tf.multiply(weight, cross_entropy), alpha)
    loss = tf.reduce_sum(fl, name=name)

    return loss

# tf.nn.softmax_cross_entropy_with_logits_v2(target_tensor, prediction_tensor, axis=None, name=None, dim=None)
# tf.nn.softmax_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor, dim=-1, name=None, axis=None)

# _自己使用tf基础函数实现交叉熵
def focal_loss3(prediction_tensor, target_tensor, weights=None, gamma=2., alpha=.25, name='focal_loss'):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    # y_true = tf.cast(target_tensor, tf.float32)
    y_true = tf.one_hot(target_tensor, depth=prediction_tensor.get_shape().as_list()[-1])
    y_pred = tf.clip_by_value(prediction_tensor, epsilon, 1. - epsilon)

    alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
    loss = tf.reduce_mean(fl, name=name)

    return loss


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff

def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')



def glass_pad(image, points, prob=0.5):
    index = np.random.randint(0, 100)
    if index < prob*100:
        offset = np.random.randint(6, 12)
        # offset = 15
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]

        dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)) / (offset + 10)
        _y1 = y1 + (y1 - y2) / dist
        _x1 = x1 - (x2 - x1) / dist
        x2 = x2 + x1 - _x1
        y2 = y2 - (_y1 - y1)
        x1, y1 = _x1, _y1

        fraction = (x2 - x1)
        if fraction == 0:
            x10 = x20 = x1 - offset
            y10, y20 = y1, y2
            x11 = x21 = x1 + offset
            y11, y21 = y1, y2
        else:
            x10 = x1 - (y1 - y2) * offset / fraction
            y10 = y1 - offset
            x11 = x10 + 2 * (x1 - x10)
            y11 = y1 + offset

            x20 = x2 - (y1 - y2) * offset / fraction
            y20 = y2 - offset
            x21 = x20 + 2 * (x2 - x20)
            y21 = y2 + offset

        pt1 = int(round(x10)), int(round(y10))
        pt2 = int(round(x11)), int(round(y11))
        pt3 = int(round(x20)), int(round(y20))
        pt4 = int(round(x21)), int(round(y21))

        pts = np.array([[pt1, pt2, pt4, pt3]], dtype=np.int32)
        # cv2.polylines(image, pts, isClosed=False, color=(128, 128, 128))
        cv2.fillPoly(image, pts, (127.5, 127.5, 127.5))

        if False:
            image = image.astype(np.uint8)
            centerpt1 = points[0].astype(np.int)
            centerpt2 = points[1].astype(np.int)

            cv2.circle(image, tuple(centerpt1), 5, (255, 0, 0), thickness=-1)
            cv2.circle(image, tuple(centerpt2), 5, (0, 255, 0), thickness=-1)
            # cv2.ellipse(image, tuple(centerpt2), (20, 10), 0, 0, 360, (255, 255, 255), 3)

            cv2.circle(image, tuple(pt1), 3, (255, 0, 0), thickness=-1)
            cv2.circle(image, tuple(pt2), 3, (0, 255, 0), thickness=-1)
            cv2.circle(image, tuple(pt3), 3, (0, 0, 255), thickness=-1)
            cv2.circle(image, tuple(pt4), 3, (255, 255, 0), thickness=-1)

    # return image


g_FaceEyePoints = {}
def loading_face_keypoint():
    global g_FaceEyePoints

    special = {'CASIA-FaceV5': 'CASIAFaceV5',
               }

    try:
        if RELEASE:
            keypoint_files = ['/disk1/home/xiaj/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44_keypoints.csv',
                              '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_mainland_cleaning_mtcnn_align182x182_margin44_keypoints.csv',
                              '/disk2/res/CASIA-FaceV5/CASIA-FaceV5-000-499-mtcnn_align182x182_margin44_mtcnn_align182x182_margin44_keypoints.csv']
        else:
            keypoint_files = ['/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44_keypoints.csv']
        for file in keypoint_files:
            for k, v in special:
                if k in file:
                    pass
            with open(file, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip().split(',')
                key = os.path.join(line[0], line[1])
                points = np.array(line[2:12]).astype(np.float)
                points = points.reshape((2, 5)).transpose()
                points = points[:2, :]
                g_FaceEyePoints[key] = points

                if i % 10000 == 0:
                    print('[loading_face_keypoint]:: {}/{}'.format(i, len(lines)))
    except Exception as e:
        print(e)

def random_glass_padding(image, filename):
    global g_FaceEyePoints
    global g_count

    info = filename.decode('utf-8').split('/')[-2:]
    key = os.path.join(info[0], info[1])
    # print('filename: {}, info: {}, key: {}'.format(filename, info, key))
    if key in g_FaceEyePoints.keys():
        points = g_FaceEyePoints[key]
        glass_pad(image, points, 0.05)
    else:
        # points = np.array([[50, 70], [120, 80]])
        # glass_pad(image, points)
        pass

    return image


def random_color_failed(image):
    prob = 0.0
    random_index = np.random.randint(0, 101)  # TODO：100 or 101
    print('[random_color]:: random_index={}, type(random_index)={}, image.shape={}'.format(random_index, type(random_index), image.shape))
    if random_index < prob * 100:
        index = np.random.randint(0, 4)
        print('[random_color]:: index={}, type(index)={}'.format(index, type(index)))
        if index == 0:
            image = tf.image.random_brightness(image, 0.4)
        elif index == 1:
            image = tf.image.random_contrast(image, 0.8, 2)
        elif index == 2:
            image = tf.image.random_hue(image, 0.08)
        elif index == 3:
            image = tf.image.random_saturation(image, 0, 1)

    print('[random_color]:: image.shape={}'.format(image.shape))
    return image


def random_contract_failed(image):
    raw_image_size = 182
    contract_size = 132

    central_crop = tf.image.central_crop(image, contract_size / raw_image_size)

    # 随机的稍微放大一点
    h = w = tf.random_uniform([], contract_size, raw_image_size+1, dtype=tf.int32)
    resize_image = tf.image.resize_image_with_pad(central_crop, h, w)
    # resize_image = tf.cast(resize_image, dtype=tf.uint8)

    # 填充至目标大小
    pad_image = tf.image.resize_image_with_crop_or_pad(resize_image, raw_image_size, raw_image_size)

    # random_crop = tf.random_crop(pad_image, (160, 160) + (3,))
    return pad_image


def random_color(image, control):
    randcolor = tf.random_uniform([], 0, 39, dtype=tf.int32)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctrl_bright = tf.cond(get_control_flag(control, RANDOM_COLOR),
                          lambda: tf.cond(tf.equal(randcolor, 0), lambda: True, lambda: False),
                          lambda: False)
    image = tf.cond(ctrl_bright,
                    lambda: tf.image.random_brightness(image, 0.4),
                    lambda: tf.identity(image))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctrl_contrast = tf.cond(get_control_flag(control, RANDOM_COLOR),
                            lambda: tf.cond(tf.equal(randcolor, 1), lambda: True, lambda: False),
                            lambda: False)
    image = tf.cond(ctrl_contrast,
                    lambda: tf.image.random_contrast(image, 0.8, 2),
                    lambda: tf.identity(image))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctrl_hue = tf.cond(get_control_flag(control, RANDOM_COLOR),
                       lambda: tf.cond(tf.equal(randcolor, 2), lambda: True, lambda: False),
                       lambda: False)
    image = tf.cond(ctrl_hue,
                    lambda: tf.image.random_hue(image, 0.08),
                    lambda: tf.identity(image))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ctrl_saturation = tf.cond(get_control_flag(control, RANDOM_COLOR),
                              lambda: tf.cond(tf.equal(randcolor, 3), lambda: True, lambda: False),
                              lambda: False)
    image = tf.cond(ctrl_saturation,
                    lambda: tf.image.random_saturation(image, 0, 1),
                    lambda: tf.identity(image))

    return image


def random_resize(image, control, raw_size=(182, 182)):
    scale_size = (raw_size[0]-16, raw_size[1]+16)
    rand_w = tf.random_uniform([], scale_size[0], scale_size[1], dtype=tf.int32)
    rand_h = tf.random_uniform([], scale_size[0], scale_size[1], dtype=tf.int32)

    image = tf.cond(get_control_flag(control, RANDOM_RESIZE),
                    lambda: tf.cast(tf.image.resize(image, (rand_h, rand_w)), tf.uint8),
                    lambda: tf.identity(image))

    # image = tf.image.resize_images(image, (rand_h, rand_w), align_corners=False, preserve_aspect_ratio=False)  # 如果preserve_aspect_ratio为True，则保持宽高比对原图进行缩放，缩放后的图像宽或高等于image_size中的最小值
    # image = tf.cast(image, dtype=tf.uint8)

    return image

def fixed_contract(image, control, raw_size=(182, 182)):
    if raw_size[0] != raw_size[1]:
        raise Exception('At present, the image width and height are equal. \n\
        If the width and height are not equal, part of the code of this function needs to be modified. \n\
        If you have known this idea, you can screen this exception.')

    contract_size = 132

    '''
    central_crop = tf.image.central_crop(image, contract_size / raw_image_size)
    print(
        '6 type(central_crop)={}, central_crop.shape={}, central_crop={}'.format(type(central_crop), central_crop.shape,
                                                                                 central_crop))

    # 随机的稍微放大一点
    h = w = tf.random_uniform([], contract_size, raw_image_size + 1, dtype=tf.int32)
    resize_image = tf.image.resize_image_with_pad(central_crop, h, w)
    # resize_image = tf.cast(resize_image, dtype=tf.uint8)

    # 填充至目标大小
    image = tf.image.resize_image_with_crop_or_pad(resize_image, raw_image_size, raw_image_size)
    '''

    # contract_size = tf.random_uniform([], 124, 144, dtype=tf.float64)
    image = tf.cond(get_control_flag(control, FIXED_CONTRACT),
                    # lambda: tf.image.central_crop(image, tf.divide(contract_size, raw_image_size)),  # 随机中心裁剪
                    lambda: tf.image.central_crop(tf.ensure_shape(image, (None, None, 3)), contract_size / raw_size[0]),  # 固定中心裁剪
                    lambda: tf.identity(image))

    image = tf.cond(get_control_flag(control, FIXED_CONTRACT),
                    lambda: tf.cast(tf.image.resize_image_with_pad(image, raw_size[0], raw_size[1]), tf.uint8),
                    lambda: tf.identity(image))

    return image


def load_image(filename):
    '''
    :param filename: 图像文件名
    :param resized_shape: 缩放后图像大小
    reference: https://blog.csdn.net/qq_20084101/article/details/87440231
    '''

    image = tf.read_file(filename)

    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(image),
        lambda: tf.image.decode_png(image))

    image = tf.image.convert_image_dtype(image, tf.uint8)

    return image


# 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
RANDOM_GLASS = 32
RANDOM_COLOR = 64
FIXED_CONTRACT = 128
RANDOM_RESIZE = 256
def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder):
    # raise Exception('Your train image size must be 182x182, If you have known this idea, you can screen this exception.')

    raw_size = (182, 182)

    images_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        filenames, label, control = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            # file_contents = tf.read_file(filename)
            # image = tf.image.decode_image(file_contents, 3)
            # image = tf.image.decode_png(file_contents, 3)
            image = load_image(filename)

            image = tf.cond(get_control_flag(control[0], RANDOM_GLASS),
                            lambda: tf.ensure_shape(tf.py_func(random_glass_padding, [image, filename], tf.uint8), (None, None, 3)),
                            lambda: tf.identity(image))

            image = random_color(image, control[0])

            image = random_resize(image, control[0], raw_size)

            image = fixed_contract(image, control[0], raw_size)

            image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),
                            lambda: tf.py_func(random_rotate_image, [image], tf.uint8),
                            lambda: tf.identity(image))

            image = tf.cond(get_control_flag(control[0], RANDOM_CROP),
                            lambda: tf.random_crop(image, image_size + (3,)),
                            lambda: tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))

            image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),
                            lambda: tf.image.random_flip_left_right(image),
                            lambda: tf.identity(image))

            image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),
                            lambda: (tf.cast(image, tf.float32) - 127.5)/128.0,
                            lambda: tf.image.per_image_standardization(image))

            if False:
                image = tf.cast(image, tf.float32)

            image = tf.cond(get_control_flag(control[0], FLIP),
                            lambda: tf.image.flip_left_right(image),
                            lambda: tf.identity(image))

            #pylint: disable=no-member
            image.set_shape(image_size + (3,))
            images.append(image)
        images_and_labels_list.append([images, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_placeholder,
        shapes=[image_size + (3,), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * 2000,
        allow_smaller_final_batch=True)

    return image_batch, label_batch

def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)

def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)  # epsilon=0.1
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        elif optimizer=='GD':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image

def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

def get_label_batch(label_data, batch_size, batch_index):
    nrof_examples = np.size(label_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = label_data[j:j+batch_size]
    else:
        x1 = label_data[j:nrof_examples]
        x2 = label_data[0:nrof_examples-j]
        batch = np.vstack([x1,x2])
    batch_int = batch.astype(np.int64)
    return batch_int

def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float

def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch

def get_learning_rate_from_file(filename, epoch):
    learning_rate = -1
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                if par[1]=='-':
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    return learning_rate

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def impath2imclass(images_path, images_label):
    dataset = []
    images_info = np.array((images_path, images_label))
    images_info = images_info.transpose()
    classes = set(images_label)
    for i, c in enumerate(classes):
        c = str(c)
        index = np.where(images_info[:, 1] == c)
        iminfo = images_info[index]
        impath = iminfo[:, 0].tolist()
        dataset.append(ImageClass(c, impath))
        if i % 500 == 0:
            print('impath2imclass: {}/{}'.format(i+1, len(classes)))

    return dataset

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*(1-split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            nrof_images_in_class = len(paths)
            split = int(math.floor(nrof_images_in_class*(1-split_ratio)))
            if split==nrof_images_in_class:
                split = nrof_images_in_class-1
            if split>=min_nrof_images_per_class and nrof_images_in_class-split>=1:
                train_set.append(ImageClass(cls.name, paths[:split]))
                test_set.append(ImageClass(cls.name, paths[split:]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set

def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc



def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    precision = np.zeros(nrof_folds)
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        precision[fold_idx], val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    prec_mean = np.mean(precision)
    prec_std = np.std(precision)
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return prec_mean, prec_std, val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)  # 这里的val实际上就是TPR(也即TAR,查全率，recall)
    far = float(false_accept) / float(n_diff)
    total_accept = true_accept + false_accept
    if total_accept == 0:
        precision = 0
    else:
        precision = float(true_accept) / float(total_accept)
    return precision, val, far

def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' +  e.strerror

    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' +  e.strerror

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names

def put_images_on_grid(images, shape=(16,8)):
    nrof_images = images.shape[0]
    img_size = images.shape[1]
    bw = 3
    img = np.zeros((shape[1]*(img_size+bw)+bw, shape[0]*(img_size+bw)+bw, 3), np.float32)
    for i in range(shape[1]):
        x_start = i*(img_size+bw)+bw
        for j in range(shape[0]):
            img_index = i*shape[0]+j
            if img_index>=nrof_images:
                break
            y_start = j*(img_size+bw)+bw
            img[x_start:x_start+img_size, y_start:y_start+img_size, :] = images[img_index, :, :, :]
        if img_index>=nrof_images:
            break
    return img

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))
