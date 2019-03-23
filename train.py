import os
import sys
import numpy as np
import tensorflow as tf

from custom_vgg19 import Vgg19
from data_loader import TXTLoader
from thumbnail_generartion import GCA, RPN
from ops import gen_anchors_op, overlaps_op, norm_bbox_op, rpn_class_loss_op, rpn_bbox_loss_op
from ops import bbox_refinement_op
from utils import py_rpn_match

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定GPU
sys.path.append("/home/shizai/xushiqi/projects/tg/")

RPN_BBOX_STD_DEV = tf.constant([0.1, 0.1, 0.2, 0.2])

# 构造图
# 不显示构造图的话，tensorflow默认会构造一个图，所有的op都会放在默认图中
graph = tf.Graph()
with graph.as_default():
    # 输入占位符
    image_ph = tf.placeholder(tf.float32, [1, 650, 800, 3])  # [batch, h, w, c]
    ratio_ph = tf.placeholder(tf.float32, [1, 1])  # [batch, num_ratios]
    gt_bbox_ph = tf.placeholder(tf.float32, [1, 1, 4])  # [batch, num_bbox, (y1, x1, y2, x2)]

    # vgg19
    vgg19 = Vgg19(vgg19_npy_path='./model/vgg19_imagenet_pretrained.npy')
    vgg19.build(image_ph)

    # GCA
    f_attn = GCA(vgg19.conv5_4)  # f_attn: [batch, h, w, 512]

    # RPN
    bbox_op, objectness_op = RPN(f_attn, ratio_ph)
    # bbox_op: [batch, h, w, num_scales, 4]
    # objectness_op: [batch, h, w, num_scales, 2]

    # 获取feature map的height和width
    shape = tf.shape(f_attn)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]
    feature_shape = tf.stack([h, w])

    # 根据feature map的height和width产生所有的anchors
    scales = [128, 256, 512]
    feature_stride, anchor_stride = 16, 1
    anchors = gen_anchors_op(scales, ratio_ph, feature_shape, feature_stride, anchor_stride)
    # anchors: [batch, h, w, num_scales, 4]

    rpn_bbox = tf.reshape(bbox_op, [-1, 4])  # rpn_bbox: [batch*num_anchor, 4]
    rpn_objectness = tf.reshape(objectness_op, [1, -1, 2])  # rpn_objectness: [batch*num_anchor, 2]
    gt_bbox = tf.reshape(gt_bbox_ph, [-1, 4])  # gt_bbox: [batch*num_bbox, 4]
    anchors = tf.reshape(anchors, [-1, 4])  # anchors: [batch*num_anchor, 4]

    # 计算iou
    overlaps = overlaps_op(anchors, gt_bbox)  # overlaps: [batch*num_anchor, 1]
    # 注：overlaps_op暂不支持batch操作，batch默认为1
    iou_argmax_ = tf.argmax(overlaps, axis=1)
    iou_argmax = tf.reshape(iou_argmax_, [-1, 1])
    index = tf.reshape(tf.range(tf.shape(overlaps)[0]), [-1, 1])
    index = tf.cast(index, tf.int64)
    indices = tf.concat([index, iou_argmax], axis=1)
    iou_max = tf.gather_nd(overlaps, indices)

    # 每个anchor都会对应一个gt_bbox
    target_bbox_ = tf.gather(gt_bbox, iou_argmax_)  # target_bbox: [batch*num_anchor, 4]
    # iou_max2 = tf.reduce_max(overlaps, axis=1)  # iou_max2: [batch*num_anchor]

    # 计算正样本索引
    pos_indices = tf.where(iou_max >= 0.5)  # pos_indices: [batch*num_pos, 1]
    pos_indices = tf.reshape(pos_indices, [-1])  # pos_indices: [batch*num_pos]

    # 计算中立样本索引
    neu_indices = tf.where(tf.logical_and(iou_max > 0.3, iou_max < 0.7))  # neu_indices: [batch*num_neu, 1]
    neu_indices = tf.reshape(neu_indices, [-1])  # neu_indices: [batch*num_neu]

    # 计算负样本索引
    neg_indices = tf.where(iou_max <= 0.3)  # neg_indices: [batch*num_neg, 1]
    neg_indices = tf.reshape(neg_indices, [-1])  # neg_indices: [batch*num_neg]

    # 随机选取256个正负样本
    shuffle_pos_indices = tf.random_shuffle(pos_indices)[:128]
    pos_count = tf.shape(shuffle_pos_indices)[0]
    shuffle_neg_indices = tf.random_shuffle(neg_indices)[:256 - pos_count]
    neg_count = tf.shape(shuffle_neg_indices)[0]

    # 制作rpn中anchors的0、1标签
    rpn_match = tf.py_func(
        py_rpn_match,
        [tf.shape(anchors)[0], shuffle_pos_indices, shuffle_neg_indices],
        tf.int32)

    # rpn classify loss
    rpn_class_loss = rpn_class_loss_op(tf.reshape(rpn_match, [b, tf.shape(anchors)[0], 1]), rpn_objectness)

    target_bbox = bbox_refinement_op(anchors, target_bbox_)
    rpn_bbox_loss = rpn_bbox_loss_op(
        tf.reshape(target_bbox, [1, -1, 4]),
        tf.reshape(rpn_match, [1, -1, 1]),
        tf.reshape(rpn_bbox, [1, -1, 4]))

    loss = rpn_class_loss + 10 * rpn_bbox_loss

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# data loader
loader = TXTLoader(root='/Users/aiyoj/Downloads/Thumbnail Data Set/PQ_Set',
                   txt_path='./data/train_set.txt',
                   batch_size=1,
                   shuffle=False)
# loader = TXTLoader(root='./data/Thumbnail Data Set/PQ_Set',
#                    txt_path='./data/train_set.txt',
#                    batch_size=1,
#                    shuffle=True)
num_epoch = 10
num_batch = 60000

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
# 将上面的构造好的图graph传给session，这样session就可以run graph中的op
with tf.Session(graph=graph, config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epoch):
        for step in range(epoch * num_batch, (epoch + 1) * num_batch):
            image_batch, gt_bbox_batch, thumbnail_dim_batch, ratio_batch, meta_batch, name_batch = loader.batch()
            _ = sess.run(
                [iou_max],
                feed_dict={
                    image_ph: image_batch,
                    ratio_ph: np.reshape(ratio_batch, [1, 1]),
                    gt_bbox_ph: np.reshape(gt_bbox_batch, [1, 1, 4]),
                }
            )
            print(_[0])
            break
