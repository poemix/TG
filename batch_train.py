import os
import sys
import numpy as np
import tensorflow as tf

from custom_vgg19 import Vgg19
from data_loader import TXTLoader
from thumbnail_generartion import GCA, RPN
from ops import gen_anchors_op, overlaps_op
from ops import norm_bbox_op, rpn_class_loss_op
from ops import rpn_bbox_loss_op, batch_overlaps_op
from ops import bbox_refinement_op
from utils import py_rpn_match
from utils import py_rpn_target_match
from ops import rpn_target_bbox_op

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定GPU
sys.path.append("/home/shizai/xushiqi/projects/tg/")

# 构造图
# 不显示构造图的话，tensorflow默认会构造一个图，所有的op都会放在默认图中
graph = tf.Graph()
with graph.as_default():
    # 输入占位符
    input_image = tf.placeholder(tf.float32, [None, 650, 800, 3])  # [batch, h, w, c]
    input_ratio = tf.placeholder(tf.float32, [None, 1])  # [batch, num_ratios]
    input_gt_bbox = tf.placeholder(tf.float32, [None, 1, 4])  # [batch, num_bbox, (y1, x1, y2, x2)]
    num_bboxes = tf.shape(input_gt_bbox)[0]

    # vgg19
    vgg19 = Vgg19(vgg19_npy_path='./model/vgg19_imagenet_pretrained.npy')
    vgg19.build(input_image)

    # GCA
    f_attn = GCA(vgg19.conv5_4)  # f_attn: [b, h, w, 512]

    # RPN
    rpn_bbox, rpn_class_logits = RPN(f_attn, input_ratio)
    # rpn_bbox: [b, h, w, num_scales, 4]
    # rpn_class_logits: [b, h, w, num_scales, 2]

    # 获取feature map的height和width
    fmap_shape = tf.shape(f_attn)
    b, h, w, c = fmap_shape[0], fmap_shape[1], fmap_shape[2], fmap_shape[3]
    spatial_shape = tf.stack([h, w])

    rpn_class_logits = tf.reshape(rpn_class_logits, [b, -1, 2])
    rpn_bbox = tf.reshape(rpn_bbox, [b, -1, 4])

    # 根据feature map的height和width产生所有的anchors
    scales = [128, 256, 512]
    feature_stride, anchor_stride = 16, 1
    anchors = gen_anchors_op(scales, input_ratio, spatial_shape, feature_stride, anchor_stride)
    # anchors: [b, h, w, num_scales, 4]

    anchors = tf.reshape(anchors, [b, -1, 4])  # anchors: [b, num_anchors, 4]
    num_anchors = tf.shape(anchors)[1]

    # 计算iou
    overlaps = batch_overlaps_op(anchors, input_gt_bbox)  # overlaps: [b, num_anchors, num_bboxes]
    anchor_iou_argmax = tf.argmax(overlaps, axis=2, output_type=tf.int32)  # iou_argmax: [b, num_anchors]
    anchor_iou_argmax_ = tf.reshape(anchor_iou_argmax, [-1])
    indices = tf.stack([tf.range(b * num_anchors), anchor_iou_argmax_], axis=1)
    anchor_iou_max_ = tf.gather_nd(tf.reshape(overlaps, [-1, num_bboxes]), indices)
    anchor_iou_max = tf.reshape(anchor_iou_max_, [b, num_anchors])
    print(anchor_iou_max)

    # rpn_target_match
    rpn_target_match = tf.py_func(
        py_rpn_target_match,
        [anchor_iou_max],
        tf.int32
    )

    # rpn_target_bbox
    target_bbox_ = tf.gather(tf.reshape(input_gt_bbox, [-1, 4]), anchor_iou_argmax_)
    target_bbox = tf.reshape(target_bbox_, [b, -1, 4])

    rpn_target_bbox_ = rpn_target_bbox_op(tf.reshape(anchors, [-1, 4]), tf.reshape(target_bbox, [-1, 4]))
    rpn_target_bbox = tf.reshape(rpn_target_bbox_, [b, -1, 4])

    rpn_class_loss = rpn_class_loss_op(rpn_target_match, rpn_class_logits)

    rpn_bbox_loss = rpn_bbox_loss_op(rpn_target_bbox, rpn_target_match, rpn_bbox)

    loss = rpn_class_loss + 10 * rpn_bbox_loss
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# data loader
loader = TXTLoader(root='/Users/aiyoj/Downloads/Thumbnail Data Set/PQ_Set',
                   txt_path='./data/train_set.txt',
                   batch_size=1,
                   shuffle=True)
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
            _, loss_value = sess.run(
                [train_step, loss],
                feed_dict={
                    input_image: image_batch,
                    input_ratio: np.reshape(ratio_batch, [1, 1]),
                    input_gt_bbox: np.reshape(gt_bbox_batch, [1, 1, 4]),
                }
            )
            print(_, loss_value)
