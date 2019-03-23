import os
import sys
import datetime

sys.path.append("/home/shizai/xushiqi/projects/tg/")

import numpy as np
import tensorflow as tf
from utils import average_gradients
from custom_vgg19 import Vgg19
from thumbnail_generartion import GCA, RPN
from ops import gen_anchors_op
from ops import rpn_bbox_loss_op, rpn_class_loss_op
from utils import py_rpn_target_match
from ops import batch_overlaps_op
from ops import rpn_target_bbox_op
from data_loader import TXTLoader

gpus = '3,5,6,7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpus  # 指定GPU
num_gpus = len(gpus.split(','))
n_sample = 63043
batch_size = num_gpus * 1
num_batch = n_sample // batch_size


def build_model(image, ratio, gt_bbox):
    num_bboxes = tf.shape(gt_bbox)[0]

    # vgg19
    vgg19 = Vgg19(vgg19_npy_path='./model/vgg19_imagenet_pretrained.npy')
    vgg19.build(image)

    # GCA
    f_attn = GCA(vgg19.conv5_4)  # f_attn: [b, h, w, 512]

    # RPN
    rpn_bbox, rpn_class_logits = RPN(f_attn, ratio)
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
    anchors = gen_anchors_op(scales, ratio, spatial_shape, feature_stride, anchor_stride)
    # anchors: [b, h, w, num_scales, 4]

    anchors = tf.reshape(anchors, [b, -1, 4])  # anchors: [b, num_anchors, 4]
    num_anchors = tf.shape(anchors)[1]

    # 计算iou
    overlaps = batch_overlaps_op(anchors, gt_bbox)  # overlaps: [b, num_anchors, num_bboxes]
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
    target_bbox_ = tf.gather(tf.reshape(gt_bbox, [-1, 4]), anchor_iou_argmax_)
    target_bbox = tf.reshape(target_bbox_, [b, -1, 4])

    rpn_target_bbox_ = rpn_target_bbox_op(tf.reshape(anchors, [-1, 4]), tf.reshape(target_bbox, [-1, 4]))
    rpn_target_bbox = tf.reshape(rpn_target_bbox_, [b, -1, 4])

    rpn_class_loss = rpn_class_loss_op(rpn_target_match, rpn_class_logits)

    rpn_bbox_loss = rpn_bbox_loss_op(rpn_target_bbox, rpn_target_match, rpn_bbox)

    loss = rpn_class_loss + 10 * rpn_bbox_loss

    return loss


loader = TXTLoader(root='./data/Thumbnail Data Set/PQ_Set',
                   txt_path='./data/train_set.txt',
                   batch_size=batch_size,
                   shuffle=True)
num_epoch = 10
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

input_image = tf.placeholder(tf.float32, [None, 650, 800, 3])  # [batch, h, w, c]
input_ratio = tf.placeholder(tf.float32, [None, 1])  # [batch, num_ratios]
input_gt_bbox = tf.placeholder(tf.float32, [None, 1, 4])  # [batch, num_bbox, (y1, x1, y2, x2)]

image_splits = tf.split(input_image, num_gpus)
ratio_splits = tf.split(input_ratio, num_gpus)
gt_bbox_splits = tf.split(input_gt_bbox, num_gpus)

opt = tf.train.AdamOptimizer(0.001)
global_step = tf.Variable(0, name='global_step', trainable=False)

tower_grads = []
tower_loss = []
counter = 0

with tf.variable_scope(tf.get_variable_scope()):
    for d in range(num_gpus):
        with tf.device('/gpu:{}'.format(d)):
            with tf.name_scope('{}_{}'.format('tower', d)):
                loss = build_model(image_splits[counter], ratio_splits[counter], gt_bbox_splits[counter])
                tf.get_variable_scope().reuse_variables()
                counter += 1
                with tf.variable_scope("loss"):
                    grads_vars_all = opt.compute_gradients(loss)
                    tower_grads.append(grads_vars_all)
                    tower_loss.append(loss)

mean_loss = tf.stack(axis=0, values=tower_loss)
mean_loss = tf.reduce_mean(mean_loss, 0)
mean_grads = average_gradients(tower_grads)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = opt.apply_gradients(mean_grads, global_step=global_step)

with tf.Session(config=config) as sess:
    saver = tf.train.Saver(max_to_keep=50)
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epoch):
        for step in range(epoch * num_batch, (epoch + 1) * num_batch):
            image_batch, gt_bbox_batch, thumbnail_dim_batch, ratio_batch, meta_batch, name_batch = loader.batch()

            _, loss_value = sess.run(
                [train_op, mean_loss],
                feed_dict={
                    input_image: image_batch,
                    input_ratio: np.reshape(ratio_batch, [-1, 1]),
                    input_gt_bbox: np.reshape(gt_bbox_batch, [-1, 1, 4]),
                }
            )
            if step % 1000 == 0:
                now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(now_time, step, loss_value)
        saver.save(sess, '{}_{}.ckpt'.format(epoch, step))
