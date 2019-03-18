import numpy as np
import tensorflow as tf

from custom_vgg19 import Vgg19
from data_loader import TXTLoader
from thumbnail_generartion import GCA, RPN
from ops import gen_anchors_op, overlaps_op, norm_boxes_op

# 构造图
# 不显示构造图的话，tensorflow默认会构造一个图，所有的op都会放在默认图中
graph = tf.Graph()
with graph.as_default():
    # 输入占位符
    image_ph = tf.placeholder(tf.float32, [1, 650, 800, 3])
    ratio_ph = tf.placeholder(tf.float32, [1, 1])
    gt_bbox_ph = tf.placeholder(tf.float32, [1, 1, 4])

    # vgg19
    vgg19 = Vgg19(vgg19_npy_path='./model/vgg19_imagenet_pretrained.npy')
    vgg19.build(image_ph)

    # GCA
    f_attn = GCA(vgg19.conv5_4)

    # RPN
    bbox_op, objectness_op = RPN(f_attn, ratio_ph)
    # bbox_op: [1, 41, 50, 3, 4]  objectness_op: [1, 41, 50, 3]

    # 获取feature map的height和width
    feature_shape = tf.shape(f_attn)
    feature_shape = tf.stack([feature_shape[1], feature_shape[2]])

    # 根据feature map的height和width产生所有的anchors
    anchors = gen_anchors_op([128, 256, 512], ratio_ph, feature_shape, 16, 1)

    rpn_bbox = tf.reshape(bbox_op, [-1, 4])  # rpn_bbox: [6150, 4]
    objectness = tf.reshape(objectness_op, [-1, 1])  # objectness: [6150, 1]
    gt_bbox = tf.reshape(gt_bbox_ph, [-1, 4])  # gt_bbox: [1, 4]
    anchors = tf.reshape(anchors, [-1, 4])  # anchors: [6150, 4]

    # 计算iou
    overlaps = overlaps_op(gt_bbox, anchors)  # overlaps: [1, 6150]
    positive_indices = tf.where(overlaps >= 0.5)  # positive_indices: [num_pos, 2]
    negative_indices = tf.where(overlaps < 0.5)  # negative_indices: [num_neg, 2]

    # 随机选取256个正负样本
    shuffle_positive_indices = tf.random_shuffle(positive_indices)[:64]
    positive_count = tf.shape(shuffle_positive_indices)[0]
    shuffle_negative_indices = tf.random_shuffle(negative_indices)[:256 - positive_count]

    # rpn classify loss
    eps = 1e-12
    positive_objectness = tf.gather(objectness, shuffle_positive_indices[:, 1])
    rpn_class_loss = tf.reduce_mean(- tf.log(positive_objectness + eps))

    # rpn bbox loss
    positive_rpn_bbox = tf.gather(rpn_bbox, shuffle_positive_indices[:, 1])

    # 归一化
    shape = tf.stack([image_ph.shape[1], image_ph.shape[2]])
    norm_gt_bbox = norm_boxes_op(gt_bbox, shape)  # norm_gt_bbox: [1, 4]
    print(norm_gt_bbox)

    # 计算smooth l1 loss
    gt_bbox = tf.tile(norm_gt_bbox, [positive_count, 1])
    diff = tf.abs(norm_gt_bbox - positive_rpn_bbox)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    rpn_bbox_loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)

    loss = rpn_class_loss + rpn_bbox_loss

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# data loader
loader = TXTLoader(root='/Users/aiyoj/Downloads/Thumbnail Data Set/PQ_Set',
                   txt_path='./data/train_set.txt',
                   batch_size=1,
                   shuffle=False)
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
                train_step,
                feed_dict={
                    image_ph: image_batch,
                    ratio_ph: np.reshape(ratio_batch, [1, 1]),
                    gt_bbox_ph: np.reshape(gt_bbox_batch, [1, 1, 4]),
                }
            )
