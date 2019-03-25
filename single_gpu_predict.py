import os
import re
import cv2
import sys
import datetime

sys.path.append("/home/shizai/xushiqi/projects/tg/")

import numpy as np
import tensorflow as tf
import pandas as pd
from model import Model
from ops import apply_box_deltas_op
from ops import clip_bbox_op

pattern = re.compile(r'\d+')

# default graph
gpus = '3'  # just only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = gpus  # 指定GPU
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

height, width = 650, 800
input_image = tf.placeholder(tf.float32, [None, height, width, 3])  # [batch, h, w, c]
input_ratio = tf.placeholder(tf.float32, [None, 1])  # [batch, num_ratios]
input_gt_bbox = tf.placeholder(tf.float32, [None, 1, 4])  # [batch, num_bbox, (y1, x1, y2, x2)]
input_window_h = tf.placeholder(tf.float32)
input_window_w = tf.placeholder(tf.float32)
input_top_pad = tf.placeholder(tf.float32)
input_left_pad = tf.placeholder(tf.float32)

batch = tf.shape(input_image)[0]
model = Model(input_image, input_ratio, input_gt_bbox,
              vgg19_npy_path='./model/vgg19_imagenet_pretrained.npy',
              trainable=False)
rpn_bbox = model.rpn_bbox
rpn_class_probs = model.rpn_class_probs
anchors = model.anchors  # [b, num_anchors, 4]

scores = rpn_class_probs[:, :, 1]

deltas = rpn_bbox
bbox_std_dev = tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)
deltas = deltas * tf.reshape(bbox_std_dev, [1, 1, 4])

# 排序
ix = tf.nn.top_k(scores, tf.shape(anchors)[1], sorted=True, name="top_anchors").indices
ix = tf.reshape(ix, [-1])
# scores = tf.batch_gather(scores, ix)
# deltas = tf.batch_gather(deltas, ix)
# anchors = tf.batch_gather(anchors)
scores = tf.gather(tf.reshape(scores, [-1]), ix)
deltas = tf.gather(tf.reshape(deltas, [-1, 4]), ix)
anchors = tf.gather(tf.reshape(anchors, [-1, 4]), ix)

pbboxes = apply_box_deltas_op(tf.reshape(anchors, [-1, 4]), tf.reshape(deltas, [-1, 4]))
# window = tf.constant([0, 0, height, width], dtype=tf.float32)
window = tf.stack([input_top_pad, input_left_pad, input_window_h + input_top_pad, input_window_w + input_left_pad])
pbboxes = clip_bbox_op(pbboxes, window)

indices = tf.image.non_max_suppression(
    pbboxes, tf.reshape(scores, [-1]), 1,
    0.7, name="rpn_non_max_suppression")
proposals = tf.gather(pbboxes, indices)

# data reader
root = '/Users/aiyoj/Downloads/Thumbnail Data Set/PQ_Set'
txt_path = './data/test_set.txt'
df = pd.read_csv(txt_path, header=None, sep=';')
images_path_df = df.apply(lambda line: '{}/{}'.format(root, line[0]), axis=1)
thumbnail_dims_df = df.apply(lambda line: list(map(int, pattern.findall(line[1]))), axis=1)
bboxes_df = df.apply(lambda line: list(map(int, pattern.findall(line[2]))), axis=1)

images_path = images_path_df.values
thumbnail_dims = thumbnail_dims_df.values
ratios_df = thumbnail_dims_df.apply(lambda line: line[1] / line[0])
ratios = ratios_df.values
bboxes = bboxes_df.values

with tf.Session(config=config) as sess:
    saver = tf.train.Saver(max_to_keep=50)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './models/restore_0_5000.ckpt')
    batch_size = 2
    images = np.empty([batch_size, height, width, 3], dtype=np.uint8)

    for index, (image_path, bbox, ratio, thumbnail_dim) in enumerate(zip(images_path, bboxes, ratios, thumbnail_dims)):
        ratio = np.array([ratio], dtype=np.float32)
        # read image file
        image = cv2.imread(image_path)
        h, w, c = image.shape
        print(h, w)

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        bbox = np.array([y1, x1, y2, x2])

        # padding image
        top_pad = (height - h) // 2
        bottom_pad = height - h - top_pad
        left_pad = (width - w) // 2
        right_pad = width - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        pad_image = np.pad(image, padding, mode='constant', constant_values=0)

        shift = np.array([top_pad, left_pad, top_pad, left_pad])
        gt_bbox = bbox + shift

        # 可视化
        # blue = (255, 0, 0)
        # cv2.rectangle(pad_image, (gt_bbox[1], gt_bbox[0]), (gt_bbox[3], gt_bbox[2]), blue, 3)
        # cv2.namedWindow("Image1")
        # cv2.imshow("Image1", pad_image)
        #
        # cv2.namedWindow("Image2")
        # cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), blue, 3)
        # cv2.imshow("Image2", image)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # norm image
        input_pad_image = pad_image / 255

        pred_bbox = sess.run(
            proposals,
            feed_dict={
                input_image: np.reshape(input_pad_image, [-1, 650, 800, 3]),
                input_ratio: np.reshape(ratio, [-1, 1]),
                input_window_h: h,
                input_window_w: w,
                input_top_pad: top_pad,
                input_left_pad: left_pad
            }
        )
        print(pred_bbox.shape, pred_bbox)
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now_time, pred_bbox.shape, pred_bbox)
        pred_bbox = np.reshape(pred_bbox, [-1])
        pred_bbox = pred_bbox.astype(np.int32)

        red = (0, 0, 255)
        cv2.rectangle(pad_image, (pred_bbox[1], pred_bbox[0]), (pred_bbox[3], pred_bbox[2]), red, 3)
        green = (0, 255, 0)
        cv2.rectangle(pad_image, (gt_bbox[1], gt_bbox[0]), (gt_bbox[3], gt_bbox[2]), green, 3)
        cv2.namedWindow("Image1")
        cv2.imshow("Image1", pad_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite('./2.jpg', pad_image[top_pad:top_pad + h, left_pad:left_pad + w, :])
        # break
