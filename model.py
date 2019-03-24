import tensorflow as tf
from custom_vgg19 import Vgg19
from thumbnail_generartion import GCA
from thumbnail_generartion import RPN
from ops import gen_anchors_op
from ops import batch_overlaps_op
from ops import rpn_target_bbox_op
from ops import rpn_class_loss_op
from ops import rpn_bbox_loss_op
from utils import py_rpn_target_match


class Model(object):
    def __init__(self, image, ratio, gt_bbox,
                 vgg19_npy_path='./model/vgg19_imagenet_pretrained.npy',
                 trainable=True):
        self.image = image
        self.ratio = ratio
        self.gt_bbox = gt_bbox
        self.build_model(image, ratio, gt_bbox, vgg19_npy_path, trainable)

    def build_model(self, image, ratio, gt_bbox, vgg19_npy_path, trainable):
        num_bboxes = tf.shape(gt_bbox)[0]

        # vgg19
        vgg19 = Vgg19(vgg19_npy_path=vgg19_npy_path, trainable=trainable)
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
        rpn_class_probs = tf.nn.softmax(rpn_class_logits)

        self.rpn_bbox = rpn_bbox
        self.rpn_class_logits = rpn_class_logits
        self.rpn_class_probs = rpn_class_probs

        # 根据feature map的height和width产生所有的anchors
        scales = [128, 256, 512]
        feature_stride, anchor_stride = 16, 1
        anchors = gen_anchors_op(scales, ratio, spatial_shape, feature_stride, anchor_stride)
        # anchors: [b, h, w, num_scales, 4]

        anchors = tf.reshape(anchors, [b, -1, 4])  # anchors: [b, num_anchors, 4]

        self.anchors = anchors

        num_anchors = tf.shape(anchors)[1]

        # 计算iou
        overlaps = batch_overlaps_op(anchors, gt_bbox)  # overlaps: [b, num_anchors, num_bboxes]
        anchor_iou_argmax = tf.argmax(overlaps, axis=2, output_type=tf.int32)  # iou_argmax: [b, num_anchors]
        anchor_iou_argmax_ = tf.reshape(anchor_iou_argmax, [-1])
        indices = tf.stack([tf.range(b * num_anchors), anchor_iou_argmax_], axis=1)
        anchor_iou_max_ = tf.gather_nd(tf.reshape(overlaps, [-1, num_bboxes]), indices)
        anchor_iou_max = tf.reshape(anchor_iou_max_, [b, num_anchors])

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

        self.rpn_class_loss = rpn_class_loss
        self.rpn_bbox_loss = rpn_bbox_loss
        self.loss = loss


if __name__ == '__main__':
    pass
