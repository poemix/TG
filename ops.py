import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None,
                                            epsilon=self.epsilon, scale=True, scope=self.name)


def binary_cross_entropy(preds, targets, name=None):
    """
    Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                                (1. - targets) * tf.log(1. - preds + eps)))


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("Bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def gen_anchors_op(scales, ratios, spatial_shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 2D array of anchor ratios of width/height. Example: [[0.5], [0.9]]  shape:[batch, 1]
    spatial_shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.

    Returns:
        anchors: [batch, height, width, num_scales, 4]
    """
    spatial_shape = tf.convert_to_tensor(spatial_shape)

    scales = tf.convert_to_tensor(scales)
    scales = tf.cast(scales, tf.float32)
    num_scales = tf.shape(scales)[0]

    ratios = tf.convert_to_tensor(ratios)
    batch_size = tf.shape(ratios)[0]
    num_ratios = tf.shape(ratios[0])[0]

    # Get all combinations of scales and ratios
    scales, ratios = tf.meshgrid(scales, ratios)
    # scales:[batch, num_scales], ratios:[batch, num_scales]

    # Enumerate heights and widths from scales and ratios
    heights = scales / tf.sqrt(ratios)
    widths = scales * tf.sqrt(ratios)
    # heights:[batch, num_scales], widths:[batch, num_scales]

    # Enumerate shifts in feature space
    shifts_y = tf.range(0, spatial_shape[0], anchor_stride) * feature_stride
    shifts_x = tf.range(0, spatial_shape[1], anchor_stride) * feature_stride

    shifts_y = tf.cast(shifts_y, tf.float32)
    shifts_x = tf.cast(shifts_x, tf.float32)
    shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)
    # shifts_x:[h, w], shifts_y:[h, w]

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)  # (n, 3) (n, 3)
    box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)  # (n, 3) (n, 3)

    # Reshape to get a list of (y, x) and a list of (h, w)
    # (n, 3, 2) -> (3n, 2)
    box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=2), [-1, 2])
    box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2), [-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = tf.concat([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
    boxes = tf.reshape(boxes, tf.stack([spatial_shape[0], spatial_shape[1], batch_size, num_ratios * num_scales, 4]))
    boxes = tf.transpose(boxes, [2, 0, 1, 3, 4])

    return boxes


def clip_bbox_op(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)

    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    return clipped


def batch_overlaps_op(boxes1, boxes2):
    """

    Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [batch, N, (y1, x1, y2, x2)].

    :param boxes1: [batch, num_anchors, 4]
    :param boxes2: [batch, num_boxes, 4]
    :return: [batch, num_anchors, num_boxes]
    """

    assert_op = tf.assert_equal(tf.shape(boxes1)[0], tf.shape(boxes2)[0])
    with tf.control_dependencies([assert_op]):
        # 1. Tile boxes2 and repeate boxes1. This allows us to compare
        # every boxes1 against every boxes2 without loops.
        batch = tf.shape(boxes1)[0]
        num_boxes1 = tf.shape(boxes1)[1]
        num_boxes2 = tf.shape(boxes2)[1]
        b1 = tf.reshape(tf.tile(boxes1, [1, num_boxes2, 1]), [-1, 4])
        b2 = tf.reshape(tf.tile(boxes2, [1, num_boxes1, 1]), [-1, 4])

        # 2. Compute intersections
        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection

        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = tf.reshape(iou, [batch, num_boxes1, num_boxes2])
    return overlaps


def overlaps_op(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def norm_bbox_op(boxes, shape):
    """
    Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_bboxes_op(boxes, shape):
    """
    Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)


def bbox_refinement_op(box, gt_box):
    """
    Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    # Convert coordinates(坐标) to center plus width/height.
    # anchor
    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    # gt box
    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    # Compute the bbox refinement that the RPN should predict.
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)

    # Normalize
    bbox_std_dev = tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)
    result = result / bbox_std_dev

    return result


def smooth_l1_loss_op(y_true, y_pred):
    """
    Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def batch_pack_op(x, counts, num_rows):
    """
    Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def rpn_bbox_loss_op(target_bbox, rpn_match, rpn_bbox):
    """
    Return the RPN bounding box loss graph.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = tf.squeeze(rpn_match, -1)  # rpn_match: [batch, anchors]
    indices = tf.where(tf.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    target_bbox = tf.gather_nd(target_bbox, indices)

    # # Trim target bounding box deltas to the same length as rpn_bbox.
    # batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)  # batch_counts: [batch]
    #
    # target_bbox = batch_pack_op(target_bbox, batch_counts)

    loss = smooth_l1_loss_op(target_bbox, rpn_bbox)

    loss = tf.cond(tf.size(loss) > 0, lambda: tf.reduce_mean(loss), lambda: tf.constant(0.0))
    return loss


def rpn_class_loss_op(rpn_match, rpn_class_logits):
    """
    RPN anchor classifier loss.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.not_equal(rpn_match, 0))

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)  # anchor_class: [num_pos+num_neg]

    # Cross entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=anchor_class, logits=rpn_class_logits)
    loss = tf.cond(tf.size(loss) > 0, lambda: tf.reduce_mean(loss), lambda: tf.constant(0.0))
    return loss


def rpn_match_op(num_anchors, pos_indices, neg_indices):
    def py_rpn_match(num_anchors_, pos_indices_, neg_indices_):
        # pos:1(iou>0.7)  neu:0(0.7>iou>0.3)  neg:-1(iou<0.3)
        rpn_match = np.zeros([num_anchors_], dtype=np.int32)
        rpn_match[pos_indices_] = 1
        rpn_match[neg_indices_] = -1
        return rpn_match

    match = tf.py_func(
        py_rpn_match,
        [num_anchors, pos_indices, neg_indices],
        tf.int32)
    return match


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def trim_zeros_op(boxes, name='trim_zeros'):
    """
    Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def apply_box_deltas_op(boxes, deltas):
    """
    Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")

    return result


def build_rpn_targets(anchors, gt_bboxes):
    """
    Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    :param anchors: [batch, num_anchors, (y1, x1, y2, x2)]
    :param gt_bboxes: [batch, num_gt_boxes, (y1, x1, y2, x2)]

    :return: rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
             rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    num_anchors = tf.shape(anchors)[1]
    num_bboxes = tf.shape(gt_bboxes)[0]
    overlaps = batch_overlaps_op(anchors, gt_bboxes)
    iou_argmax = tf.argmax(overlaps, axis=2, output_type=tf.int32)  # iou_argmax: [b, num_anchors]
    iou_argmax_ = tf.reshape(iou_argmax, [-1])
    indices = tf.stack([tf.range(b * num_anchors), iou_argmax_], axis=1)
    iou_max_ = tf.gather_nd(tf.reshape(overlaps, [-1, num_bboxes]), indices)
    iou_max = tf.reshape(iou_max_, [b, num_anchors])  # iuo_max: [b, num_anchors]


def rpn_target_bbox_op(box, gt_box):
    """
    Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    # Convert coordinates(坐标) to center plus width/height.
    # anchor
    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    # gt box
    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    # Compute the bbox refinement that the RPN should predict.
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)

    # Normalize
    bbox_std_dev = tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)
    result = result / bbox_std_dev

    return result


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定GPU
    with tf.Session() as sess:
        # boxes = gen_anchors_op([128, 256, 512], [[0.5, 1., 2]], [60, 40], 1, 1)
        # result = sess.run(boxes)
        # print(result.shape)
        # print(result[0][0][0])
        #
        # re2 = generate_anchors([128, 256, 512], [0.5, 1., 2], [60, 40], 16, 1)
        # print(re2.reshape([-1, 60, 40, 9, 4])[0][0][0])
        a = tf.constant([
            [0, 2, 3, 4],
            [1, 3, 4, 5]
        ])
        b = tf.constant([
            [0, 2, 3, 4]
        ])
        o = overlaps_op(a, b)
        ro = sess.run(o)
        print(ro)

        a1 = tf.constant([
            [[1, 2, 3, 4], [1, 3, 4, 5]],
            [[0, 2, 3, 4], [1, 3, 4, 5]]
        ])
        print(a1.shape)
        b1 = tf.constant([
            [[1, 3, 4, 5]],
            [[0, 2, 3, 4]]
        ])
        print(b1.shape)
        o1 = batch_overlaps_op(a1, b1)
        ro1 = sess.run(o1)
        print(ro1[1])
        print(ro1.shape)

# [[ -84.  -40.   99.   55.]
# [-176.  -88.  191.  103.]
# [-360. -184.  375.  199.]
# [ -56.  -56.   71.   71.]
# [-120. -120.  135.  135.]
# [-248. -248.  263.  263.]
# [ -36.  -80.   51.   95.]
# [ -80. -168.   95.  183.]
# [-168. -344.  183.  359.]]
