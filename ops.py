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
    """Computes binary cross entropy given `preds`.

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


def gen_anchors_op(scales, ratios, feature_shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratio: 2D array of anchor ratios of width/height. Example: [[0.5], [0.9]]  shape:[N, 1]
    fm_shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    feature_shape = tf.convert_to_tensor(feature_shape)

    scales = tf.convert_to_tensor(scales)
    scales = tf.cast(scales, tf.float32)
    num_scales = tf.shape(scales)[0]

    ratios = tf.convert_to_tensor(ratios)
    batch_size = tf.shape(ratios)[0]

    # Get all combinations of scales and ratios
    scales, ratios = tf.meshgrid(scales, ratios)
    # scales:[batch_size, num_scales], ratios:[batch_size, num_scales]

    # Enumerate heights and widths from scales and ratios
    heights = scales / tf.sqrt(ratios)
    widths = scales * tf.sqrt(ratios)
    # heights:[batch_size, num_scales], widths:[batch_size, num_scales]

    # Enumerate shifts in feature space
    shifts_y = tf.range(0, feature_shape[0], anchor_stride) * feature_stride
    shifts_x = tf.range(0, feature_shape[1], anchor_stride) * feature_stride
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
    boxes = tf.reshape(boxes, tf.stack([feature_shape[0], feature_shape[1], batch_size, num_scales, 4]))
    boxes = tf.transpose(boxes, [2, 0, 1, 3, 4])

    return boxes


def clip_boxes_op(boxes, window):
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


def norm_boxes_op(boxes, shape):
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


if __name__ == '__main__':
    with tf.Session() as sess:
        boxes = gen_anchors_op([128, 256], [[0.5], [0.5], [0.9]], [14, 14], 32, 1)
        result = sess.run(boxes)
        print(result.shape)
        print(333, result[0][0][0][0])
        print(444, result[0][0][1][0])

        print(256, result[0][0][0][1])
        print(256, result[0][0][1][1])

        print(333, result[1][0][0][0])
        print(444, result[1][0][1][0])
