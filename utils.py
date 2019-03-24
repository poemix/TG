import numpy as np
import tensorflow as tf


def py_rpn_match(num_anchors, pos_indices, neg_indices):
    # pos:1(iou>0.7)  neu:0(0.7>iou>0.3)  neg:-1(iou<0.3)
    rpn_match = np.zeros([num_anchors], dtype=np.int32)
    rpn_match[pos_indices] = 1
    rpn_match[neg_indices] = -1
    return rpn_match


def py_rpn_target_match(iou_max):
    """

    :param iou_max: [batch, num_anchors]
    :return: rpn_target_match, rpn_target_bbox
    """
    batch, num_anchors = iou_max.shape
    rpn_target_match = np.zeros([batch, num_anchors], dtype=np.int32)
    for b in range(batch):
        pos_indices = np.where(iou_max[b] >= 0.5)[0]
        np.random.shuffle(pos_indices)

        neg_indices = np.where(iou_max[b] < 0.3)[0]
        np.random.shuffle(neg_indices)

        shuffle_pos_indices = pos_indices[:128]
        shuffle_neg_indices = neg_indices[:256 - shuffle_pos_indices.shape[0]]

        rpn_target_match[b, shuffle_pos_indices] = 1
        rpn_target_match[b, shuffle_neg_indices] = -1

    rpn_target_match = np.reshape(rpn_target_match, [batch, -1, 1])

    return rpn_target_match


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
