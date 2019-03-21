import numpy as np


def py_rpn_match(num_anchors, pos_indices, neg_indices):
    # pos:1(iou>0.7)  neu:0(0.7>iou>0.3)  neg:-1(iou<0.3)
    rpn_match = np.zeros([num_anchors], dtype=np.int32)
    rpn_match[pos_indices] = 1
    rpn_match[neg_indices] = -1
    return rpn_match
