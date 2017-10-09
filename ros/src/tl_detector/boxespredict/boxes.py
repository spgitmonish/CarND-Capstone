import numpy as np
from math import *
from boxespredict.nms import *

# Logistic function
def expit(x):
    y = 1/(1 + exp(-x))
    return y

# Maximum of two numbers
def max(a, b):
    if(a > b):
        return a
    return b

# Construct the boxes
def box_constructor(meta, net_out_in):
    threshold = meta['thresh']
    anchors = np.asarray(meta['anchors'])
    H, W, _ = meta['out_size']
    C = meta['classes']
    B = meta['num']
    arr_max = 0
    sum_of_classes = 0

    net_out = net_out_in.reshape([H, W, B, net_out_in.shape[2]/B])
    Classes = net_out[:, :, :, 5:]
    Bbox_pred =  net_out[:, :, :, :5]
    probs = np.zeros((H, W, B, C), dtype=np.float32)

    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                arr_max = 0
                sum_of_classes = 0
                Bbox_pred[row, col, box_loop, 4] = expit(Bbox_pred[row, col, box_loop, 4])
                Bbox_pred[row, col, box_loop, 0] = (col + expit(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + expit(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = exp(Bbox_pred[row, col, box_loop, 2]) * anchors[2 * box_loop + 0] / W
                Bbox_pred[row, col, box_loop, 3] = exp(Bbox_pred[row, col, box_loop, 3]) * anchors[2 * box_loop + 1] / H

                # Softmax Block
                for class_loop in range(C):
                    arr_max = max(arr_max, Classes[row, col, box_loop, class_loop])

                for class_loop in range(C):
                    Classes[row, col, box_loop, class_loop] = exp(Classes[row, col, box_loop, class_loop] - arr_max)
                    sum_of_classes += Classes[row, col, box_loop, class_loop]

                for class_loop in range(C):
                    temp = Classes[row, col, box_loop, class_loop] * Bbox_pred[row, col, box_loop, 4]/sum_of_classes
                    if(temp > threshold):
                        probs[row, col, box_loop, class_loop] = temp

    # Non-Maximum Suppression
    return NMS(np.ascontiguousarray(probs).reshape(H*W*B, C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W, 5))
