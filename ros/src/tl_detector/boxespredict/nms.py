import numpy as np
import math
from boxespredict.Box import *

# Non maximum suppression
def NMS(final_probs, final_bbox):
    boxes = []
    indices = set()

    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]

    for class_loop in range(class_length):
        for index in range(pred_length):
            if final_probs[index, class_loop] == 0:
                continue
            for index2 in range(index + 1, pred_length):
                if final_probs[index2, class_loop] == 0:
                    continue
                if index == index2:
                    continue
                if box_iou(final_bbox[index, 0], final_bbox[index, 1], final_bbox[index, 2], final_bbox[index, 3], final_bbox[index2, 0], final_bbox[index2, 1], final_bbox[index2, 2], final_bbox[index2, 3]) >= 0.4:
                    if final_probs[index2, class_loop] > final_probs[index, class_loop]:
                        final_probs[index, class_loop] = 0
                        break
                    final_probs[index2,class_loop] = 0

            if index not in indices:
                bb = BoundBox(class_length)
                bb.x = final_bbox[index, 0]
                bb.y = final_bbox[index, 1]
                bb.w = final_bbox[index, 2]
                bb.h = final_bbox[index, 3]
                bb.c = final_bbox[index, 4]
                bb.probs = np.asarray(final_probs[index,:])
                boxes.append(bb)
                indices.add(index)
    return boxes
