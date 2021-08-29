import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

def calc_TP(gt_boxes, pred_boxes, image_TP, image_GT, image_pred_num):
    temp_mask = np.ones(len(gt_boxes))
    for i in temp_mask:
        i = 1
    for k in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            iou = jaccard(gt_boxes[j], pred_boxes[k])
            if temp_mask[j] == 1 and iou >= 0.4:
                image_TP += 1
                temp_mask[j] = 0
    img_gt = len(gt_boxes)
    image_GT += img_gt
    img_pred_num = len(pred_boxes)
    image_pred_num += img_pred_num

    return image_TP, image_GT, image_pred_num


def jaccard(gt_box, pred_box):

    iw = max(0, min(gt_box[2], pred_box[2]) - max(gt_box[0], pred_box[0]))
    ih = max(0, min(gt_box[3], pred_box[3]) - max(gt_box[1], pred_box[1]))

    inter = iw * ih

    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])

    union = area_gt + area_pred - inter

    return inter / union


def calc_ap(TP, GT, pred_num_all):
    TP = np.array(TP, dtype=float)
    GT = np.array(GT, dtype=float)
    prec = TP / pred_num_all
    rec = TP / GT


    with open('map4.txt', 'w') as f:
        for i in range(len(prec)):
            f.write('tp: %f gt: %f pred: %f prec: %f rec: %f\n' % (TP[i], GT[i], pred_num_all[i], prec[i], rec[i]))

    # fig = plt.figure()
    plt.scatter(rec, prec, color='red')
    plt.show()
    ap = simps(prec, rec, dx=0.001)

    return ap
