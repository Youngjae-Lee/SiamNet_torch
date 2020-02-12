import numpy as np
from sklearn.metrics import roc_auc_score


def calculateIou(gt_bbox, target_bbox):
    # bbox is (x1, y1, x2, y2)
    xx1 = float(max(gt_bbox['x1'], target_bbox['x1']))
    xx2 = float(min(gt_bbox['x2'], target_bbox['x2']))
    yy1 = float(max(gt_bbox['y1'], target_bbox['y1']))
    yy2 = float(min(gt_bbox['y2'], target_bbox['y2']))

    gt_w = float(gt_bbox['x2'] - gt_bbox['x1'] + 1)
    gt_h = float(gt_bbox['y2'] - gt_bbox['y1'] + 1)
    gt_area = gt_w * gt_h
    target_w = float(target_bbox['x2'] - target_bbox['x1'] + 1)
    target_h = float(target_bbox['y2'] - target_bbox['y1'] + 1)
    target_area = target_w*target_h
    inter_w = xx2 - xx1 + 1
    inter_h = yy2 - yy1 + 1
    inter_area = inter_w * inter_h
    iou = inter_area / (gt_area + target_area - inter_area)
    return iou


def AUC(output, label):
    b = output.shape[0]
    output = output.reshape(b, -1)
    mask = label[:, :, :,1].reshape(b, -1)
    label = label[:, :, :, 0].reshape(b, -1)

    total_auc = 0
    for i in range(b):
        total_auc += roc_auc_score(label[i], output[i], sample_weight=mask[i])

    return total_auc/b


def calc_center_error(output, label, upscale_factor=4):
    b = output.shape[0]
    s = output.shape[-1]
    out_flat = output.reshape(b, -1)
    max_idx = np.argmax(out_flat, axis=1)
    estim_cen = np.stack([max_idx//s, max_idx % s], axis=1)
    dist = np.linalg.norm(estim_cen - s//2, axis=1)
    c_error = dist.mean()
    c_error = c_error * upscale_factor
    return c_error






