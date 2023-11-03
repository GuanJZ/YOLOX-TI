import copy
import os
from tqdm import tqdm
import tempfile
import json

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

fine2coarse = {}
fine2coarse['van'] = 'car'
fine2coarse['car'] = 'car'
fine2coarse['bus'] = 'big_vehicle'
fine2coarse['truck'] = 'big_vehicle'
fine2coarse['cyclist'] = 'cyclist'
fine2coarse['motorcyclist'] = 'cyclist'
fine2coarse['tricyclist'] = 'cyclist'
fine2coarse['pedestrian'] = 'pedestrian'
fine2coarse['barrow'] = 'pedestrian'

CLASS_NAMES = ['pedestrian', 'cyclist', 'car', 'big_vehicle']

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def compute_location(detections, labels):
    """
    iou>=0.5的情况下,detetcions 匹配到labels,从而得到 detections的location
    :param detections: [xyxy, conf, cls, ...]
    :param labels: [cls, xyxy, ...]
    :return:
    """
    iou = box_iou(detections[:, :4], labels[:, 1:5])
    x = torch.where(iou >= 1e-6)
    matches = None
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return matches

gt_dir = "datasets/Rope3D/labels_raw/val"
pred_dir = "datasets/Rope3D/preds_kitti_MONO_2.5D_fp32/val"
imgs_dir = "datasets/Rope3D/val"


gt_coco_json_file = 'datasets/Rope3D/annotations_2.5D/instances_val.json'
pred_coco_json_file = "datasets/Rope3D/predictions_2.5D_fp32/instances_val.json"

if True:
    coco_gt = COCO(gt_coco_json_file)

    coco_pred = coco_gt.loadRes(pred_coco_json_file)

    # 初始化评估器
    cocoEval = COCOeval(coco_gt, coco_pred, 'bbox')

    # 运行评估
    cocoEval.evaluate()
    cocoEval.accumulate()

    val_dataset_img_count = cocoEval.cocoGt.imgToAnns.__len__()
    val_dataset_anns_count = 0
    label_count_dict = {"images":set(), "anns":0}
    label_count_dicts = [copy.deepcopy(label_count_dict) for _ in range(len(CLASS_NAMES))]

    for _, ann_i in cocoEval.cocoGt.anns.items():
        if ann_i["ignore"]:
            continue
        val_dataset_anns_count += 1
        nc_i = ann_i["category_id"]
        label_count_dicts[nc_i]["images"].add(ann_i["image_id"])
        label_count_dicts[nc_i]["anns"] += 1

    s = ('%-16s' + '%12s' * 7) % ('Class', 'Labeled_images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
    print(s)

    #IOU , all p, all cats, all gt, maxdet 100
    coco_p = cocoEval.eval['precision']
    coco_p_all = coco_p[:, :, :, 0, 2]
    map = np.mean(coco_p_all[coco_p_all>-1])

    coco_p_iou50 = coco_p[0, :, :, 0, 2]
    map50 = np.mean(coco_p_iou50[coco_p_iou50>-1])
    mp = np.array([np.mean(coco_p_iou50[ii][coco_p_iou50[ii]>-1]) for ii in range(coco_p_iou50.shape[0])])
    mr = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    mf1 = 2 * mp * mr / (mp + mr + 1e-16)
    i = mf1.argmax()  # max F1 index

    pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
    print(pf % ('all', val_dataset_img_count, val_dataset_anns_count, mp[i], mr[i], mf1[i], map50, map))

    #compute each class best f1 and corresponding p and r
    for nc_i in range(len(CLASS_NAMES)):
        coco_p_c = coco_p[:, :, nc_i, 0, 2]
        map = np.mean(coco_p_c[coco_p_c>-1])

        coco_p_c_iou50 = coco_p[0, :, nc_i, 0, 2]
        map50 = np.mean(coco_p_c_iou50[coco_p_c_iou50>-1])
        p = coco_p_c_iou50
        r = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        f1 = 2 * p * r / (p + r + 1e-16)
        i = f1.argmax()
        print(pf % (CLASS_NAMES[nc_i], len(label_count_dicts[nc_i]["images"]), label_count_dicts[nc_i]["anns"], p[i], r[i], f1[i], map50, map))

if False:
    # 可视化
    gts_file, preds_file, imgs_file = sorted(os.listdir(gt_dir)), sorted(os.listdir(pred_dir)), sorted(os.listdir(imgs_dir))
    img_paths = [os.path.join(imgs_dir, img_file) for img_file in imgs_file]
    # 读取
    print("read predictions and labels to list ...")
    preds_3d, labels_3d = [], []
    for pred_3d_file, label_3d_file in tqdm(zip(preds_file, gts_file)):
        pred_3d = np.loadtxt(os.path.join(pred_dir, pred_3d_file))
        nonzero_rows = np.any(pred_3d, axis=1)
        pred_3d = pred_3d[nonzero_rows]
        preds_3d.append(pred_3d if len(pred_3d.shape) == 2 else pred_3d.reshape(-1, 1))

        with open(os.path.join(gt_dir, label_3d_file), "r") as f:
            labels = [
                x.split() for x in f.read().strip().splitlines() if len(x)
            ]

            label_3d = []
            for lbs in labels:
                if lbs[0] not in fine2coarse.keys():
                    continue
                lbs[0] = CLASS_NAMES.index(fine2coarse[lbs[0]])
                label_3d.append(lbs)

            label_3d = [
                [float(lb) for lb in lbs] for lbs in label_3d
            ]

            label_3d = [
                lbs for lbs in label_3d if abs(lbs[8]) >= 1e-6 and abs(lbs[9]) >= 1e-6 and abs(lbs[10]) >= 1e-6
            ]

            label_3d = np.array(label_3d, dtype=np.float32)

        labels_3d.append(label_3d if len(label_3d.shape) == 2 else label_3d.reshape(-1, 1))

    # 赋值距离
    for pred_3d, label_3d in tqdm(zip(preds_3d, labels_3d)):
        if len(label_3d):
            pred_2d, label_2d = torch.zeros((pred_3d.shape[0], 6)), torch.zeros((label_3d.shape[0], 5))
            pred_2d[:, :4] = torch.tensor(pred_3d[:, 4:8])
            pred_2d[:, 4] = torch.tensor(pred_3d[:, 15])
            pred_2d[:, 5] = torch.tensor(pred_3d[:, 0])
            label_2d[:, 0] = torch.tensor(label_3d[:, 0])
            label_2d[:, 1:] = torch.tensor(label_3d[:, 4:8])

            matches = compute_location(pred_2d, label_2d)

            if matches is not None:
                for enum, m in enumerate(matches):
                    pred_3d[enum, 11:14] = label_3d[int(m[1]), 11:14]
    from yolox.utils.show_2d3d_box import show_2d3d_box
    # conf_thres = AP50_F1_max_idx / 1000.0
    conf_thres = 0.5
    final_preds_3d = [pred[pred[:, 15] >= conf_thres] for pred in preds_3d]
    vis_save_dir = os.path.join(os.path.dirname(pred_dir), "vis_ti_quant_preds")
    print("writing 3D BBoxes")
    show_2d3d_box(final_preds_3d, labels_3d, img_paths, CLASS_NAMES, vis_save_dir, True)