import copy

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = ['pedestrian', 'cyclist', 'car', 'big_vehicle']

# 加载ground truth和预测结果
gt_file = 'datasets/Rope2D291/annotations_2D/instances_val.json'
# gt_file = 'datasets/Rope3D/annotations_2D/instances_val.json'
pred_file = 'datasets/Rope2D291/predictions_2D/instances_val.json'
# pred_file = 'YOLOX_outputs/yolox_s_ti_lite_rope2d/predictions.json'
images_dir = "datasets/Rope2D291/images/val"
# pred_file="/media/junzhi/8e78e258-6a68-4733-8ec2-b837743b11e6/workspace/github/YOLOv6-MONO2.5D/runs/val/exp26/predictions.json"

coco_gt = COCO(gt_file)
coco_pred = coco_gt.loadRes(pred_file)

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

# cocoEval.summarize()

