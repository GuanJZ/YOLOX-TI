#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tqdm import tqdm
import numpy as np
import os

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh,
    box_iou,
)


class RopeEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        onnx_file=None,
        onnx_nms_file=None,
        decoder=None,
        test_size=None,
        file_name=None,
        training=True,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        preds_3d, labels_3d, img_paths = [], [], []

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        if onnx_file is not None:
            import onnxruntime
            logger.info("Init ONNX Model ...")
            model = onnxruntime.InferenceSession(onnx_file, None)
            input_name = model.get_inputs()[0].name

        if onnx_nms_file is not None:
            import onnxruntime
            logger.info("Init ONNX Model with postprocess ...")
            model = onnxruntime.InferenceSession(onnx_nms_file, None)
            input_name = model.get_inputs()[0].name

        for cur_iter, (imgs, targets, info_imgs, ids, paths) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                if onnx_file is not None or onnx_nms_file is not None:
                    outputs = torch.tensor(model.run([], {input_name: imgs.cpu().numpy()})[0])
                else:
                    outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                if onnx_nms_file is None:
                    outputs = postprocess(
                        outputs, self.num_classes, self.confthre, self.nmsthre
                    )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids, onnx_nms_file))

            import copy
            eval_outputs = copy.deepcopy([x.detach().cpu() for x in outputs])

            for si, pred in enumerate(eval_outputs):
                labels = targets[si]
                labels = labels[labels[:, 1:5].sum(dim=1) > 0]
                nl = len(labels)

                if len(pred) == 0:
                    if nl:
                        preds_3d.append(np.zeros((0, 16)))
                    continue

                # Predictions
                predn = torch.zeros((pred.shape[0], pred.shape[1]-1))
                predn[:, :4] = pred[:, :4]
                predn[:, 4] = pred[:, 4] * pred[:, 5]
                predn[:, 5:] = pred[:, 6:]

                img_h, img_w = info_imgs[0][si], info_imgs[1][si]
                scale = min(
                    self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
                )

                predn[:, :4] /= scale

                # intrinsic_path = paths[si].replace("images", "calibs").replace("jpg", "txt")
                path_tmp = paths[si].split("/")
                intrinsic_path = os.path.join(path_tmp[0], path_tmp[1], "calibs", path_tmp[2], path_tmp[3].replace("jpg", "txt"))
                with open(intrinsic_path, 'r')as f:
                    parse_file = f.read().strip().splitlines()
                    for line in parse_file:
                        if line is not None and line.split()[0] == "P2:":
                            proj_matrix = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
                theta_ray = self.calc_theta_ray(img_w, predn[:, :4], proj_matrix)

                orint, conf = predn[:, 9:13], predn[:, 13:15]
                _, conf_idxs = torch.max(conf, dim=1)
                alpha = torch.zeros(conf.shape[0])
                for enum, orient_idx in enumerate(conf_idxs):
                    cos, sin = orint[enum, orient_idx * 2], orint[enum, orient_idx * 2 + 1]
                    # 因为数据预处理将alpha的区间从[-pi, pi]平移到[0, 2*pi], 所以这里还需要再减去pi, 将 alpha 区间还原到 [-pi, pi]
                    alpha[enum] = torch.atan2(sin, cos) + (orient_idx + 0.5 - 1) * torch.pi

                Ry = theta_ray + alpha

                # predn
                # from ([xyxy, conf, cls, (H, W, L), ([cos, sin], [cos, sin]), (conf_cos, conf_sin)])
                # to
                # [xyxy, conf, cls, (H, W, L), Ry]
                predn_processed = torch.cat((predn[:, :9], Ry[:, None]), dim=1)

                if nl:
                    from yolox.utils.boxes import cxcywh2xyxy
                    tbox = cxcywh2xyxy(labels[:, 1:2], labels[:, 2:3], labels[:, 3:4], labels[:, 4:5])
                    tbox /= scale

                    # labelsn
                    # from (cls, (x, y, x, y), (H, W, L), (X, Y, Z), Ry, ([cos, sin], [cos, sin]), (conf_cos, conf_sin))
                    # to
                    # (cls, xyxy, HWL, XYZ, Ry)
                    labelsn = torch.cat((labels[:, 0:1], tbox, labels[:, 5:12]), 1)  # native-space labels

                    matches = compute_location(predn_processed, labelsn)
                    # predn3d
                    # from predn [0:x, 1:y, 2:x, 3:y, 4:conf, 5:cls, 6:H, 7:W, 8:L, 9:Ry]
                    # to
                    # (ndarray)[cls, 0, 0, 0, x1, y1, x2, y2, H, W, L, X, Y, Z, Ry, conf]
                    predn_3d = np.zeros((predn_processed.shape[0], 16))
                    predn_arr = predn_processed.numpy()
                    labelsn_arr = labelsn.numpy()
                    if matches is not None:
                        for enum, m in enumerate(matches):
                            predn_3d[enum, 0] = predn_arr[int(m[0]), 5]
                            predn_3d[enum, 4:8] = predn_arr[int(m[0]), :4]
                            predn_3d[enum, 8:11] = predn_arr[int(m[0]), 6:9]
                            predn_3d[enum, 11:14] = labelsn_arr[int(m[1]), 8:11]
                            predn_3d[enum, 14] = predn_arr[int(m[0]), 9]
                            predn_3d[enum, 15] = predn_arr[int(m[0]), 4]
                            # predn_3d[enum, 16:18] = predn_arr[int(m[0]), 10:12]
                    preds_3d.append(predn_3d)

                    # labelsn_3d
                    # from labelsn (cls, xyxy, HWL, XYZ, Ry)
                    # to
                    # (ndarray)[cls, 0, 0, 0, x1, y1, x2, y2, H, W, L, X, Y, Z, Ry]
                    labelsn_3d = np.zeros((labelsn.shape[0], 15))
                    labelsn_3d[:, 0] = labelsn[:, 0]
                    labelsn_3d[:, 4:8] = labelsn[:, 1:5]
                    labelsn_3d[:, 8:] = labelsn[:, 5:]

                    labels_3d.append(labelsn_3d)
                    img_paths.append(paths[si])

        logger.info("Save preditions 2.5D ...")
        pred_3d_save_dir = os.path.join("/".join(img_paths[0].split("/")[:2]), "preds_kitti_MONO_2.5D_fp32", "val")
        if not os.path.exists(pred_3d_save_dir):
            os.makedirs(pred_3d_save_dir)
        for pred_3d, img_path in tqdm(zip(preds_3d, img_paths)):
            pred_3d_save_path = os.path.join(pred_3d_save_dir, os.path.basename(img_path).replace("jpg", "txt"))
            np.savetxt(pred_3d_save_path, pred_3d, delimiter=" ")

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])

        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        if not training:
            *_, summary = eval_results
            logger.info("\n" + summary)

            # show 3d bboxes
            from yolox.utils.show_2d3d_box import show_2d3d_box
            # conf_thres = AP50_F1_max_idx / 1000.0
            conf_thres = 0.5
            final_preds_3d = [pred[pred[:, 15] >= conf_thres] for pred in preds_3d]

            logger.info("writing 3D BBoxes")
            show_2d3d_box(final_preds_3d, labels_3d, img_paths, ['pedestrian', 'cyclist', 'car', 'big_vehicle'], file_name,
                          True)
        else:
            return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, onnx_nms_file):
        data_list = []
        if onnx_nms_file is not None:
            outputs = [outputs]
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            if onnx_nms_file is None:
                cls = output[:, 6]
                scores = output[:, 4] * output[:, 5]
            else:
                cls = output[:, 5]
                scores = output[:, 4]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        import copy
        json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()

            CLASS_NAMES = ['pedestrian', 'cyclist', 'car', 'big_vehicle']
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

            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

    def calc_theta_ray(self, width, box_2d, proj_matrix):
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0, 0]))
        center = (box_2d[:, 2] + box_2d[:, 0]) / 2
        dx = center - (width / 2)

        if dx.shape[0] == 1:
            mult = -np.ones(dx.shape) if dx[0] < 0 else -np.ones(dx.shape)
        else:
            mult = np.ones(dx.shape)
            mult[dx < 0] = -1
        dx = np.abs(dx)
        angle = np.arctan((2 * dx * np.tan(fovx / 2)) / width)
        angle = angle * mult

        return angle

def compute_location(detections, labels):
    """
    iou>=0.5的情况下,detetcions 匹配到labels,从而得到 detections的location
    :param detections:
    :param labels:
    :return:
    """
    iou = box_iou(detections[:, :4], labels[:, 1:5])
    x = torch.where(iou >= 1e-6)
    matches = None
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            # matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return matches

