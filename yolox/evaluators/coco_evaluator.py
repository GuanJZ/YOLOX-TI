#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
import os
from loguru import logger
from tqdm import tqdm
import numpy as np

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, dataloader, img_size, confthre, nmsthre, num_classes, data_dir, testdev=False, save_yolo_text=False,
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
        self.data_dir = data_dir
        self.testdev = testdev
        self.save_yolo_text = save_yolo_text

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
        model_file_dir=None,
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
        if self.save_yolo_text:
            yolo_list = []
            pred_files_name = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

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

        for cur_iter, (imgs, _, info_imgs, ids, files_name) in enumerate(
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
            if self.save_yolo_text:
                yolo_list.extend(self.convert_to_yolo_format(outputs, info_imgs, ids, files_name, onnx_nms_file))
                pred_files_name.extend(files_name)
        if self.save_yolo_text:
            files_save_dir = os.path.join(model_file_dir, "preds_yolo_MONO_2D")
            if not os.path.exists(files_save_dir):
                os.makedirs(files_save_dir)
            for yolo_res, file_name in zip(yolo_list, pred_files_name):
                file_save_path = os.path.join(files_save_dir, file_name.replace("jpg", "txt"))
                np.savetxt(file_save_path, np.around(yolo_res, 6), delimiter=" ")

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics, model_file_dir)
        synchronize()
        return eval_results

    def convert_to_yolo_format(self, outputs, info_imgs, ids, files_name, onnx_nms_file):
        data_list = []
        if onnx_nms_file is not None:
            outputs = [outputs]
        for (output, img_h, img_w, img_id) in zip(
                outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue

            output = output.cpu().numpy()

            if onnx_nms_file is None:
                cls = output[:, 6]
                scores = output[:, 4] * output[:, 5]
            else:
                cls = output[:, 5]
                scores = output[:, 4]

            predn = np.zeros((output.shape[0], 6))
            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale

            bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / self.img_size[0]
            bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / self.img_size[1]
            bboxes[:, 1] = bboxes[:, 1] / self.img_size[0] + bboxes[:, 3] / 2
            bboxes[:, 1] = bboxes[:, 0] / self.img_size[1] + bboxes[:, 2] / 2

            predn[:, 2:] = bboxes
            predn[:, 0] = cls
            predn[:, 1] = scores

            data_list.append(predn)
        return data_list

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

    def evaluate_prediction(self, data_dict, statistics, model_file_dir):
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
        if len(data_dict) > 0:
            if self.save_yolo_text:
                json.dump(data_dict, open(os.path.join(model_file_dir, "./predictions.json"), "w"))
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
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
