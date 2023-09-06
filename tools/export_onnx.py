#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger
import cv2

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module, PostprocessExport
from yolox.data.data_augment import preproc as preprocess
from yolox.utils.proto import tidl_meta_arch_yolox_pb2
from google.protobuf import text_format
from yolox.utils.proto.pytorch2proto import prepare_model_for_layer_outputs, retrieve_onnx_names


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("--export-det", action='store_true', help='export the nms part in ONNX model')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser

def export_prototxt(model, img, onnx_model_name):
    # model: pytorch model without nms part
    # img: numpy data
    # onnx_model_name

    # Prototxt export for a given ONNX model

    anchor_grid = model.head.strides
    num_heads = len(model.head.strides)
    num_keypoint = None
    keypoint_confidence = None
    keep_top_k = 200
    matched_names = retrieve_onnx_names(img, model, onnx_model_name)
    prototxt_name = onnx_model_name.replace('onnx', 'prototxt')

    background_label_id = -1
    num_classes = model.head.num_classes
    assert len(matched_names) == num_heads; "There must be a matched name for each head"
    proto_names = [f'{matched_names[i]}' for i in range(num_heads)]
    yolo_params = []
    for head_id in range(num_heads):
        yolo_param = tidl_meta_arch_yolox_pb2.TIDLYoloParams(input=proto_names[head_id],
                                                        anchor_width=[anchor_grid[head_id]],
                                                        anchor_height=[anchor_grid[head_id]])
        yolo_params.append(yolo_param)
    nms_param = tidl_meta_arch_yolox_pb2.TIDLNmsParam(nms_threshold=0.65, top_k=500)
    camera_intrinsic_params = None
    name = 'yolox'
    sub_code_type = None
    detection_output_param = tidl_meta_arch_yolox_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param, camera_intrinsic_params=camera_intrinsic_params,
                                            code_type=tidl_meta_arch_yolox_pb2.CODE_TYPE_YOLO_X, keep_top_k=keep_top_k, sub_code_type=sub_code_type,
                                            confidence_threshold=0.01, num_keypoint=num_keypoint, keypoint_confidence=keypoint_confidence)

    yolox = tidl_meta_arch_yolox_pb2.TidlYoloOd(name=name, output=["detections"],
                                            in_width=img.shape[3], in_height=img.shape[2],
                                            yolo_param=yolo_params,
                                            detection_output_param=detection_output_param,
                                            )
    arch = tidl_meta_arch_yolox_pb2.TIDLMetaArch(name=name, tidl_yolo=[yolox])

    with open(prototxt_name, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    # if not args.export_det:
    #     model.head.decode_in_inference = False
    if args.export_det:
        post_process = PostprocessExport(conf_thre=0.01, nms_thre=0.65, num_classes=exp.num_classes)
        model_det = nn.Sequential(model, post_process)
        model_det.eval()
        args.output = 'detections'

    logger.info("loading checkpoint done.")

    img = cv2.imread("./assets/dog.jpg")
    img, ratio = preprocess(img, exp.test_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = img[None, ...]
    img = img.astype('float32')
    img = torch.from_numpy(img)
    # dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])

    if args.export_det:
        output = model_det(img)
        torch.onnx.export(
            model_det,
            img,
            args.output_name,
            input_names=[args.input],
            output_names=[args.output],
            dynamic_axes={args.input: {0: 'batch'},
                          args.output: {0: 'batch'}} if args.dynamic else None,
            opset_version=args.opset,
        )
        logger.info("generated onnx model with nms part named {}".format(args.output_name))
    else:
        torch.onnx.export(
            model,
            img,
            args.output_name,
            input_names=[args.input],
            output_names=[args.output],
            dynamic_axes={args.input: {0: 'batch'},
                          args.output: {0: 'batch'}} if args.dynamic else None,
            opset_version=args.opset,
        )
        logger.info("generated onnx model without nms part named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        onnx.checker.check_model(onnx_model)

        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))

    if args.export_det:
        export_prototxt(model, img, args.output_name)
        logger.info("generated prototxt {}".format(args.output_name.replace('onnx', 'prototxt')))


if __name__ == "__main__":
    main()
