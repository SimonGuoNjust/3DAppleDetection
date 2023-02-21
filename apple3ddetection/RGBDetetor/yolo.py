import warnings
import sys

import torch
from .base import RGBDetector
from ..thirdparty.yolov5.models.experimental import attempt_load
from ..thirdparty.yolov5.utils.general import  check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh
from ..thirdparty.yolov5.utils.plots import plot_one_box
from ..thirdparty.yolov5.utils.torch_utils import select_device
from ..thirdparty.yolov5.utils.datasets import letterbox

import numpy as np
import copy
import cv2


class YoloDetector(RGBDetector):
    def __init__(self, detector_cfg=None):
        super(YoloDetector,self).__init__(detector_cfg)
        self.inputsz = detector_cfg.input.inputsz
        self.imgsz = detector_cfg.model.imgsz
        self.device = select_device(detector_cfg.device)
        self.half = self.device.type != 'cpu'
        self.model = self.build_model(detector_cfg.model)

    def build_model(self, cfg):
        model = attempt_load(cfg.ckg_path, map_location=self.device)
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz[0], s=stride)  # check img_size
        if self.half:
            model.half()  # to FP16
        # # Run inference
        if self.device.type != 'cpu':
            model(
                torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        img = torch.zeros(1, 3, imgsz, imgsz).to(self.device)
        #
        # # Warmup
        if self.device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=True)
        return model

    def preprocess(self, img):
        img = letterbox(img, self.imgsz, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        self.actual_imgsz = img.shape[2:]
        return img
    
    def postprocess(self, result, post_cfg=None):
        pred = non_max_suppression(result, post_cfg.conf_thres, post_cfg.iou_thres, classes=None, agnostic=False)
        result = list()
        for i, det in enumerate(pred):  # detections per image
            if len(det):  # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(self.actual_imgsz, det[:, :4], self.inputsz).round()
                if self.ifverbose:
                    print(f"Detected {len(det)} apples.",end='\r')
            for N, (*xyxy, conf, cls) in enumerate(reversed(det)):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                if post_cfg.bbox == 'xyxy':
                    bbox = torch.tensor(xyxy).view(1, 4).view(-1).cpu().tolist()
                elif post_cfg.bbox == 'xywh':
                    bbox = xywh
                else:
                    warnings.WarningMessage("Unsupported bbox format, using default xyxy.")
                    bbox = (torch.tensor(xyxy).view(1, 4).view(-1).cpu().tolist())
                result.append(bbox + [conf.cpu().tolist()])
        return result

    def show_result(self, img, result, show_cfg=None):
        show_img = copy.deepcopy(img)
        for _instance in result:
            if _instance[4] > show_cfg.show_conf:
                plot_one_box(_instance[:4],show_img, show_cfg.color,show_cfg.label)
        cv2.imshow('Detection Result', show_img)
        cv2.waitKey(10)