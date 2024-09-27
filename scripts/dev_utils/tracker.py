import sys
sys.path.insert(0, './')
from yolox.data.datasets.coco_classes import COCO_CLASSES
from dev_utils.detector import Detector
from dev_utils.deep_sort.utils.parser import get_config
from dev_utils.deep_sort.deep_sort import DeepSort
import torch
import cv2
from dev_utils.utils.visualize import vis_track


class_names = COCO_CLASSES

class Tracker():
    def __init__(self, filter_class=None, model='yolox-x', ckpt='weights/yolox_x.pth',):
        self.detector = Detector(model, ckpt)
        cfg = get_config()
        # cfg.merge_from_file("scripts/dev_utils/deep_sort/configs/deep_sort.yaml")
        
        cfg.REID_CKPT = "scripts/dev_utils/deep_sort/deep_sort/deep/checkpoint/ckpt.t7"
        cfg.MAX_DIST =  0.2
        cfg.MIN_CONFIDENCE = 0.3
        cfg.NMS_MAX_OVERLAP = 0.5
        cfg.MAX_IOU_DISTANCE = 0.7
        cfg.MAX_AGE = 70
        cfg.N_INIT = 3
        cfg.NN_BUDGET = 100
            

        self.deepsort = DeepSort(cfg.REID_CKPT,
                            max_dist=cfg.MAX_DIST, min_confidence=cfg.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.NMS_MAX_OVERLAP, max_iou_distance=cfg.MAX_IOU_DISTANCE,
                            max_age=cfg.MAX_AGE, n_init=cfg.N_INIT, nn_budget=cfg.NN_BUDGET,
                            use_cuda=True)
        self.filter_class = filter_class

    
   
    # def update(self, image,i,frame_data):
    #     info = self.detector.detect(image, visual=False)
        
    #     outputs = []         
    #     if info['box_nums'] > 0:
    #         bbox_xywh = []
    #         scores = []
    #         class_ids = []  # Store class IDs

    #         for (x1, y1, x2, y2), class_id, score in zip(info['boxes'], info['class_ids'], info['scores']):
    #             if self.filter_class and class_names[int(class_id)] not in self.filter_class:
    #                 continue
                
    #             bbox_xywh.append([int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1])
    #             scores.append(score)
    #             class_ids.append(class_id)  # Append class ID

    #         bbox_xywh = torch.Tensor(bbox_xywh)
           
    #         outputs = self.deepsort.update(bbox_xywh, scores, class_ids, image) 
    #         image = vis_track(image, outputs,class_names)
     
    #     return image, outputs
    
    
    def update(self, image,i,frame_data):
        info = frame_data
        outputs = []
         
        if len(frame_data) > 0:
            bbox_xywh = []
            scores = []
            class_ids = []  # Store class IDs

            for item in info:
                class_id = item['class_id']
                box = item['boxes']
                score = item['scores']
                
                x1, y1, x2, y2 = box

                if self.filter_class and class_names[int(class_id)] not in self.filter_class:
                    continue
                
                bbox_xywh.append([int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1])
                scores.append(score)
                class_ids.append(int(class_id))

            
            bbox_xywh = torch.Tensor(bbox_xywh)
           
            outputs = self.deepsort.update(bbox_xywh, scores, class_ids, image) 

            image = vis_track(image, outputs,class_names)
        return image, outputs