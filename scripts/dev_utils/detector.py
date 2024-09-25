import sys
import torch
import numpy as np
import cv2

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp.build import get_exp_by_name,get_exp_by_file
from yolox.utils import postprocess
from yolox.utils.visualize import vis
SPECIFIC_CLASSES = [0, 1, 2, 3, 5, 6, 7]

class Detector():
    def __init__(self, model='yolox-x', ckpt='weights/yolox_x.pth'):
        super(Detector, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.exp = get_exp_by_name(model)
        self.test_size = (960,960)  
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])

  
    def detect(self, raw_img, visual=True, conf=0.4):
        info = {}
        img, ratio = preproc(raw_img, self.test_size)
        info['raw_img'] = raw_img
        info['img'] = img

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)[0].cpu().numpy()

        if outputs is None:
            return info

        info['boxes'] = outputs[:, 0:4] / ratio
        info['scores'] = outputs[:, 4] * outputs[:, 5]
        info['class_ids'] = outputs[:, 6]

        mask = np.isin(info['class_ids'], SPECIFIC_CLASSES) & (info['scores'] >= conf)
        
        info['boxes'] = info['boxes'][mask]
        info['scores'] = info['scores'][mask]
        info['class_ids'] = info['class_ids'][mask]
        info['box_nums'] = mask.sum()

        if visual:
            info['visual'] = vis(info['raw_img'], info['boxes'], info['scores'], info['class_ids'], conf, COCO_CLASSES)
        return info


if __name__ == '__main__':
    detector = Detector()
    video_path = 'src_videos/YamateRoad_DayTime_01_01_00.mp4'
    cap = cv2.VideoCapture(video_path)  # Replace with your video path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        out = detector.detect(frame)

        if 'visual' in out:
            cv2.imshow('Detection', out['visual'])

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

