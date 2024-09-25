import numpy as np

class TrackingInput:

    def __init__( self, detection_results ):
        self.bbox_list   = []
        self.scores_list = []
        self.class_list  = []
        self.iou_list   = []
    
        for detection in detection_results:
            
            self.bbox_list.append(detection.bbox)
            self.scores_list.append(float(detection.confidence))
            self.class_list.append(detection.class_id)
            self.iou_list.append(float(detection.iou))

      
