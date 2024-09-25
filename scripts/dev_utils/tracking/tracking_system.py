
import sys, os, json
sys.path.append('./src/dev_utils/tracking/')
# print( sys.path )
from tracking.boxmot import OCSORT
from tracking.boxmot.tracker_zoo import create_tracker
from pathlib import Path
import numpy as np

class TrackingSystem:

    def __init__( self, tracking_method, video_loader, output_dir ):
        
        self.__video_loader    = video_loader
        self.__tracking_method = tracking_method
        self.output_dir        = output_dir 
        self.__class_names     = ['person']
        self.__tracking_id_lst = []
        self.output_json       = []
        self.setupOutputDir()
        self.__initTrackingSystem()

    def setupOutputDir( self ):
        self.output_dir = Path( self.output_dir )
        self.output_json_path:str    = f"{self.output_dir}/{self.__video_loader.movieName}.json"
        self.output_dir.mkdir( parents=True, exist_ok=True )


    def __initTrackingSystem( self ):

        self.min_hits = 30
        if self.__tracking_method == 'ocsort':
            self.__trackingsystem = OCSORT(
                min_hits=self.min_hits
            )

        print( self.output_json_path )


    def __updateId( self, previousId_list, currentId ):
        is_new_id = False
        if currentId in previousId_list:
            pass
        else:
            is_new_id = True
            previousId_list.append( currentId )
        id = previousId_list.index( currentId )
        id = f"{self.camera_prefix}_{id+1}"
        return id, is_new_id

    def __addPreviousFrames( self, track_obj , id, class_id, frameId ):
        index = 0
        # for det, conf in zip( track_obj.unconfirmed_bbox_lst, track_obj.unconfirmed_conf_lst ):
        #     frame_to_append = frameId - self.min_hits + index 
        #     if frame_to_append == frameId:
        #         break
        #     # print( frame_to_append )
        #     track_dict = {}
        #     track_dict['class'] = self.__class_names[int(class_id)]
        #     track_dict['id']    = id
        #     track_dict['bbox']  = list(np.around(np.array(det), 2))
        #     track_dict['detection_confidence'] = round(conf, 2)
        #     self.output_json[frame_to_append]['data'].append( track_dict)
        #     index += 1
        for key, val in track_obj.unconfirmed_bbox_dict.items():
            if key == frameId:
                break
            conf = track_obj.unconfirmed_conf_dict[key]
            track_dict = {}
            track_dict['class'] = self.__class_names[int(class_id)]
            track_dict['id']    = id
            track_dict['bbox']  = list(np.around(np.array(val[:]), 2))
            track_dict['detection_confidence'] = round(conf, 2)
            # print(key)
            self.output_json[key]['data'].append( track_dict)
            index += 1

    def __getTrackDict( self, track, frameId):
        x1, y1, x2, y2, track_id, conf, class_id, track_obj = track

        track_dict                         = {}
        track_dict['class']                = self.__class_names[int(class_id)]
        
        id, is_new_id                      = self.__updateId( self.__tracking_id_lst, int(track_id))
            # add unconfirmed dict
        if is_new_id is True:
            self.__addPreviousFrames( track_obj, id, class_id, frameId )
                

        track_dict['id']                   = id
        track_dict['bbox']                 = list(np.around(np.array([x1,y1,x2,y2]),2))
        track_dict['detection_confidence'] = round(conf, 2)
        

        return track_dict

    def __postprocessResults( self, tracking_result_lst, frame_dict, frameId ):

        for tracking_result in tracking_result_lst:
            # x1, y1, x2, y2, track_id, conf, class_id, _ = tracking_result
            tracking_obj_dict =  self.__getTrackDict( tracking_result, frameId )
            frame_dict['data'].append( tracking_obj_dict)

    def updateTracking( self, detection_list, video_frame, frameId ):

        #  dets = np.array([[144, 212, 578, 480, 0.82, 0],
        #             [425, 281, 576, 472, 0.56, 65]])

        # self.__tracking_system.update(, im) # --> M X (x, y, x, y, id, conf, cls, ind)

        frame_dict = {}
        frame_dict['frame'] = frameId
        frame_dict['data']  = []
        det_input_array = np.array([ detection.tracking_input for detection in detection_list ])
        if len(det_input_array) == 0:
            det_input_array = np.empty((0, 6))
        # print( f"Detresult: {det_input_array}" )
        tracking_result = self.__trackingsystem.update( det_input_array, video_frame )
        # print( f"Trackresutl: {list(tracking_result)}" )
        self.__postprocessResults( list(tracking_result), frame_dict, frameId )
        self.output_json.append( frame_dict )


    def releaseJSON( self ):
        self.track_json               = []
        os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)
        with open(f"{self.output_json_path}", 'w') as jsonFile:
            json.dump( self.output_json, jsonFile, indent=4, ensure_ascii=False )