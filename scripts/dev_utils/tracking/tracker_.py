from dev_utils.tracking.Tracker.strongSort.strong_sort import StrongSORT
from dev_utils.tracking.tracking_input import TrackingInput
import torch, os, json
from pathlib import Path
import numpy as np

class Tracker:

    @property
    def sourceVideoFPS( self ):
        return self.__videoFPS

    @property
    def sourceVideoName( self ):
        return self.__videoName

    @property
    def personTracks( self ):
        return self.__person_updated_tracks
    
    @property
    def personMaxIds( self ):
        return max( self.person_ids )

    def __init__( self, movieLoader=None, starting_id = 1, frame_skip = 1, camera_index='A', classes=['person']):
        self.movieLoader = movieLoader
        self.__videoFPS:float          = movieLoader.FPS
        self.__videoName:str           = movieLoader.movieName
        self.__strongsort_weights_file = "./src/dev_utils/tracking/Tracker/strongSort/deep/checkpoint/mobilenetv2_x1_0_market1501.pt"
        self.unique_id                 = starting_id
        self.class_names               = classes
        self.camera_prefix             = camera_index
        self.output_json               = []
        self.all_tracks                = []
        self.class_ids                 = []
        self.frame_skip                = frame_skip
        self.setupSystem()

    def __str__( self ):
        return str( self.__track_settings )

    def setupOutputDir( self, output_dir = './output/tracking_results' ):
        self.output_dir = Path( output_dir )
        self.output_json_path:str    = f"{self.output_dir}/{self.sourceVideoName}.json"
        self.output_dir.mkdir( parents=True, exist_ok=True )
        
    def setupSystem( self ):

        self.tracking_system_lst    = []
        self.tracking_system_tracks = []
        for class_ in self.class_names:
            system = StrongSORT(  self.__strongsort_weights_file, f"src/dev_utils/tracking/Tracker/strongSort/configs/person_config.yaml" )
            self.tracking_system_lst.append( system )
            self.tracking_system_tracks.append( [] )
            self.class_ids.append( [] )

    def getDetectionListByClass( self, detection_list, class_name ):
        interested_list = []
        if detection_list is not None and len( detection_list ) > 0:
            for detections in detection_list:

                if detections.class_name == class_name:
                    interested_list.append( detections )
        return interested_list

    def updateTracking( self, results, currentFrame, frameId ):
        for index, class_ in enumerate( self.class_names ):
            self.updateTrackFrame( self.getDetectionListByClass(results, class_) , currentFrame, frameId , class_, index )
        self.all_tracks = self.__getAllTracks()
        self.frame_dict = self.__addUnconfirmedData( self.all_tracks, frameId )
        self.output_json.append( self.frame_dict )
            
    def updateTrackFrame( self, detection_results, currentFrame, frameId, class_name, index ):
        tracking_system = self.tracking_system_lst[index]
        with torch.no_grad():
            if len( detection_results ) == 0:
                tracking_system.increment_ages()
                tracks = []
            else:
                tracking_input = TrackingInput( detection_results )
                tracks = tracking_system.trackFrame( tracking_input, currentFrame, frameId, class_name )

        self.tracking_system_tracks[index] = tracks
        
    def releaseJSON( self ):
        self.track_json               = []
        os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)
        with open(f"{self.output_json_path}", 'w') as jsonFile:
            json.dump( self.output_json, jsonFile, indent=4, ensure_ascii=False )


    def __getTrackIds( self, class_name ):
        class_index = self.class_names.index( class_name )
        return self.class_ids[class_index]

    def __updateId( self, previousId_list, currentId, class_name ):
        if currentId in previousId_list:
            pass
        else:
            previousId_list.append( currentId )
        id = previousId_list.index( currentId )
        class_ = class_name[0].upper()
        id = f"{self.camera_prefix}_{class_}{id+1}"
        return id

    def __getTrackDict( self, track ):
        track_dict                         = {}
        track_dict['class']                = self.class_names[track.class_id]
        previous_id_list                   = self.__getTrackIds( track_dict['class'] )
        id                                 = self.__updateId( previous_id_list, track.track_id, track_dict['class'])
        track_dict['id']                   = id
        track_dict['bbox']                 = list(np.around(np.array(track.bbox.tolist()),2))
        track_dict['detection_confidence'] = round(track.conf.item(), 2)
        track_dict['tracking_confidence']  = round(track.track_conf, 2)
        track_dict['iou']                  = round(track.iou.item(), 2)

        return track_dict

    def __getUnconfirmed( self, id, class_name, detection_confidence, tracking_confidence, unconfirmed_bbox, iou ):
        unconfirmed_dict                         = {}
        unconfirmed_dict['class']                = class_name
        unconfirmed_dict['id']                   = id
        unconfirmed_dict['bbox']                 = list(np.around(np.array(unconfirmed_bbox),2))
        unconfirmed_dict['detection_confidence'] = detection_confidence
        unconfirmed_dict['tracking_confidence']  = tracking_confidence
        unconfirmed_dict['iou']                  = iou

        return unconfirmed_dict

    def __addUnconfirmedData( self, tracks, frameId, reuse=False ):
        frame_dict = {}
        frame_dict['frame'] = frameId
        frame_dict['data']  = []
        for track in tracks:
            if track.is_confirmed():

                track_dict =  self.__getTrackDict( track )
                frame_dict['data'].append( track_dict )

                if track.hits == track._n_init + 1 and reuse==False :
                    unconfirmed_list = []
                    for detection in track.unconfirmed_list:
                        unconfirmed_list.append( detection )
                    total_unconfirmed_frames = len(unconfirmed_list)
                    for index, frame in enumerate( range( frameId - total_unconfirmed_frames , frameId )):
                        unconfirmed_dict = self.__getUnconfirmed( track_dict['id'], track_dict['class'], \
                        track_dict['detection_confidence'], track_dict['tracking_confidence'], unconfirmed_list[index], track_dict['iou'] )
                        self.output_json[frame]['data'].append( unconfirmed_dict )

        return frame_dict
    

    def __getAllTracks( self ):
        track_ = []
        for index, class_ in enumerate( self.class_names ):
            self.__appendTrack( track_, self.tracking_system_tracks[index] )
        return track_
    
    def __appendTrack( self, track_list, interested_tracks ):
        if interested_tracks != []:
            for track in interested_tracks:
                track_list.append( track )
        else:
            pass