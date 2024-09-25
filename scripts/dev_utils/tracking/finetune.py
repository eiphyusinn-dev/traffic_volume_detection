"""
@ script  : Tracking Script
@ project : ITSP Phase-3 REID
@ author  : Min Thaw Zin
@ date    : March 2023
"""

from dev_utils.file.yaml_parser import YamlParser
from dev_utils.file.file_loader import FileLoader
from dev_utils.file.logger import logger
from dev_utils.detection.detection_results import Detection_Results
from dev_utils.tracking.tracker_ import Tracker
from dev_utils.tracking.mot_generator import MOTGenerator
from dev_utils.drawing import draw_output
import cv2, os, time, tqdm, pandas

def getDetectionFrame( detection_list ):
    det_object_list = []
    for object_data in detection_list:
        class_name, frameId, objectId, bbox, confidence = object_data
        det_result = Detection_Results( bbox, confidence, class_name, class_index=0 )
        det_object_list.append( det_result )

    return det_object_list

def writeToText( track, frameId ):
    bbox   = list(map( int, track.bbox))
    width  = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]  
    conf   = round(track.conf.item(), 2)
    line_  = f"{frameId},{track.track_id},{bbox[0]},{bbox[1]},{width},{height},1,1,{conf}\n"
    return line_

def runTracking( movie_loader, detection_data, tracker, drawer ):

    start_time  = time.time()
    seq_frame   = 0
    seq_        = 0 
    total_frame = 0
    mot_gen     = MOTGenerator( movie_loader.moviePath, seq_ )
    mot_gen.generateSeqInfo( movie_loader )
    gtFile      = open( f"{mot_gen.gt_file}/gt.txt", 'w' )

    for frame in tqdm.tqdm( range( movie_loader.maxMovieFrame )):
        movie_loader.readFrame()
        object_data = detection_data.loc[detection_data['frame'] == frame].values.tolist()
        det_list    = getDetectionFrame( object_data )
        tracker.updateTracking( det_list, movie_loader.currentFrame, movie_loader.frameId )

        tracking_image = drawer.draw( movie_loader.currentFrame, tracker.personTracks )
        
        cv2.imshow('output', tracking_image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            seq_frame += 1
            for track in tracker.personTracks:
                line_ = writeToText( track, seq_frame )
                gtFile.writelines( line_ )
            
            mot_gen.generateImg( movie_loader.currentFrame, seq_frame )

        if key == ord('q'):
            break

        if key == ord('b'):
            seq_ += 1
            mot_gen     = MOTGenerator( movie_loader.moviePath, seq_ )
            mot_gen.generateSeqInfo( movie_loader )
            gtFile.close()
            seq_frame   = 0
            gtFile      = open( f"{mot_gen.gt_file}/gt.txt", 'w' )


        total_frame += 1

    fps = total_frame / ( time.time() - start_time )
    print( f'FPS : { fps }')
    tracker.releaseJSON()

def main():

    movie_config    = YamlParser( './cfg/movie.yaml' )
   
    file_loader     = FileLoader( movie_config['input_movie'], movie_config['start_time'] )
    movie_loader    = file_loader.movieLoader

    json_path       = f"./output/detection_results/{file_loader.baseName}.json"
    tracker         = Tracker( movie_loader, camera_index="D" )

    with open( json_path ) as json_file:
        detection_data = pandas.read_json( json_file )

    logger.info( f"Total movie frames: {movie_loader.maxMovieFrame}" )
    logger.info(  "Starting Finetune Process" )

    default_color  = [(0,255,0),(255,0,0),(0,0,255),(122,255,122)]
    drawer         = draw_output.TrackingDraw( default_color, output_video=False )
    runTracking( movie_loader, detection_data, tracker, drawer )

if __name__ == "__main__":
    main()
    os._exit(0)