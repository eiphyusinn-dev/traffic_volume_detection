"""
@ script  : Tracking Script
@ project : ITSP Phase-3 REID
@ author  : Min Thaw Zin
@ date    : June 2023
"""

from file.file_loader import FileLoader
from detection.detection_results import Detection_Results
from tracking.tracking_system import TrackingSystem

import os, time, tqdm, pandas, glob, json
import ijson


def convertToDetectionList( detection_list ):
    """
        Function to convert detection list of frame to Detection_Results object list
        Input:
            detection_list : normal list
        Return:
            det_object_list : list of Detection_Results
    """
    det_object_list = []
    for object_data in detection_list:
        cls = object_data['category']
        det_result = Detection_Results( object_data['bbox'], object_data['confidence'], object_data['category'], class_index=cls )
        det_object_list.append( det_result )

    return det_object_list

def runTracking(movie_loader, json_detection, tracker):
    """
    Function to initialise tracking process
    """

    start_time = time.time()
    with open(json_detection, 'r') as json_file:
        detection_results = json.load(json_file)  # Load JSON data directly

        for object_data in tqdm.tqdm(detection_results['data'], total=movie_loader.maxMovieFrame):
            movie_loader.readFrame()
            
            # Create detection list using the new format
            det_list = convertToDetectionList(detection_results['data'])
            print(f'det_list > {det_list}')
            # tracker.updateTracking(det_list, movie_loader.currentFrame, movie_loader.frameId)

    fps = detection_results['frame'] / (time.time() - start_time)  # Calculate FPS
    print(f'FPS : {fps}')
    tracker.releaseJSON()


def main():
    
    # Directly specify your single video path
    input_movie = f"src_videos/YamateRoad_DayTime_01_01_00.mp4"
    json_path = f"output/detection_json/instance.json"

    file_loader = FileLoader(input_movie)
    movie_loader = file_loader.movieLoader
    tracker = TrackingSystem("ocsort", movie_loader, json_path)

    runTracking(movie_loader, json_path, tracker)

if __name__ == "__main__":
    main()
    os._exit(0)
