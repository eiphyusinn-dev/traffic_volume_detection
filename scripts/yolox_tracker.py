import cv2
import argparse
import os
from dev_utils.tracker import Tracker
import json
from tqdm import tqdm  # Import tqdm for the progress bar

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX Video Inference with options to show and save results.")
    parser.add_argument("--save", action="store_true", help="whether to save the inference result of image/video")
    parser.add_argument('--show', action="store_true", help="Flag to display the video with detection.")
    parser.add_argument('--json', action="store_true", help="Path to save the detection results in JSON format.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    tracker = Tracker(filter_class=["person", "bicycle", "car", "motorcycle", "bus", "truck"]) 
    
    video_path = 'src_videos/YamateRoad_DayTime_01_01_00.mp4'
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames

    if args.save:
        video_basename = os.path.basename(video_path)
        output_dir = 'outputs/tracking_videos'
        output_path = os.path.join(output_dir, video_basename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Use tqdm for the progress bar
    for _ in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, img = cap.read() 
        if not ret or img is None:
            break
        
        img_visual, bbox = tracker.update(img)  # Feed one frame and get result
        
        if args.save:
            out_video.write(img_visual)
        if args.show:
            cv2.imshow('Detection', img_visual)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    if args.save:
        out_video.release()  # Release the video writer if saving

    cap.release()
    cv2.destroyAllWindows()
