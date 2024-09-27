import cv2
import argparse
import os
from dev_utils.tracker import Tracker
import json
from tqdm import tqdm  

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX Video Inference with options to show and save results.")
    parser.add_argument("--save", action="store_true", help="whether to save the inference result of image/video")
    parser.add_argument('--show', action="store_true", help="Flag to display the video with detection.")
    parser.add_argument('--json', action="store_true", help="Path to save the detection results in JSON format.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing videos to process.")
    return parser.parse_args()

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_data_by_frame(data, frame_number):
    for frame_data in data:
        if frame_data['frame'] == frame_number:
            return frame_data['data']
    return None

def process_video(video_path, args):
    args = parse_args()
    tracker = Tracker(filter_class=["person", "bicycle", "car", "motorcycle", "bus", "truck"]) 
    
    video_basename = os.path.basename(video_path)
    detection_json_path  = f'outputs/detection_json/{video_basename}.json'
    data = read_json(detection_json_path)
  
    tracking_results = [] 
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

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

    
    for i in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, img = cap.read() 
        if not ret or img is None:
            break
        frame_data = get_data_by_frame(data, i)  
      
        img_visual, outputs = tracker.update(img,i,frame_data)  

        if args.json and len(outputs)>0:
            frame_data = []
            for output in outputs:
                frame_data.append({
                    'track_id': int(output[4]),  # Convert np.int32 to int
                    'class_id': int(output[5]),  # Convert np.int32 to int
                    'boxes': output[:4].tolist() 
             
                })
            tracking_results.append({
                'frame': i,
                'data': frame_data
            })
            # print(tracking_results)
        
        
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

    if args.json:
        output_dir = "outputs/tracking_json"
        json_output_path = os.path.join(output_dir, video_basename + ".json")

        os.makedirs(output_dir, exist_ok=True)

        with open(json_output_path, "w") as json_file:
            json.dump(tracking_results, json_file, indent=4)

if __name__ == '__main__':
    args = parse_args()
    
    for video_file in os.listdir(args.input_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):  # Add other video formats if needed
            video_path = os.path.join(args.input_dir, video_file)
            print(f"Processing video: {video_path}")
            process_video(video_path, args)
    print('all finished')
