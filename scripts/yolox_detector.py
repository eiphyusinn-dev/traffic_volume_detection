import cv2
import argparse
import os
from dev_utils.detector import Detector
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX Video Inference with options to show and save results.")
    parser.add_argument("--save", action="store_true", help="whether to save the inference result of image/video")
    parser.add_argument('--show', action="store_true", help="Flag to display the video with detection.")
    parser.add_argument('--json', action="store_true", help="Path to save the detection results in JSON format.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing videos to process.")
    return parser.parse_args()

def process_video(video_path, detector, args):
    detection_results = []  # List to store detection data for each frame
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0  # Track the frame number

    if args.save:
        video_basename = os.path.basename(video_path)
        output_dir = 'outputs/detection_videos'
        output_path = os.path.join(output_dir, video_basename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        detection_out = detector.detect(frame)

        # Save frame data if --json is provided
        if args.json and 'boxes' in detection_out:
            frame_data = []
            for class_id, box, score in zip(detection_out['class_ids'], detection_out['boxes'], detection_out['scores']):
                frame_data.append({
                    'class_id': int(class_id),
                    'boxes': [round(coord, 2) for coord in box.tolist()],
                    'scores': round(float(score), 2)
                })
            detection_results.append({
                'frame': frame_idx,
                'data': frame_data
            })

        # Visualize detection result
        if 'visual' in detection_out:
            result_frame = detection_out['visual']

            # Save result to file if requested
            if args.save:
                out_video.write(result_frame)

            # Display the result if --show is passed
            if args.show:
                cv2.imshow('Detection', result_frame)

            # Press 'q' to quit
            if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1  # Increment frame number

    # Release resources
    cap.release()
    if args.save:
        out_video.release()
    if args.show:
        cv2.destroyAllWindows()

    # Save detection data to JSON file if --json is provided
    if args.json:
        output_dir = "outputs/detection_json"
        json_output_path = os.path.join(output_dir, os.path.basename(video_path) + ".json")

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save results to JSON
        with open(json_output_path, "w") as json_file:
            json.dump(detection_results, json_file, indent=4)

if __name__ == '__main__':
    args = parse_args()
    
    detector = Detector()

    # Process all videos in the specified input directory
    for video_file in os.listdir(args.input_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):  # Add other video formats if needed
            video_path = os.path.join(args.input_dir, video_file)
            print(f"Processing video: {video_path}")
            process_video(video_path, detector, args)
    print('all finished')
