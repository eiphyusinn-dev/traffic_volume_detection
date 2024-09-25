import cv2

video_path = "src_videos/YamateRoad_DayTime_01_01_00.mp4"
save_dir = "datasets/images/"
cap = cv2.VideoCapture(video_path)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_name = f"frame_{frame_id:06d}.png"
    cv2.imwrite(save_dir + frame_name, frame)
    frame_id += 1

cap.release()
