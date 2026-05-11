import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# 1. Configuration & Constants
# Use the homography matrix calculated in the notebook environment
HOMOGRAPHY_MATRIX = np.array([
    [-0.4, -0.27, 261.0],
    [2.067e-16, -1.84, 552.0],
    [1.2691e-17, -0.054, 1.0]
])

SOURCE_VIDEO = 'data/raw/padel_exemplo.mp4'
TARGET_VIDEO = 'output/production_tracking_output.mp4'
MODEL_PATH = 'models/yolov8n.pt'
TARGET_CLASSES = [0]  # Person class

# 2. Modular Logic Functions
def get_metric_coordinates(pixel_point, matrix):
    """Converts pixel (x, y) to real-world (x, y) in meters."""
    point = np.array([[pixel_point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, matrix)
    return transformed[0][0]

def process_frame_tracking(model, frame, target_classes):
    """Runs YOLOv8 tracking on a frame and returns supervision detections."""
    results = model.track(
        source=frame,
        persist=True,
        classes=target_classes,
        tracker='botsort.yaml',
        conf=0.5,
        verbose=False
    )[0]
    return sv.Detections.from_ultralytics(results)

def detect_padel_events(ball_pos, players_data, frame_idx, ball_history, fps=30):
    """Detects Serves and Smashes based on proximity and velocity heuristics."""
    events = []
    velocity = 0
    if ball_history:
        dist = np.linalg.norm(np.array(ball_pos) - np.array(ball_history[-1]))
        velocity = dist * fps

    for pid, p_pos in players_data.items():
        dist_to_ball = np.linalg.norm(np.array(ball_pos) - np.array(p_pos))
        if dist_to_ball < 1.5 and velocity > 15.0 and len(ball_history) < 15:
            events.append({"frame": frame_idx, "type": "Serve", "player_id": pid})
        if dist_to_ball < 2.0 and velocity > 25.0 and p_pos[1] > 10:
            events.append({"frame": frame_idx, "type": "Smash", "player_id": pid})
    return events

# 3. Main Execution Loop
def main():
    model = YOLO(MODEL_PATH)
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    print(f"Initializing production tracking for {SOURCE_VIDEO}...")
    
    with sv.VideoSink(target_path=TARGET_VIDEO, video_info=video_info) as sink:
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(SOURCE_VIDEO)):
            # A. Object Tracking
            detections = process_frame_tracking(model, frame, TARGET_CLASSES)
            
            # B. Metric Conversion and Annotation
            labels = []
            if detections.tracker_id is not None:
                for i in range(len(detections)):
                    tracker_id = detections.tracker_id[i]
                    x1, y1, x2, y2 = detections.xyxy[i]
                    px_pos = ((x1 + x2) / 2, y2)
                    
                    # Apply Homography
                    m_pos = get_metric_coordinates(px_pos, HOMOGRAPHY_MATRIX)
                    labels.append(f"ID:{tracker_id} | {m_pos[0]:.1f}m, {m_pos[1]:.1f}m")
            
            # C. Visual Output Construction
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            sink.write_frame(frame=annotated_frame)
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")

    print(f"Processing complete. Output saved to: {TARGET_VIDEO}")

if __name__ == '__main__':
    main()
