import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["QT_LOGGING_RULES"] = "*=false"

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pyrealsense2 as rs
import time
from PIL import Image
from nanoowl.owl_predictor import OwlPredictor
import threading

text_prompts = []
stop_signal = "q" 
keep_running = True
threshold = 0.1
 

def writing_prompt_thread():
    global keep_running
    while True:
        print("""
--------------------------------------------------
Enter the object you want the camera to find.

Available Commands:
  [clear] : Empty the current prompt
  [show]  : Display the current prompt
  [q]     : Shut down the camera and exit
--------------------------------------------------
>""", end=" ")
        x = input().strip()
        if x == "clear":
            text_prompts.clear()
            print("The prompt has been cleaned")
        elif x == "":
            print("Invalid prompt")
            continue
        elif x == stop_signal:
            print("shutting down the process")
            keep_running = False
        elif x == "show":
            print(f"The current prompt is: {', '.join(text_prompts)}")
        else:
            text_prompts.append(x)
            print(f"The prompt has been saved")

def compute_average_distance(depth_frame, center_x, center_y):
    distances = []
    for x in range(-10, 11, 2):
        dx = center_x + x
        if dx >= 0 and dx < depth_frame.width:
            for y in range(-10, 11, 2):
                dy = center_y + y
                if dy >=0 and dy < depth_frame.height:
                    distance = depth_frame.get_distance(dx, dy)
                    if distance > 0:
                        distances.append(distance)
    if len(distances) > 0:
        avg = sum(distances) / len(distances)
        final_distance = avg
    else:
        final_distance = 0
    return final_distance



# 1. Initialize NanoOWL Predictor
print("--- Loading NanoOWL Engine ---")
try:
    predictor = OwlPredictor(
        "google/owlvit-base-patch32",
        image_encoder_engine="owl_image_encoder_patch32.engine"
    )
    print("AI Engine loaded successfully!")
except Exception as e:
    print(f"Error loading engine: {e}")
    exit(1)

# 2. Initialize Intel RealSense D415
print("--- Initializing RealSense Camera ---")
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.depth) 

# Start streaming
try:
    print("Starting pipeline...")
    rs.log_to_console(rs.log_severity.error)
    pipeline.start(config)
    
    print("Camera started! Waiting 3 seconds for hardware to stabilize...")
    time.sleep(3) 
    
except Exception as e:
    print(f"Could not start the camera: {e}")
    exit(1)

try:
    prompt_thread = threading.Thread(target=writing_prompt_thread, daemon=True)
    prompt_thread.start()
    
    last_warning_time = 0

    align_to_color = rs.align(rs.stream.color)

    while keep_running:
        frames = pipeline.wait_for_frames(10000)
        aligned_frames = align_to_color.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert RealSense frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        color_image_rgb = np.asanyarray(color_frame.get_data())
        
        # Convert to BGR for OpenCV display
        color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)

        # NanoOWL expects a PIL Image in RGB format
        pil_image = Image.fromarray(color_image_rgb)

        if not text_prompts:
            cv2.imshow('NanoOWL + RealSense (Emergency Mode)', color_image_bgr)
            cv2.waitKey(1)
            continue

        # 3. Perform Object Detection
        if keep_running == False:
            break 
        output = predictor.predict(image=pil_image, text=text_prompts, text_encodings=None, threshold=threshold)
        if len(output.boxes) == 0: 
            now = time.time()
            if now - last_warning_time >= 2:
                last_warning_time = now
                print(f"No object in prompt: {', '.join(text_prompts)} has been detected.")
        else:
            # 4. Draw Bounding Boxes
            for i, box in enumerate(output.boxes):
                x0, y0, x1, y1 = map(int, box.tolist())
                label_idx = int(output.labels[i])
                score = float(output.scores[i])

                center_x = int(( x0 + x1 ) / 2)
                center_y = int(( y0 + y1 ) / 2)
                distance = compute_average_distance(depth_frame, center_x, center_y)

                combined_text = f"{text_prompts[label_idx]} {score:.2f} | Avg_Dist: {distance:.2f}m"

                # Draw the box and the single combined string
                cv2.rectangle(color_image_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(color_image_bgr, combined_text, (x0, y0 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
        cv2.imshow('NanoOWL + RealSense (Emergency Mode)', color_image_bgr)
        cv2.waitKey(1)

except Exception as e:
    print(f"Runtime Error: {e}")

finally:
    print("Shutting down...")
    pipeline.stop()
    cv2.destroyAllWindows()