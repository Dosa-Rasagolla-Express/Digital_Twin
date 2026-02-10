import cv2
from ultralytics import YOLO

# 1. Load your custom model
model = YOLO('ambulance_best.pt')

# 2. Open the video file
video_path = 'videos/Ambulance.mp4'  # Replace with your video file name
cap = cv2.VideoCapture(video_path)

# Get video properties for saving the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_ambulance_detection.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (frame_width, frame_height))

print("Processing video... Press 'q' to stop.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Run YOLOv8 inference on the frame
    # We use stream=True for better memory management on video
    results = model.predict(frame, conf=0.5, stream=True)

    for r in results:
        # Plot the results on the frame
        annotated_frame = r.plot()

        # 4. Show the frame in a window
        cv2.imshow("Traffix-AI Ambulance Detection", annotated_frame)
        
        # Write the frame to the output file
        out.write(annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print("Done! Result saved as output_ambulance_detection.mp4")