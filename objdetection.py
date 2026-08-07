import cv2
from ultralytics import YOLO

def main():
    # Load the pre-trained YOLOv8 model (it will download automatically on first run)
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt') 

    # Open the default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to quit the video window.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # Run YOLOv8 detection and tracking on the frame
        # persist=True ensures tracking IDs are maintained across frames
        results = model.track(frame, persist=True, verbose=False)

        # Plot the bounding boxes and labels onto the frame
        annotated_frame = results[0].plot()

        # Display the real-time feed
        cv2.imshow("YOLOv8 Real-Time Object Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()