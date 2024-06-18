import cv2
import cvzone
from ultralytics import YOLO

# Load the YOLOv9 model
model = YOLO('yolov9c.pt')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Set the frame width and height (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Constants for distance calculation (example values, adjust based on your setup)
W_dict = {
    "person": 40,  # Average shoulder width of a person in cm
    "phone": 7.1,  # Average width of a phone in cm
    "laptop": 30,  # Approximate width of a laptop in cm
    # Add other objects with their real-world dimensions here
}

# Focal length, to be calibrated
f = 840

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference on the frame
    results = model(frame)

    # Extract the annotated frame from the results
    annotated_frame = results[0].plot()  # Use plot() method to get the annotated frame

    # Iterate over detections
    for result in results[0].boxes.data.tolist():
        # Unpack the result
        x1, y1, x2, y2, confidence, class_id = result

        # Calculate width of the bounding box in pixels
        w = x2 - x1
        
        # Get the label of the detected object
        label = model.names[int(class_id)]
        
        # Check if the detected object is in our W_dict
        if label in W_dict:
            W = W_dict[label]
            
            # Calculate the distance from the camera to the object
            d = (W * f) / w
            
            # Display the distance on the frame
            cvzone.putTextRect(annotated_frame, f'{label} Depth: {int(d)}cm',
                               (int(x1), int(y1) - 10), scale=2)
            
            # Draw the bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLO Real-Time Detection', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
