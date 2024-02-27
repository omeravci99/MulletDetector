import cv2
import torch

torch.cuda.empty_cache()

# Load YOLOv5 model
device = 'cuda'  # Use CPU for inference
my_model = 'yolov5/runs/train/exp3/weights/best.pt'
model = torch.hub.load('yolov5', 'custom', path=my_model, source='local', force_reload=True)

# Load the classes (define your classes or use default COCO classes)
classes = [...]  # Define your classes here

# Use camera of the laptop and load the model
cap = cv2.VideoCapture(0)

# Start image processing thread
while True:
    ret, frame = cap.read()
    if ret:
        # Detect objects
        results = model(frame)

        # Plot boxes
        for label, cord in zip(results.names[0], results.xyxy[0][:, :-1]):
            cv2.rectangle(frame, (int(cord[0]), int(cord[1])), (int(cord[2]), int(cord[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(cord[0]), int(cord[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
