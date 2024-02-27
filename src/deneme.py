import cv2
import torch



# Use camera of the laptop and load the model
cap = cv2.VideoCapture(0)

# load the model 
model = torch.hub.load('ultralytics/yolov5', 'custom','../model/yolov5/runs/train/exp3/weights/best.pt', force_reload=True)

# Load the classes

# Initialize flag for image processing

# Start image processing thread
while True:
    ret, frame = cap.read()
    if True:
        # Detect objects
        model.to('cuda')
        frame = cv2.resize(frame, (640, 640))
        results = model(frame, size=640)
        labels, cord = results.xyxy[0][:, -1], results.xyxy[0][:, :-1]

        # Plot boxes
        for i in range(len(labels)):
            label = int(labels[i])
            box = cord[i]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, classes[label], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break