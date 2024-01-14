#!/usr/bin/env python3
from typing import Any
import numpy as np
import torch
import cv2
import time
"""
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ozurover_messages.msg import Mullet
"""


class MulletDetection:
    def __init__(self,capture_index,model_name):
        """rospy.init_node("ares/mullet/detecter")
        self.pub = rospy.Publisher("ares/goal/Mullet",Mullet,1)
        self.sub = rospy.Subscriber("rgb/image_rect_color",Image,self.callback)"""

        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = "cuda"
        print("Using Device:",self.device)

    
    def load_model(self, model_name):
        model = torch.hub.load("ultralytics/yolov5","custom", path=model_name, force_reload=True)
        return model


    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self,x):
        return self.classes[int(x)]
        
        
    def plot_boxes(self, results, frame):
        labels, cord = results
        print(labels)
        n = cord.shape[0]
        x_shape, y_shape = frame.shape[1],frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                box_color = (0,255,255)
                if labels[i] == 34:  # Assuming 0 corresponds to "mullet"
                    box_color = (0, 165, 255)  # Orange color for mullet
                elif labels[i] == 39:  # Assuming 1 corresponds to "bottle"
                    box_color = (255, 0, 0)  # Blue color for bottle
                else:
                    box_color = (0, 0, 255)  # Red color for other objects
                cv2.rectangle(frame,(x1,y1), (x2, y2), box_color, 2)
                label_position = (x1, y1 - 10)  # Above the box
                cv2.putText(frame, self.class_to_label(labels[i]), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        return frame


    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640,640), interpolation = cv2.INTER_AREA)
            results, cord = self.score_frame(frame)
            if len(results) > 0:
                print(f"Found {len(results)} mullets")
                print(f"Labels: {results}")
                print(f"Coordinates: {cord}")
                frame = self.plot_boxes((results, cord), frame)
            cv2.imshow("Mullet", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    mullet_detector = MulletDetection(capture_index=0, model_name="yolov5l.pt")
    mullet_detector()