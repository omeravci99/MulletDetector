#!/usr/bin/env python3
from typing import Any
import torch
import numpy as np
import cv2
"""import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ozurover_messages.msg import Mullet"""


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


    def score_frame(self,frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxy[0][:, -1], results.xyxy[0][:, :-1]
        print(labels,cord)

    def class_to_label(self,x):
        return self.classes[int(x)]
        
    def plot_boxes(self,results,frame):
        labels, cord = results
        n = len(results)
        x_shape, y_shape = frame.shape[1],frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                bgr = (0,255,0)
                cv2.rectangle(frame,(x1,y1), (x2, y2), bgr, 2)

        return frame

    def __call__(self):
        cap = self.get_video_capture()
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame,(640,640))
            results = self.score_frame(frame)
            frame = self.plot_boxes(results,frame)
            cv2.imshow("anan",frame)
        
if __name__ == "__main__":

    mullet_detector = MulletDetection(capture_index=0, model_name="best(3).pt")
    mullet_detector()