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
    def __init__(self,path_to_image,model_name):
        """rospy.init_node("ares/mullet/detecter")
        self.pub = rospy.Publisher("ares/goal/Mullet",Mullet,1)
        self.sub = rospy.Subscriber("rgb/image_rect_color",Image,self.callback)"""

        self.path_to_image = path_to_image
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = "cuda"
        print("Using Device:",self.device)

    
    def load_model(self, model_name):
        model = torch.hub.load("ultralytics/yolov5","custom", path=model_name, force_reload=True)
        return model


    def DetectMullet(self,frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        print(labels,cord)

    def __call__(self):
        self.DetectMullet(self.path_to_image)
        


if __name__ == "__main__":
    path_to_image = "hammer_640x640.jpg"  # Replace with the actual path
    model_name = "best(3).pt"  # Replace with the actual path

    mullet_detector = MulletDetection(path_to_image, model_name)
    mullet_detector()