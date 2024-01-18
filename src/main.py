#!/usr/bin/env python3
import rospy
from typing import Any
import numpy as np
import torch
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ozurover_messages.msg import Marker
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String



class MulletDetection:
    # Init function
    def __init__(self,capture_index,model_name):

        # İnitializing node, publisher and subcribe topics
        rospy.init_node("ares_mullet_detecter")
        self.pub = rospy.Publisher("ares/goal/Mullet",String,queue_size=1)
        self.sub = rospy.Subscriber("/zed2/left/image_rect_color",Image,self.callback)

        # defining necessary variables
        self.model_name = model_name                  # Path to model that is going to be used
        self.capture_index = capture_index            # It's required for visualation not for Jetson
        self.model = self.load_model(self.model_name) # Loading model
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else "! cpu" 
        print("Using Device:",self.device)

    # Loads the model using torch.hub.load()
    def load_model(self,model_name):

        print(f"Loading model from: {model_name}")
        model = torch.hub.load("ultralytics/yolov5","custom", path=model_name, force_reload=True)
        print(f"{model_name} loaded successfully")
        return model

    # This function takes frames and detect objects, and returns them
    def detection(self, frame):

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        print(labels,cord)
        return labels, cord


    def class_to_label(self,x):

        return self.classes[int(x)]

    # Callback Function
    def callback(self, image_data):
        
        try:
            # Creates a bridge for forwarding images
            bridge = CvBridge() 
            frame = bridge.imgmsg_to_cv2(image_data, desired_encoding='bgr8')  # It converts image to opencv format
            
            # Change from cv2.waitKey(0) to cv2.waitKey(1)
            frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
            
            # Object detection
            labels, cord = self.detection(frame)

            n = cord.shape[0] # Number of detected objects
            x_shape, y_shape = frame.shape[1], frame.shape[0] # Width and height of the image

            # Iterating for every detected object
            for i in range(n):
                row = cord[i] # Sample cord[i] = [0.77514, 0.12554, 0.78478, 0.13924, 0.32837] 
                
                # Checks for confidince score for detected object
                if row[4] >= 0.2: 
                    # Left top corner and right bottom corner cordinates
                    x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)

                    # Draws the bounding box and displays the frame
                    cv2.rectangle(frame,(x1,y1), (x2, y2), (0,255,0), 2)
                    cv2.imshow("Data", frame)
                    cv2.waitKey(1) 


                    # Specified Message type will be written here.


                    # Creates message that will be published and publishes it
                    message = f"{self.class_to_label(labels[i])},{labels[i]},{x1},{y1},{x2},{y2}"
                    self.pub.publish(String(f"zaa {message}"))

        except Exception as e:
            rospy.logerr(f"Error processing image: {repr(e)}")

        
        
if __name__ == "__main__":
    mullet_detector = MulletDetection(capture_index=0, model_name="model/yolov5/yolov5l.pt")
    rospy.spin()