#!/usr/bin/env python3
#import rospy
import pyzed.sl as sl
from typing import Any
import numpy as np
import torch
import cv2
import time
"""from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ozurover_messages.msg import Marker
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String"""


def init_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    err = zed.open(init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)
    else:
        print("Zed Camera Started!")
        return zed

def load_model(model_name):
    model = torch.hub.load("ultralytics/yolov5","custom", path=model_name, force_reload=True)
    return model

def detector(model,frame,device):
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

def main():
    model_name = "model/yolov5/yolov5l.pt"
    device = 'cuda' if torch.cuda.is_available() else "! cpu" 

    # İnitalizing zed and setting parameters
    zed = init_zed()
    runtime_params = sl.RuntimeParameters()
    image_left = sl.Mat()
    depth_left = sl.Mat()
    point_cloud = sl.Mat()

    # Setting object detection parameters
    obj_params = sl.ObjectDetectionParameters()
    obj_params.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    zed.enable_object_detection(obj_params)

    
    obj_runtime_params = sl.ObjectDetectionRuntimeParameters()

    model = load_model(model_name)





    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_left, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            image = image_left.get_data()
            labels, cords = detector(model, image, device)


            objects_in = []
            # The "detections" variable contains your custom 2D detections
            for label,cord in zip(labels,cords):
                tmp = sl.CustomBoxObjectData()
                # Fill the detections into the correct SDK format
                tmp.unique_object_id = sl.generate_unique_id()
                tmp.probability = cord[4]
                tmp.label = int(label)
                tmp.bounding_box_2d = cord[:4]
                tmp.is_grounded = True # objects are moving on the floor plane and tracked in 2D only
                objects_in.append(tmp)
            zed.ingest_custom_box_objects(objects_in)

            
            objects = sl.Objects() # Structure containing all the detected objects
            zed.retrieve_objects(objects, obj_runtime_params) # Retrieve the 3D tracked objects
            for object in objects.object_list:
                print("{} {}".format(object.id, object.position))

            
            
            




    zed.close()