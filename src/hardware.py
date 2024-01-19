#!/usr/bin/env python3
#import rospy
import torch
import pyzed.sl as sl
from typing import Any
import numpy as np
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

def detector(model,frame,device):
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

def main():


    print("Initializing Camera...")
    zed = init_zed()

    cap = cv2.VideoCapture(0)
    if cap.isOpened() == 0:
        exit(-1)
    
    model = torch.hub.load(("ultralytics/yolov5"), "yolov5m")
    runtime_params = sl.RuntimeParameters()
    image_left, depth_left, point_cloud = sl.Mat(), sl.Mat(), sl.Mat()
    obj_params = sl.ObjectDetectionParameters()
    obj_params.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_runtime_params = sl.ObjectDetectionRuntimeParameters()

    print("Enabling object detection module.")
    zed.enable_object_detection(obj_params)
    objects = sl.Objects()
    device = 'cuda' if torch.cuda.is_available() else "! cpu"
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_left, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            labels, cords = detector(model, frame, device)

            retval, frame = cap.read()
            left_right_image = np.split(frame, 2, axis=1)

            label_list = labels.tolist()
            cord_list = cords.tolist()
            objects_in = []
            # The "detections" variable contains your custom 2D detections
            for label,cord in zip(label_list,cord_list):
                # Create object data and fill the detections into the correct SDK format
                """tmp = sl.CustomBoxObjectData()
                tmp.unique_object_id = sl.generate_unique_id()
                tmp.label = int(label)"""
                
                # Assuming cord is in the format [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax, conf = cord
                #tmp.probability = conf
                # Convert to [top-left, top-right, bottom-right, bottom-left]
                top_left = [xmin, ymin]
                top_right = [xmax, ymin]
                bottom_right = [xmax, ymax]
                bottom_left = [xmin, ymax]

                # Arrange as [top-left, top-right, bottom-right, bottom-left]
                #tmp.bounding_box_2d = np.array([top_left, top_right, bottom_right, bottom_left])
                top_left = (int(xmin), int(ymin))
                bottom_right = (int(xmax), int(ymax))
                cv2.rectangle(left_right_image[0], top_left, bottom_right, (0,255,0), 2)
                #tmp.is_grounded = True # objects are moving on the floor plane and tracked in 2D only
                #objects_in.append(tmp)

            
            cv2.imshow("frame", left_right_image[0])
            
            print(objects_in)
            #zed.ingest_custom_box_objects(objects_in) # Ingest the objects into the SDK
            # Structure containing all the detected objects
            print("sa ben geldim1")
            zed.retrieve_objects(objects, obj_runtime_params) # Retrieve the 3D tracked objects
            print(objects.object_list)
            print("sa ben geldim2")
            for object in objects.object_list:
                print(f"{object.id} {object.position}")
        else:
            print("error with grabbing")
            
if __name__ == "__main__":
    main()