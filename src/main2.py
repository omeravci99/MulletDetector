import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
from ozurover_messages.msg import Marker
import torch
import socket

class MarkerDetector:
    def _init_(self):
        rospy.init_node("ares_aruco_detecter")
        self.pub = rospy.Publisher("ares/goal/marker",Marker,queue_size = 1)
        self.sub = rospy.Subscriber("/zed2i/zed_node/left/image_rect_color",Image,self.callback)
        self.MARKER_SIZE = 20.0
        self.CAM_MAT = np.array([[263.95489501953125, 0, 320],
                         [0, 263.95489501953125, 180],
                         [0, 0, 1]])    
    
        self.DIST_COEF = np.array([-0.784998, 2.44096, 0.000561938, -7.78445e-05, 0.113489, -0.680278, 2.29567, 0.281928])
        self.MARKER_DICT = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        self.PARAM_MARKERS = cv.aruco.DetectorParameters_create()

        # Object detection
        self.model = self.load_model()
        self.classes = self.model.names if hasattr(self.model, 'names') else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize flag for image processing
        self.process_image = True

        # Start image processing thread
     

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5' , 'custom', path='best.pt')
        model.eval()
        return model

    def detect_objects(self,frame):
        self.model.to(self.device)
        frame = cv2.resize(frame, (640, 640))
        results = self.model(frame, size=640)
        labels, cord = results.xyxy[0][:, -1], results.xyxy[0][:, :-1]        
        return labels, cord
    
    def plot_boxes(self,frame,labels,cord):
        for i in range(len(labels)):
            label = int(labels[i])
            box = cord[i]
            cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv.putText(frame, self.classes[label], (int(box[0]), int(box[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    
    def detect_markers(self,frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.marker_corners, self.marker_IDs, reject = cv.aruco.detectMarkers(gray, self.MARKER_DICT, parameters=self.PARAM_MARKERS)
        return self.marker_corners, self.marker_IDs

    def estimate_pose(self,marker_corners):
        self.rVec, self.tVec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, self.MARKER_SIZE, self.CAM_MAT, self.DIST_COEF)
        return self.tVec
        
    def aruco_func(self,frame):
        marker_corners, marker_IDs = self.detect_markers(frame)
        if marker_corners:
            for (markerCorner, markerID) in zip(marker_corners, marker_IDs):
                # extract the marker corners (which are always returned
                # in top-left, top-right, bottom-right, and bottom-left
                # order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # Convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # Draw the bounding box of the ArUco detection
                cv.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                # Compute and draw the center (x, y)-coordinates of the ArUco marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

                # Draw the ArUco marker ID on the frame
                cv.putText(frame, str(markerID),
                            (topLeft[0], topLeft[1] - 15),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
            tVec = self.estimate_pose(marker_corners)

            total_markers = range(0, marker_IDs.size)
            for ids, i in zip(marker_IDs, total_markers):
                aruco_tag = Marker()
                aruco_tag.pose.pose.position.x = tVec[i][0][0]
                aruco_tag.pose.pose.position.y = tVec[i][0][1]
                aruco_tag.pose.pose.position.z = tVec[i][0][2]
                aruco_tag.pose.header.frame_id = "zed2i_left_camera_frame"
                aruco_tag.type = ids[0]
                self.pub.publish(aruco_tag)
        return frame

    def image_processing_thread(self):
        rate = rospy.Rate(30)  # Adjust the rate as needed
        while not rospy.is_shutdown():
            if self.process_image:
                # Perform image processing here
                # Convert ROS Image message to OpenCV image
                bridge = CvBridge()
                image_data = rospy.wait_for_message("/zed2i/zed_node/left/image_rect_color", Image)
                cv_image = bridge.imgmsg_to_cv2(image_data, desired_encoding="bgr8")
                detected_frame = self.aruco_func(cv_image)

                # Object detection
                labels, cord = self.detect_objects(detected_frame)
                frame_with_boxes = self.plot_boxes(detected_frame, labels, cord)
                cv.imshow("Detected Markers", frame_with_boxes)
                cv.waitKey(1)

                self.process_image = False  # Reset the flag after processing
            rate.sleep()

    def callback(self, image_data):
        # Set the flag to indicate that a new image is available for processing
        self.process_image = True

    def spin(self):
        rospy.spin()
    
    
if _name_ == "_main_":  

    node = MarkerDetector()
    node.spin()