#!/usr/bin/env python3

from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference

bridge = CvBridge()

# Define the intrinsic parameters of the camera
fx = 558.8916045265271
fy = 419.1687033948953
cx = 320.0
cy = 240.0

class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.yolov8_inference = Yolov8Inference()
        self.model = YOLO('~/mavic/src/object_detection/object_detection/yolov8m.pt')
        self.depth_subscriber = self.create_subscription(Image,'/Mavic_2_PRO/camera_depth',self.depth_callback,10)
        self.rgb_subscriber = self.create_subscription(Image,'/Mavic_2_PRO/camera_rgb',self.camera_callback,10)
        self.camera_info_subscriber = self.create_subscription(CameraInfo, '/Mavic_2_PRO/camera_depth/camera_info', self.camera_info_callback, 10)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.intrinsics = None
        self.rgb_image = None
        self.depth_image = None
    
    def depth_callback(self, msg):
        self.depth_image = bridge.imgmsg_to_cv2(msg)
        #print(self.depth_image.shape[1])
        
    def camera_info_callback(self, msg):
        self.intrinsics = np.array(msg.k).reshape(3, 3)

    def camera_callback(self, msg):
        self.rgb_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.rgb_image is None or self.depth_image is None or self.intrinsics is None:
            return
        self.yolov8_inference.header.frame_id = "inference"
        self.yolov8_inference.header.stamp = camera_subscriber.get_clock().now().to_msg()
        results = self.model(self.rgb_image, verbose=False)
        annotated_frame = results[0].plot()
        img_msg = bridge.cv2_to_imgmsg(annotated_frame)  
        self.img_pub.publish(img_msg)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                print(box)
                self.inference_result = InferenceResult()
                b = box.xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
                distance = self.measure_distance(b)
                c = box.cls
                self.inference_result.class_name = self.model.names[int(c)]
                self.inference_result.left = int(b[0])
                self.inference_result.top = int(b[1])
                self.inference_result.right = int(b[2])
                self.inference_result.bottom = int(b[3])
                self.inference_result.distance = distance
                self.yolov8_inference.yolov8_inference.append(self.inference_result)
                print(f'Distance from {self.model.names[int(c)]}: {distance}')
                
        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()

    def measure_distance(self, box):
        x1, y1, x2, y2 = box
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        depth = self.depth_image[int(y), int(x)]
        x_normalized = (x - cx) / fx
        y_normalized = (y - cy) / fy
        distance = np.sqrt(x_normalized ** 2 + y_normalized ** 2 + 1) * depth
        return distance


if __name__ == '__main__':
    rclpy.init(args=None)
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()
