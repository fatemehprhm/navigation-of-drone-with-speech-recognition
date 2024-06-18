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

from sahi import AutoDetectionModel
from sahi.utils.cv import visualize_object_predictions,read_image_as_pil
from sahi.predict import get_sliced_prediction

from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference

bridge = CvBridge()
yolov8_model_path = '~/mavic/src/object_detection/object_detection/yolov8m.pt'

# Define the intrinsic parameters of the camera
fx = 558.8916045265271
fy = 419.1687033948953
cx = 320.0
cy = 240.0

class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.yolov8_inference = Yolov8Inference()
        #self.model = YOLO('~/mavic/src/yolobot_recognition/yolobot_recognition/yolov8m.pt')
        self.detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=yolov8_model_path,
        confidence_threshold=0.3,
        device='cuda:0',
        )
        self.depth_subscriber = self.create_subscription(Image,'/Mavic_2_PRO/camera_depth',self.depth_callback,10)
        self.rgb_subscriber = self.create_subscription(Image,'/Mavic_2_PRO/camera_rgb',self.camera_callback,10)
        self.camera_info_subscriber = self.create_subscription(CameraInfo, '/Mavic_2_PRO/camera_depth/camera_info', self.camera_info_callback, 10)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        #self.img_pub1 = self.create_publisher(Image, "/inference_result1", 1)
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
        results = get_sliced_prediction(
            self.rgb_image,
            self.detection_model,
            slice_height = 256,
            slice_width = 256,
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2,
            verbose = 0
        )

        #results1 = self.model(self.rgb_image, verbose=False)
        #annotated_frame = results1[0].plot()
        #img_msg1 = bridge.cv2_to_imgmsg(annotated_frame)  
        #self.img_pub1.publish(img_msg1)
        
        image = read_image_as_pil(results.image)

        object_predictions = visualize_object_predictions(
            image=np.ascontiguousarray(image),
            object_prediction_list=results.object_prediction_list,
            rect_th=None,
            text_size=None,
            text_th=None,
            color=None,
            hide_labels=False,
            hide_conf=False,      
            export_format="png",
        )
        
        image = object_predictions["image"]
        img_msg = bridge.cv2_to_imgmsg(image)  
        self.img_pub.publish(img_msg)
        results = results.to_coco_annotations()
        for r in results:
            bbox = r['bbox'] # get box coordinates in (x,y,w,h) format
            self.inference_result = InferenceResult()
            distance = self.measure_distance(bbox) 
            self.inference_result.class_name = r['category_name']
            self.inference_result.left = int(bbox[0])
            self.inference_result.top = int(bbox[1])
            self.inference_result.right = int(bbox[2]) + int(bbox[0])
            self.inference_result.bottom = int(bbox[3]) + int(bbox[1])
            self.inference_result.distance = distance
            self.yolov8_inference.yolov8_inference.append(self.inference_result)
            print(f'Distance from {self.inference_result.class_name}: {distance}')
            
        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()

    def measure_distance(self, box):
        x1, y1, x2, y2 = box
        x = (2*x1 + x2) / 2
        y = (2*y1 + y2) / 2
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
