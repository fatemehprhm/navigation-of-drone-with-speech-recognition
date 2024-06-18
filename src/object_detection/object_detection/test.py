import rclpy
from rclpy.node import Node
import math

from yolov8_msgs.msg import Yolov8Inference

class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.subscriber = self.create_subscription(Yolov8Inference, "/Yolov8_Inference", self.yolo_callback, 10)
    
    def yolo_callback(self, msg):
        for r in msg.yolov8_inference:
            if r != None:
                if r.distance == math.inf:
                    print('inf')
                else:
                    print(r.distance)

if __name__ == '__main__':
    rclpy.init(args=None)
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()