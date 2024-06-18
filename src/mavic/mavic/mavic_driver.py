"""ROS2 Mavic 2 Pro driver."""

import math
import rclpy
import numpy as np
from yolov8_msgs.msg import Yolov8Inference
import math
import time
from std_msgs.msg import String


K_VERTICAL_THRUST = 70   # with this thrust, the drone lifts.
K_VERTICAL_P = 2.6          # P constant of the vertical PID.
K_ROLL_P = 50.0             # P constant of the roll PID.
K_PITCH_P = 30.0            # P constant of the pitch PID.
K_YAW_P = 2.0
MAX_YAW_DISTURBANCE = 0.4
MAX_PITCH_DISTURBANCE = -1
target_precision = 0.5
K_X_VELOCITY_P = 1
K_Y_VELOCITY_P = 1
K_X_VELOCITY_I = 0.01
K_Y_VELOCITY_I = 0.01
LIFT_HEIGHT = 1
del_dict = ['go', 'to', 'toward', 'please', 'the', 'a', 'and','find', 'an', 'on', 'move','fly']
motion_dict = ['stop','up','down','forward','backward','turn','right','left']

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


class MavicDriver:
    def init(self, webots_node, properties):
        self.robot = webots_node.robot
        self.timestep = int(self.robot.getBasicTimeStep())

        # Sensors
        self.gps = self.robot.getDevice('gps')
        self.gyro = self.robot.getDevice('gyro')
        self.imu = self.robot.getDevice('inertial unit')
        self.dis_sensor_down = self.robot.getDevice('down')
        self.dis_sensor_down.enable(16)
        self.dis_sensor_for = self.robot.getDevice('front')
        self.dis_sensor_for.enable(16)
        self.dis_sensor_left = self.robot.getDevice('left')
        self.dis_sensor_left.enable(16)
        self.dis_sensor_right = self.robot.getDevice('right')
        self.dis_sensor_right.enable(16)
        self.dis_sensor_back = self.robot.getDevice('back')
        self.dis_sensor_back.enable(16)
        self.dis_sensor_up = self.robot.getDevice('up')
        self.dis_sensor_up.enable(16)
        self.target_name = []
        # Propellers
        self.propellers = [
            self.robot.getDevice('front right propeller'),
            self.robot.getDevice('front left propeller'),
            self.robot.getDevice('rear right propeller'),
            self.robot.getDevice('rear left propeller')
        ]
        self.flag = False
        self.flag2 = False
        self.flag3 = False
        self.g = 1
        self.altitude_d = 0.5

        for propeller in self.propellers:
            propeller.setPosition(float('inf'))
            propeller.setVelocity(0)

        # State
        self.current_pose = 6 * [0]  # X, Y, Z, yaw, pitch, roll
        self.target_position = [0, 0, 0]
        self.target_altitude = 0

        # ROS interface
        rclpy.init(args=None)
        self.node = rclpy.create_node('mavic_driver')
        self.node.create_subscription(String, 'voice_command', self.listener_callback,1)
        self.node.create_subscription(Yolov8Inference, "/Yolov8_Inference", self.yolo_callback, 1)
    
    def listener_callback(self, msg):
        self.target_name = []
        unfiltered_name = msg.data
        unfiltered_name = unfiltered_name.replace('.', '')
        unfiltered_name = unfiltered_name.replace('!', '')
        unfiltered_name = unfiltered_name.replace(',', '')
        unfiltered_name = unfiltered_name.lower()
        unfiltered_list = unfiltered_name.spllistener_callbackit()
        for item in (set(unfiltered_list) - set(del_dict)):
            self.target_name.append(item)
            # pass

    def yolo_callback(self, msg):
        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
        self.distance = None
        
        if self.target_name != []:
            if msg.yolov8_inference == []:
                self.flag = False
            else:
                for r in msg.yolov8_inference:
                    self.class_name = r.class_name
                    if self.class_name == self.target_name[0]:
                        self.top = r.top
                        self.left = r.left
                        self.bottom = r.bottom
                        self.right = r.right
                        self.distance = r.distance
                        break
                if self.top == None:
                    self.flag = False
                else:
                    self.flag = True
    
    def set_position(self, pos):
        self.current_pose = pos

    def initials(self):
        # Read sensors
        self.roll, self.pitch, self.yaw = self.imu.getRollPitchYaw()
        self.x_pos, self.y_pos, self.altitude = self.gps.getValues()
        self.roll_acceleration, self.pitch_acceleration, self.yaw_acceleration = self.gyro.getValues()
        self.set_position([self.x_pos, self.y_pos, self.altitude, self.roll, self.pitch, self.yaw])
        self.roll_disturbance = 0
        self.pitch_disturbance = 0
        self.dis_down = self.dis_sensor_down.getValue()
        self.dis_for = self.dis_sensor_for.getValue()
        self.dis_left = self.dis_sensor_left.getValue()
        self.dis_right = self.dis_sensor_right.getValue()
        self.dis_back = self.dis_sensor_back.getValue()
        self.dis_up = self.dis_sensor_up.getValue()
        self.yaw_disturbance = 0
        velocity = self.gps.getSpeed()
        self.velocity_x = (self.pitch / (abs(self.roll) + abs(self.pitch))) * velocity
        self.velocity_y = - (self.roll / (abs(self.roll) + abs(self.pitch))) * velocity
        self.vertical_input = None

    def controllers(self, vx=None, vy=None):
        # Read sensors
        linear_x_integral = 0
        linear_y_integral = 0
        linear_x_error = 0 if vx is None else vx - self.velocity_x
        linear_y_error = 0 if vy is None else vy - self.velocity_y
        linear_x_integral += linear_x_error
        linear_y_integral += linear_y_error
        pitch_disturbance = - K_X_VELOCITY_P * linear_x_error - K_X_VELOCITY_I * linear_x_integral
        roll_disturbance = K_Y_VELOCITY_P * linear_y_error + K_Y_VELOCITY_I * linear_y_integral
        return pitch_disturbance, roll_disturbance
    
    def step(self):
        rclpy.spin_once(self.node, timeout_sec=0)
        # Read sensors
        self.initials()
        
        #if self.g == 1:
            #self.target_name =['up']
        #if self.altitude > 2:
        # self.target_name =['car']
        self.g =  0
        distance_sensors = [self.dis_down, self.dis_for, self.dis_right, self.dis_left, self.dis_up, self.dis_down]
        
        if self.altitude > 0.5:
            if self.target_name != []:
                if any(item in self.target_name for item in motion_dict):
                    if 'stop' in self.target_name:
                        self.vertical_input = 0.3045
                    elif 'turn' in self.target_name:
                        self.vertical_input = 0.3045
                        if 'left' in self.target_name:
                            self.yaw_disturbance = 0.5
                        else:
                            self.yaw_disturbance = -0.5
                    elif 'left' in self.target_name and not ('turn' in self.target_name):
                        self.vertical_input = 0.3045
                        if self.dis_left>0.2:
                            _, self.roll_disturbance = self.controllers(vy=1)
                        else:
                            print('There is an obstacle on the left side')
                    elif 'right' in self.target_name and not ('turn' in self.target_name):
                        self.vertical_input = 0.3045
                        if self.dis_right>0.2:
                            _, self.roll_disturbance = self.controllers(vy=-1)
                        else:
                            print('There is an obstacle on the right side')
                    elif 'up' in self.target_name:
                        if self.dis_up>0.2:
                            self.vertical_input = K_VERTICAL_P * 0.5
                        else:
                            print('There is an obstacle above')
                            self.vertical_input = 0.3045
                    elif 'down' in self.target_name:
                        if self.dis_down>0.2:
                            self.vertical_input = K_VERTICAL_P * -0.5
                        else:
                            self.vertical_input = 0.3045
                            print('There is an obstacle downside')
                    elif 'forward' in self.target_name:
                        if self.dis_for>0.2:
                            self.pitch_disturbance, _ = self.controllers(vx=4)
                            self.vertical_input = 0.5247
                        else:
                            self.vertical_input = 0.3045
                            print('There is an obstacle ahead')
                    elif 'backward' in self.target_name:
                        if self.dis_back>0.2:
                            self.pitch_disturbance,  _ = self.controllers(vx=-1)
                            self.vertical_input = 0.5247
                        else:
                            self.vertical_input = 0.3045
                            print('There is an obstacle backside')
                else:
                    if self.flag:
                        object_cx = (self.right + self.left)/2
                        object_cy = (self.top + self.bottom)/2
                        condition_x = object_cx - 320
                        condition_y = object_cy - 240
                        altitude_d = self.altitude + 0.1 if condition_y < -5 else self.altitude - 0.1 if condition_y > 5 else 0
                        self.yaw_disturbance = -0.1 if condition_x > 5 else 0.1 if condition_x < -5 else 0
                        #altitude_d = clamp(
                        #altitude_d + 0 * (self.timestep / 1000),
                        #max(self.altitude - 0.25, LIFT_HEIGHT),
                        #self.altitude + 0.25)
                        self.vertical_input = 0.3045

                        # Second edit
                        if self.dis_down<0.2:
                            if self.distance >= 2 and self.dis_for > 0.2:
                                self.pitch_disturbance, _ = self.controllers(vx=0.5)
                                self.vertical_input = 0.5247
                            else:
                                self.vertical_input = 0.3045 
                        elif (-object_cx/2<condition_x<object_cx/2):
                            if 3 < self.distance:
                                self.pitch_disturbance, _ = self.controllers(vx=1.5)
                                self.vertical_input = 0.4247
                            elif 2<= self.distance <=3 and not (-object_cy/2<condition_y<object_cy/2):
                                self.vertical_input = K_VERTICAL_P * (altitude_d - self.altitude) + 0.3045
                            elif 2<= self.distance <= 3 and (-object_cy/2<condition_y<object_cy/2):
                                self.pitch_disturbance, _ = self.controllers(vx=0.5)
                            elif self.yaw_disturbance == 0 and 1 < self.distance < 2:
                                print('we have reached the  position')
                                self.pitch_disturbance, _ = self.controllers(vx=0)
                            elif self.distance <= 1:
                                self.pitch_disturbance, _ = self.controllers(vx=-0.5)

                    else:
                        if self.yaw < 3.138 and self.flag3 == False:
                            self.yaw_disturbance =  0.5
                            self.vertical_input = 0.3045
                        else:
                            if self.dis_up > 0.2:
                                self.flag3 = True
                                self.yaw_disturbance = 0
                                if self.flag2 == False:
                                    self.altitude_d = self.altitude + 0.5
                                    self.flag2 = True
                                self.vertical_input = 0.8
                                if self.altitude >= self.altitude_d:                             
                                    self.flag2 = False
                                    self.flag3 = False
                                    self.vertical_input = 0.3045
                            else:
                                self.vertical_input = 0.3045
                                print('There is an obstacle above')
                            
                            


        if self.vertical_input == None:
            altitude_d = LIFT_HEIGHT
            self.vertical_input = K_VERTICAL_P * (altitude_d - self.altitude)
        roll_input = K_ROLL_P * clamp(self.roll, -1, 1) + self.roll_acceleration + self.roll_disturbance
        pitch_input = K_PITCH_P * clamp(self.pitch, -1, 1) + self.pitch_acceleration + self.pitch_disturbance
        yaw_input = K_YAW_P * (self.yaw_disturbance - self.yaw_acceleration)
        m1 = K_VERTICAL_THRUST + self.vertical_input + yaw_input + pitch_input + roll_input
        m2 = K_VERTICAL_THRUST + self.vertical_input - yaw_input + pitch_input - roll_input
        m3 = K_VERTICAL_THRUST + self.vertical_input - yaw_input - pitch_input + roll_input
        m4 = K_VERTICAL_THRUST + self.vertical_input + yaw_input - pitch_input - roll_input
        # Apply control
        self.propellers[0].setVelocity(-m1)
        self.propellers[1].setVelocity(m2)
        self.propellers[2].setVelocity(m3)
        self.propellers[3].setVelocity(-m4)