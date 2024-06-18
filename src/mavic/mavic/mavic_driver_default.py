"""ROS2 Mavic 2 Pro driver."""

import math
import rclpy
from geometry_msgs.msg import Twist


K_VERTICAL_THRUST = 70   # with this thrust, the drone lifts.
K_VERTICAL_P = 3.0          # P constant of the vertical PID.
K_ROLL_P = 50.0             # P constant of the roll PID.
K_PITCH_P = 30.0            # P constant of the pitch PID.
K_YAW_P = 2.0
K_X_VELOCITY_P = 1
K_Y_VELOCITY_P = 1
K_X_VELOCITY_I = 0.01
K_Y_VELOCITY_I = 0.01
LIFT_HEIGHT = 1


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

        # Setting  up target
        self.target_name = 'car'
        

        # Propellers
        self.propellers = [
            self.robot.getDevice('front right propeller'),
            self.robot.getDevice('front left propeller'),
            self.robot.getDevice('rear right propeller'),
            self.robot.getDevice('rear left propeller')
        ]
        for propeller in self.propellers:
            propeller.setPosition(float('inf'))
            propeller.setVelocity(0)

        # State
        self.target_twist = Twist()
        self.vertical_ref = LIFT_HEIGHT
        self.linear_x_integral = 0
        self.linear_y_integral = 0

        # ROS interface
        rclpy.init(args=None)
        self.node = rclpy.create_node('mavic_driver')
        self.node.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 1)

    def cmd_vel_callback(self, twist):
        self.target_twist = twist

    def step(self):
        rclpy.spin_once(self.node, timeout_sec=0)

        roll_ref = 0
        pitch_ref = 0

        # Read sensors
        roll, pitch, _ = self.imu.getRollPitchYaw()
        _, _, vertical = self.gps.getValues()
        roll_velocity, pitch_velocity, twist_yaw = self.gyro.getValues()
        velocity = self.gps.getSpeed()
        if math.isnan(velocity):
            return

        # Allow high level control once the drone is lifted
        if vertical > 0.2:
            # Calculate velocity
            velocity_x = (pitch / (abs(roll) + abs(pitch))) * velocity
            velocity_y = - (roll / (abs(roll) + abs(pitch))) * velocity

            # High level controller (linear velocity)
            linear_y_error = self.target_twist.linear.y - velocity_y
            linear_x_error = self.target_twist.linear.x - velocity_x
            self.linear_x_integral += linear_x_error
            self.linear_y_integral += linear_y_error
            roll_ref = K_Y_VELOCITY_P * linear_y_error + K_Y_VELOCITY_I * self.linear_y_integral
            pitch_ref = - K_X_VELOCITY_P * linear_x_error - K_X_VELOCITY_I * self.linear_x_integral
            self.vertical_ref = clamp(
                self.vertical_ref + self.target_twist.linear.z * (self.timestep / 1000),
                max(vertical - 0.5, LIFT_HEIGHT),
                vertical + 0.5
            )
        vertical_input = K_VERTICAL_P * (self.vertical_ref - vertical)

        # Low level controller (roll, pitch, yaw)
        yaw_ref = self.target_twist.angular.z

        roll_input = K_ROLL_P * clamp(roll, -1, 1) + roll_velocity + roll_ref
        pitch_input = K_PITCH_P * clamp(pitch, -1, 1) + pitch_velocity + pitch_ref
        yaw_input = K_YAW_P * (yaw_ref - twist_yaw)

        m1 = K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
        m2 = K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
        m3 = K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input
        m4 = K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input

        # Apply control
        self.propellers[0].setVelocity(-m1)
        self.propellers[1].setVelocity(m2)
        self.propellers[2].setVelocity(m3)
        self.propellers[3].setVelocity(-m4)
