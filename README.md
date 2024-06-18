# navigation-of-drone-with-speech-recognition
Control and navigation of a drone using speech recognition, object detection and object tracking
# How to run
```
cd navigation-of-drone-with-speech-recognition
colcon build
source ../project/bin/activate
source install/setup.zsh
ros2 launch mavic robot_launch.py
```
In another terminal
```
ros2 launch object_detection launch_yolov8.launch.py
```
In another terminal
```
ros2 run speech speech_launch.py
```