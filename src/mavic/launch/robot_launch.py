#!/usr/bin/env python

"""Launch Webots Mavic 2 Pro driver."""

import os
import pathlib
import launch
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.substitutions.path_join_substitution import PathJoinSubstitution
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.utils import controller_url_prefix

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
import launch_ros.actions

def get_ros2_nodes(*args):
    pkg_mavic = get_package_share_directory('mavic')
    robot_description = pathlib.Path(os.path.join(pkg_mavic, 'resource', 'mavic_webots.urdf')).read_text()
    
    mavic_driver = Node(
        package='webots_ros2_driver',
        executable='driver',
        output='screen',
        additional_env={'WEBOTS_CONTROLLER_URL': controller_url_prefix() + 'Mavic_2_PRO'},
        parameters=[
            {'robot_description': robot_description},
        ]
    )

    return [
        mavic_driver,
    ]


def generate_launch_description():
    pkg_mavic = get_package_share_directory('mavic')
    world = LaunchConfiguration('world')
    pkg_speech = get_package_share_directory('speech')
    
    spawn_speech = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_speech, 'launch', 'speech_launch.launch.py'),
        )
    )

    rviz_config_file = os.path.join(pkg_mavic, 'launch', 'rviz_config.rviz')

    # Create a node to launch RViz
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='my_rviz',
        output='screen',
        arguments=['-d', rviz_config_file])
    
    webots = WebotsLauncher(
        world=PathJoinSubstitution([pkg_mavic, 'worlds', world]),
        ros2_supervisor=True
    )

    # The following line is important!
    # This event handler respawns the ROS 2 nodes on simulation reset (supervisor process ends).
    reset_handler = launch.actions.RegisterEventHandler(
        event_handler=launch.event_handlers.OnProcessExit(
            target_action=webots._supervisor,
            on_exit=get_ros2_nodes,
        )
    )

    return LaunchDescription([
        #spawn_speech,
        rviz_node,
        DeclareLaunchArgument(
            'world',
            default_value='mavic_world.wbt',
            description='Choose one of the world files from `/mavic/worlds` directory'
        ),
        webots,
        webots._supervisor,

        # This action will kill all nodes once the Webots simulation has exited
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[
                    launch.actions.UnregisterEventHandler(
                        event_handler=reset_handler.event_handler
                    ),
                    launch.actions.EmitEvent(event=launch.events.Shutdown())
                ],
            )
        ),

        # Add the reset event handler
        reset_handler
    ] + get_ros2_nodes())
