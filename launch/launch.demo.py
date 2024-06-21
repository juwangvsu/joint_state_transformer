from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                executable="transformer",
                package="joint_state_transformer",
                output="screen",
                parameters=[{}],
            ),
        ]
    )
