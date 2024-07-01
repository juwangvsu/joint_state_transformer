import launch
import launch_ros
import moveit_configs_utils


def generate_launch_description():
    ros2_control_hardware_type = launch.actions.DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="mock_components",
        description="ROS2 control hardware interface type to use for the launch file -- possible values: [mock_components, isaac]",
    )

    moveit_config = (
        moveit_configs_utils.MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={
                "ros2_control_hardware_type": launch.substitutions.LaunchConfiguration(
                    "ros2_control_hardware_type"
                )
            },
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"]
        )
        .to_moveit_configs()
    )

    return launch.LaunchDescription(
        [
            ros2_control_hardware_type,
            launch_ros.actions.Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[moveit_config.robot_description],
            ),
            launch_ros.actions.Node(
                executable="transformer",
                package="joint_state_transformer",
                output="screen",
                parameters=[
                    {
                        "load": "hf:yong-tang/cspace",
                    }
                ],
                remappings=[
                    ("~/robot_description", "/robot_description"),
                ],
            ),
        ]
    )
