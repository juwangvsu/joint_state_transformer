import pytest
import launch
import launch_ros
import launch_pytest
import moveit_configs_utils


@pytest.fixture
def proc_pub():
    return launch.actions.ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "pub",
            "/joint_state_transformer/pose",
            "geometry_msgs/msg/PoseStamped",
            "{header: {frame_id: panda_hand, stamp: now}, pose: {position: {x: 0.2, y: 0.1, z: 0.3}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}",
        ],
        cached_output=True,
    )


@pytest.fixture
def proc_sub():
    return launch.actions.ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "echo",
            "/joint_state_transformer/joint_states",
        ],
        cached_output=True,
    )


@launch_pytest.fixture
def launch_description(proc_pub, proc_sub):
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
            proc_pub,
            launch.actions.RegisterEventHandler(
                event_handler=launch.event_handlers.OnProcessStart(
                    target_action=proc_pub,
                    on_start=[
                        launch.actions.TimerAction(
                            period=10.0,
                            actions=[
                                launch.actions.LogInfo(
                                    msg=f"Sub waited 10 seconds; start"
                                ),
                                proc_sub,
                            ],
                        ),
                    ],
                )
            ),
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
                        "checkpoint": "hf:yong-tang/cspace",
                    }
                ],
                remappings=[
                    ("~/robot_description", "/robot_description"),
                ],
            ),
        ]
    )


@pytest.mark.launch(fixture=launch_description)
def test_read_stdout(proc_sub, launch_context):
    def validate_output(output):
        true = """
header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: panda_link0
name:
- panda_joint1
- panda_joint2
- panda_joint3
- panda_joint4
- panda_joint5
- panda_joint6
- panda_joint7
position:
- -1.7939223223223226
- -0.43659539539539494
- 0.2376056056056055
- 0.08568389902945395
- 2.1622110110110113
- 3.8203431907911556
- 0.33264784784784807
velocity: []
effort: []
"""
        assert true in output, output

    launch_pytest.tools.process.assert_output_sync(
        launch_context, proc_sub, validate_output, timeout=15
    )
    yield
