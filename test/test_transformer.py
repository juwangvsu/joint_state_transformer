import pytest
import launch
import launch_ros
import launch_pytest


@pytest.fixture
def proc_pub():
    return launch.actions.ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "pub",
            "/joint_state_transformer/pose",
            "geometry_msgs/msg/PoseStamped",
            "{pose: {position: {x: 0.2, y: 0.1, z: 0.3}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}",
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
    return launch.LaunchDescription(
        [
            proc_pub,
            proc_sub,
            launch_ros.actions.Node(
                executable="transformer",
                package="joint_state_transformer",
                output="screen",
                parameters=[{}],
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
  frame_id: ''
name:
- demo
position:
- 0.0
velocity: []
effort: []
"""
        assert true in output, output

    launch_pytest.tools.process.assert_output_sync(
        launch_context, proc_sub, validate_output, timeout=5
    )
    yield
