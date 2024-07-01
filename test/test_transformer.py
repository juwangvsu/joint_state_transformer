import pytest
import launch
import launch_pytest


@pytest.fixture
def proc_pub():
    return launch.actions.ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "pub",
            "/joint_state_transformer/pose/panda_hand",
            "geometry_msgs/msg/PoseStamped",
            "{header: {frame_id: panda_hand, stamp: now}, pose: {position: {x: 0.2, y: 0.1, z: 0.3}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}",
        ],
        output="screen",
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
        output="screen",
    )


@launch_pytest.fixture
def launch_description(proc_pub, proc_sub):
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
            launch.actions.ExecuteProcess(
                cmd=[
                    "ros2",
                    "launch",
                    "joint_state_transformer",
                    "launch.demo.py",
                ],
                output="screen",
            ),
        ]
    )


@pytest.mark.launch(fixture=launch_description)
def test_read_stdout(proc_sub, launch_context):
    def validate_output(output):
        true = [
            """
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
- -0.9623027027027033
""",
            """
- 0.594014014014014
velocity: []
effort: []
""",
        ]
        assert all(list(entry in output for entry in true)), output

    launch_pytest.tools.process.assert_output_sync(
        launch_context, proc_sub, validate_output, timeout=25
    )
    yield
