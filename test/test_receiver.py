import pytest
import launch
import launch_pytest
import contextlib
import pathlib
import socket


@pytest.fixture
def address_rokoko():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()


@pytest.fixture
def proc_pub(address_rokoko, tmp_path_factory):
    addr, port = address_rokoko
    file = pathlib.Path(__file__).parent.joinpath("data", "rokoko.json")
    code = pathlib.Path.joinpath(tmp_path_factory.mktemp("python"), "rokoko.py")
    code.write_text(
        """
import sys, time, socket, pathlib, itertools;

print("program:", sys.argv)
text = pathlib.Path(sys.argv[1]).read_text().splitlines();
text = list(itertools.chain.from_iterable(itertools.repeat(text, 5)));
for i in range(len(text)):
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(text[i].encode(), (str(sys.argv[2]), int(sys.argv[3])))
    print("process:", i)
    time.sleep(1);
print("complete")
"""
    )
    return launch.actions.ExecuteProcess(
        cmd=["python3", str(code), str(file), str(addr), str(port)],
        output="screen",
    )


@pytest.fixture
def proc_sub_left():
    return launch.actions.ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "echo",
            "/pose_receiver/pose/panda_leftfinger",
        ],
        cached_output=True,
        output="screen",
    )


@pytest.fixture
def proc_sub_right():
    return launch.actions.ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "echo",
            "/pose_receiver/pose/panda_rightfinger",
        ],
        cached_output=True,
        output="screen",
    )


@launch_pytest.fixture
def launch_description(address_rokoko, proc_pub, proc_sub_left, proc_sub_right):
    addr, port = address_rokoko

    return launch.LaunchDescription(
        [
            proc_pub,
            launch.actions.RegisterEventHandler(
                event_handler=launch.event_handlers.OnProcessStart(
                    target_action=proc_pub,
                    on_start=[
                        launch.actions.TimerAction(
                            period=2.0,
                            actions=[
                                launch.actions.LogInfo(
                                    msg=f"Sub waited 1 seconds; start"
                                ),
                                proc_sub_left,
                            ],
                        ),
                    ],
                )
            ),
            launch.actions.RegisterEventHandler(
                event_handler=launch.event_handlers.OnProcessStart(
                    target_action=proc_pub,
                    on_start=[
                        launch.actions.TimerAction(
                            period=2.0,
                            actions=[
                                launch.actions.LogInfo(
                                    msg=f"Sub waited 1 seconds; start"
                                ),
                                proc_sub_right,
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
                    "launch.rokoko.py",
                    "addr:={}".format(addr),
                    "port:={}".format(port),
                ],
                output="screen",
            ),
        ]
    )


@pytest.mark.launch(fixture=launch_description)
def test_read_stdout(proc_sub_left, proc_sub_right, launch_context):
    def validate_output_left(output):
        true = [
            """
  frame_id: leftLittleTip
pose:
  position:
    x: -0.0540380739
    y: 1.00866354
    z: 0.0380104333
  orientation:
    x: -0.4792365
    y: -0.7526037
    z: 0.1730507
    w: -0.417101949
""",
            """
  frame_id: leftLittleTip
pose:
  position:
    x: -0.0540567338
    y: 1.00866377
    z: 0.03805481
  orientation:
    x: -0.479381263
    y: -0.752497435
    z: 0.173334673
    w: -0.417009145
""",
        ]
        assert all(list((e in output) for e in true)), output

    launch_pytest.tools.process.assert_output_sync(
        launch_context, proc_sub_left, validate_output_left, timeout=10
    )

    def validate_output_right(output):
        true = [
            """
  frame_id: rightLittleTip
pose:
  position:
    x: 0.3232254
    y: 1.28276742
    z: -0.31877774
  orientation:
    x: -0.0659752339
    y: 0.467119068
    z: 0.850298643
    w: -0.233322322
---
header:
  stamp:
""",
            """
  frame_id: rightLittleTip
pose:
  position:
    x: 0.32313475
    y: 1.28276086
    z: -0.3188029
  orientation:
    x: -0.0659843
    y: 0.467155
    z: 0.8502595
    w: -0.233391121
---
header:
  stamp:
""",
        ]
        assert all(list((e in output) for e in true)), output

    launch_pytest.tools.process.assert_output_sync(
        launch_context, proc_sub_right, validate_output_right, timeout=10
    )
    yield
