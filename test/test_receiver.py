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
def proc_sub():
    return launch.actions.ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "echo",
            "/pose_receiver/pose",
        ],
        cached_output=True,
        output="screen",
    )


@launch_pytest.fixture
def launch_description(address_rokoko, proc_pub, proc_sub):
    addr, port = address_rokoko

    return launch.LaunchDescription(
        [
            proc_pub,
            launch.actions.RegisterEventHandler(
                event_handler=launch.event_handlers.OnProcessStart(
                    target_action=proc_pub,
                    on_start=[
                        launch.actions.TimerAction(
                            period=1.0,
                            actions=[
                                launch.actions.LogInfo(
                                    msg=f"Sub waited 1 seconds; start"
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
                    "launch.rokoko.py",
                    "addr:={}".format(addr),
                    "port:={}".format(port),
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
""",
            """
header:
  stamp:
    sec: 0
    nanosec: 0
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
""",
        ]
        assert all(list((e in output) for e in true)), output

    launch_pytest.tools.process.assert_output_sync(
        launch_context, proc_sub, validate_output, timeout=5
    )
    yield
