import sys
import launch
import launch_ros


def generate_launch_description():
    addr = set(
        filter(
            None,
            map(
                lambda e: (
                    e.removeprefix("addr:=") if e.startswith("addr:=") else None
                ),
                sys.argv,
            ),
        )
    )
    assert len(addr) == 0 or len(addr) == 1
    addr = next(iter(addr)) if len(addr) == 1 else ""

    port = set(
        filter(
            None,
            map(
                lambda e: (
                    int(e.removeprefix("port:=")) if e.startswith("port:=") else None
                ),
                sys.argv,
            ),
        )
    )
    assert len(port) == 0 or len(port) == 1
    port = next(iter(port)) if len(port) == 1 else 14043

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                executable="receiver",
                package="joint_state_transformer",
                output="screen",
                parameters=[
                    {
                        "plugin": "joint_state_transformer.plugin.Rokoko",
                        "params": [
                            "addr={}".format(addr),
                            "port={}".format(port),
                        ],
                        "link": [
                            "leftLittleTip:panda_leftfinger",
                            "rightLittleTip:panda_rightfinger",
                        ],
                    }
                ],
            ),
        ]
    )
