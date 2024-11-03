import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

import message_filters
import cspace.transformers
import huggingface_hub
import pathlib
import torch


class TransformerNode(Node):
    def __init__(self):
        super().__init__("joint_state_transformer")

        self.declare_parameter("pose", "~/pose")
        self.declare_parameter("joint_states", "~/joint_states")
        self.declare_parameter("joint_commands", "~/joint_commands")
        self.declare_parameter("robot_description", "~/robot_description")
        self.declare_parameter("load", rclpy.Parameter.Type.STRING)
        self.declare_parameter("repeat", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("device", "cpu")

        self.pose_ = self.get_parameter("pose").get_parameter_value().string_value
        self.joint_states_ = (
            self.get_parameter("joint_states").get_parameter_value().string_value
        )
        self.joint_commands_ = (
            self.get_parameter("joint_commands").get_parameter_value().string_value
        )
        self.robot_description_ = (
            self.get_parameter("robot_description").get_parameter_value().string_value
        )
        self.load_ = self.get_parameter("load").get_parameter_value().string_value
        self.repeat_ = (
            self.get_parameter_or("repeat").get_parameter_value().integer_value
            if self.get_parameter_or("repeat")
            else None
        )
        self.device_ = self.get_parameter("device").get_parameter_value().string_value
        self.local_ = (
            huggingface_hub.snapshot_download(self.load_.removeprefix("hf:"))
            if self.load_.startswith("hf:")
            else None
        )
        self.get_logger().info(
            "parameters: pose={} joint_states={} joint_commands={} robot_description={} load={}(local={}) repeat={} device={}".format(
                self.pose_,
                self.joint_states_,
                self.joint_commands_,
                self.robot_description_,
                self.load_,
                self.local_,
                self.repeat_,
                self.device_,
            )
        )

        self.state_ = None
        self.kinematics_ = None

        self.publisher_ = self.create_publisher(JointState, self.joint_commands_, 10)
        self.subscription_ = self.create_subscription(
            JointState, self.joint_states_, self.subscription_callback, 10
        )
        self.description_ = self.create_subscription(
            String,
            self.robot_description_,
            self.description_callback,
            qos_profile=rclpy.qos.QoSProfile(
                depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            ),
        )

    def message_filters_callback(self, *msgs):
        self.get_logger().info(f"message_filters_callback: {msgs}")
        assert self.kinematics_.link == tuple(msg.header.frame_id for msg in msgs)
        position = torch.stack(
            tuple(
                torch.tensor(
                    (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
                )
                for msg in msgs
            ),
            dim=-1,
        )
        orientation = torch.stack(
            tuple(
                torch.tensor(
                    (
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w,
                    )
                )
                for msg in msgs
            ),
            dim=-1,
        )
        pose = cspace.torch.classes.LinkPoseCollection(
            base=self.kinematics_.base,
            name=self.kinematics_.link,
            position=position,
            orientation=orientation,
        )
        if self.state_:
            state = self.kinematics_.inverse(pose, self.state_, repeat=self.repeat_)
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(state.name)
            msg.position = list(
                state.position(self.kinematics_.spec, name).item()
                for name in state.name
            )

            self.publisher_.publish(msg)
            self.get_logger().info(f"joint_state: {msg}")

    def subscription_callback(self, msg):
        self.get_logger().info(f"subscription: {msg}")
        if self.kinematics_:
            entries = dict(zip(msg.name, msg.position))
            position = list(entries[name] for name in self.kinematics_.joint)
            self.state_ = cspace.torch.classes.JointStateCollection(
                self.kinematics_.spec, self.kinematics_.joint, position
            )

    def description_callback(self, msg):
        self.get_logger().info(f"description: {msg}")
        if not self.kinematics_:
            self.get_logger().info(f"description: kinematics load")
            spec = cspace.cspace.classes.Spec(description=msg.data)
            self.get_logger().info(f'hf download folder: {self.local_}')
            print('hf download folder ', self.local_)
            if self.local_ is None:
                kinematics = torch.load(
                    pathlib.Path(self.load_),
                    map_location=torch.device(self.device_),
                )
            else:
                kinematics = torch.load(
                    pathlib.Path(self.local_).joinpath("kinematics.pth"),
                    map_location=torch.device(self.device_),
                )
            self.kinematics_ = kinematics

            self.message_filters_ = message_filters.TimeSynchronizer(
                fs=list(
                    message_filters.Subscriber(
                        self, PoseStamped, "{}/{}".format(self.pose_, link)
                    )
                    for link in kinematics.link
                ),
                queue_size=10,
            )
            self.message_filters_.registerCallback(self.message_filters_callback)
            self.get_logger().info(f"description: kinematics done")
        self.get_logger().info(f"description: {self.kinematics_}")


def main(args=None):
    rclpy.init(args=args)

    node = TransformerNode()
    try:
        while rclpy.ok() and not node.kinematics_:
            rclpy.spin_once(node)
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
