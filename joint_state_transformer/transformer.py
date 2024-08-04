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
        self.declare_parameter("robot_description", "~/robot_description")
        self.declare_parameter("load", rclpy.Parameter.Type.STRING)

        self.pose_ = self.get_parameter("pose").get_parameter_value().string_value
        self.joint_states_ = (
            self.get_parameter("joint_states").get_parameter_value().string_value
        )
        self.robot_description_ = (
            self.get_parameter("robot_description").get_parameter_value().string_value
        )
        self.load_ = self.get_parameter("load").get_parameter_value().string_value
        self.local_ = (
            huggingface_hub.snapshot_download(self.load_.removeprefix("hf:"))
            if self.load_.startswith("hf:")
            else None
        )
        self.get_logger().info(
            "parameters: pose={} joint_states={} robot_description={} load={}(local={})".format(
                self.pose_,
                self.joint_states_,
                self.robot_description_,
                self.load_,
                self.local_,
            )
        )

        self.kinematics_ = None

        self.publisher_ = self.create_publisher(JointState, self.joint_states_, 10)
        self.description_ = self.create_subscription(
            String,
            self.robot_description_,
            self.description_callback,
            qos_profile=rclpy.qos.QoSProfile(
                depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            ),
        )

    def message_filters_callback(self, *msgs):
        self.get_logger().info(f"subscription: {msgs}")
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

        state = self.kinematics_.inverse(pose)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(state.name)
        msg.position = list(
            state.position(self.kinematics_.spec, name).item() for name in state.name
        )

        self.publisher_.publish(msg)
        self.get_logger().info(f"joint_state: {msg}")

    def description_callback(self, msg):
        self.get_logger().info(f"description: {msg}")
        if not self.kinematics_:
            self.get_logger().info(f"description: kinematics load")
            spec = cspace.cspace.classes.Spec(description=msg.data)
            kinematics = torch.load(
                pathlib.Path(self.local_).joinpath("kinematics.pth")
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
