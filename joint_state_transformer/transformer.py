import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

import cspace.transformers
import huggingface_hub
import accelerate
import torch


class TransformerNode(Node):
    def __init__(self):
        super().__init__("joint_state_transformer")

        self.declare_parameter("pose", "~/pose")
        self.declare_parameter("joint_states", "~/joint_states")
        self.declare_parameter("robot_description", "~/robot_description")
        self.declare_parameter("checkpoint", rclpy.Parameter.Type.STRING)

        self.pose_ = self.get_parameter("pose").get_parameter_value().string_value
        self.joint_states_ = (
            self.get_parameter("joint_states").get_parameter_value().string_value
        )
        self.robot_description_ = (
            self.get_parameter("robot_description").get_parameter_value().string_value
        )
        self.checkpoint_ = (
            self.get_parameter("checkpoint").get_parameter_value().string_value
        )
        self.local_ = (
            huggingface_hub.snapshot_download(self.checkpoint_.removeprefix("hf:"))
            if self.checkpoint_.startswith("hf:")
            else None
        )
        self.get_logger().info(
            "parameters: pose={} joint_states={} robot_description={} checkpoint={}(local={})".format(
                self.pose_,
                self.joint_states_,
                self.robot_description_,
                self.checkpoint_,
                self.local_,
            )
        )

        self.kinematics_ = None

        self.publisher_ = self.create_publisher(JointState, self.joint_states_, 10)
        self.subscription_ = self.create_subscription(
            PoseStamped, self.pose_, self.subscription_callback, 10
        )
        self.description_ = self.create_subscription(
            String,
            self.robot_description_,
            self.description_callback,
            qos_profile=rclpy.qos.QoSProfile(
                depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            ),
        )

    def subscription_callback(self, msg):
        self.get_logger().info(f"subscription: {msg}")
        if self.kinematics_:
            assert self.kinematics_.link == tuple([msg.header.frame_id])
            position = torch.tensor(
                (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
            ).unsqueeze(-1)
            orientation = torch.tensor(
                (
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                )
            ).unsqueeze(-1)
            pose = cspace.torch.classes.LinkPoseCollection(
                base=self.kinematics_.base,
                name=self.kinematics_.link,
                position=position,
                orientation=orientation,
            )
            state = self.kinematics_.inverse(pose)

            msg = JointState()
            msg.header.frame_id = self.kinematics_.base
            msg.name = list(state.name)
            msg.position = list(
                state.position(self.kinematics_.spec, name).item()
                for name in state.name
            )

            self.publisher_.publish(msg)
            self.get_logger().info(f"joint_state: {msg}")

    def description_callback(self, msg):
        self.get_logger().info(f"description: {msg}")
        if not self.kinematics_:
            self.get_logger().info(f"description: kinematics load")
            kinematics = cspace.transformers.Kinematics(
                msg.data, "panda_hand", model="gpt2"
            )
            kinematics.model = accelerate.load_checkpoint_and_dispatch(
                kinematics.model, checkpoint=self.local_
            )
            self.kinematics_ = kinematics
            self.get_logger().info(f"description: kinematics done")
        self.get_logger().info(f"description: {self.kinematics_}")


def main(args=None):
    rclpy.init(args=args)

    node = TransformerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
