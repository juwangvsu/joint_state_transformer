import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


class TransformerNode(Node):
    def __init__(self):
        super().__init__("joint_state_transformer")

        self.declare_parameter("pose", "~/pose")
        self.declare_parameter("joint_states", "~/joint_states")

        self.pose_ = self.get_parameter("pose").get_parameter_value().string_value
        self.joint_states_ = (
            self.get_parameter("joint_states").get_parameter_value().string_value
        )
        self.get_logger().info(
            "parameters: pose={} joint_states={}".format(
                self.pose_,
                self.joint_states_,
            )
        )

        self.publisher_ = self.create_publisher(JointState, self.joint_states_, 10)
        self.subscription_ = self.create_subscription(
            PoseStamped, self.pose_, self.subscription_callback, 10
        )

    def subscription_callback(self, msg):
        self.get_logger().info(f"subscription: pose={msg}")
        msg = JointState()
        msg.name = ["demo"]
        msg.position = [0.0]
        self.publisher_.publish(msg)


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
