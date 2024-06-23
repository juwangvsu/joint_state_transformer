import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped

import importlib


class ReceiverNode(Node):
    def __init__(self):
        super().__init__("pose_receiver")

        self.declare_parameter("pose", "~/pose")
        self.declare_parameter("plugin", rclpy.Parameter.Type.STRING)
        self.declare_parameter("params", rclpy.Parameter.Type.STRING_ARRAY)

        self.pose_ = self.get_parameter("pose").get_parameter_value().string_value
        plugin = self.get_parameter("plugin").get_parameter_value().string_value
        params = dict(
            map(
                lambda e: tuple(e.split("=", maxsplit=1)),
                self.get_parameter("params").get_parameter_value().string_array_value,
            )
        )

        self.get_logger().info(
            "parameters: pose={} plugin={} params={}".format(
                self.pose_,
                plugin,
                params,
            )
        )
        plugin_module, plugin_class = plugin.rsplit(".", maxsplit=1)

        def f_plugin():
            return getattr(importlib.import_module(plugin_module), plugin_class)(
                callback=self.receive_callback, **params
            )

        self.f_plugin = f_plugin
        self.publisher_ = self.create_publisher(PoseStamped, self.pose_, 10)

    def receive_callback(self, name, position, orientation):
        self.get_logger().info(
            "name={} position={}, orientation={}".format(name, position, orientation)
        )
        msg = PoseStamped()
        msg.header.frame_id = name
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = position
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = orientation
        self.publisher_.publish(msg)
        self.get_logger().info(f"publish: {msg}")


def main(args=None):
    rclpy.init(args=args)

    node = ReceiverNode()
    try:
        with node.f_plugin() as plugin:
            # getattr(importlib.import_module(node.plugin_module), plugin_class)(
            # callback=self.receive_callback, **params
            # ) as plugin:
            plugin.receive()

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
