import abc
import json
import socket
import contextlib
import rclpy.logging


class Plugin(contextlib.AbstractContextManager):
    @abc.abstractmethod
    def receive(self):
        raise NotImplementedError


class Rokoko(Plugin):
    def __init__(self, now, callback, addr, port):
        self._now_ = now
        self._callback_ = callback
        self._addr_ = str(addr)
        self._port_ = int(port)

    def __enter__(self):
        self._socket_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket_.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket_.bind((self._addr_, self._port_))

        rclpy.logging.get_logger("rokoko").info(
            "socket={}".format(self._socket_.getsockname())
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._socket_:
            self._socket_.close()
        return None

    def receive(self):
        while True:
            message, address = self._socket_.recvfrom(65536)
            rclpy.logging.get_logger("rokoko").info(
                "address={} message={}".format(address, len(message))
            )
            data = json.loads(message)
            assert data["version"] == "3,0"
            assert len(data["scene"]["actors"]) == 1
            now = self._now_()
            for name, item in data["scene"]["actors"][0]["body"].items():
                position = (
                    item["position"]["x"],
                    item["position"]["y"],
                    item["position"]["z"],
                )
                orientation = (
                    item["rotation"]["x"],
                    item["rotation"]["y"],
                    item["rotation"]["z"],
                    item["rotation"]["w"],
                )
                self._callback_(name, position, orientation, now)
