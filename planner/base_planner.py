import logging
from abc import ABC, abstractmethod
from datetime import datetime

import zmq

from utils.file_utils import proj_dir


class Service(ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self._config = config
        self.is_stop = False

        self.context = zmq.Context()

        self.master_receiver = self.context.socket(zmq.SUB)
        self.master_receiver.connect(self.config["url"]["master"])
        self.master_receiver.setsockopt(zmq.SUBSCRIBE, b"stop")

    @property
    def config(self) -> dict:
        return self._config

    @classmethod
    def start(cls, config: dict) -> None:
        """Start as a service (used for multi-processing)

        Args:
            config (dict): config for initialize
        """
        cls(config).run()

    @abstractmethod
    def run(self) -> None:
        """Receive command and dispatch for processing"""

    def close(self) -> None:
        self.master_receiver.close()
        self.context.term()


class BasePlanner(Service):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind(self.config["url"]["recv"])
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.connect(self.config["url"]["send"])

        self.log_path = proj_dir().joinpath(
            config["log_dir"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(
            f"{self.config['log_dir'][self.config['log_dir'].rfind('/') + 1 :]}_logger"
        )
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(str(self.log_path.resolve()), mode="w")
        file_handler.setLevel(self.logger.level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] <%(name)s>: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(file_handler)

    @abstractmethod
    def process(self, input: dict) -> dict:
        """Process received information

        Args:
            input (dict): received data

        Returns:
            dict: processed result
        """

    def run(self) -> None:
        while not self.is_stop:
            try:  # receive termination
                stop_message = self.master_receiver.recv_multipart(flags=zmq.NOBLOCK)
                if len(stop_message) == 2 and stop_message[1] == b"1":
                    self.is_stop = True
            except zmq.error.Again:
                pass

            try:  # receive information
                recv = self.receiver.recv_json(flags=zmq.NOBLOCK)
                assert isinstance(recv, dict)

                res = self.process(recv)

                self.sender.send_json(res)
            except zmq.error.Again:
                pass

        # Close
        self.close()

    def close(self) -> None:
        self.sender.close()
        self.receiver.close()
        super().close()
