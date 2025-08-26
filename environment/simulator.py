from multiprocessing import Process
from queue import Empty, Queue
from threading import Thread

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R

from environment.base_env import ActionType, Env
from environment.env_wrapper import OmniGibsonEnv
from planner.llm_planner import LLMPlanner
from planner.pog_planner import POGPlanner
from utils.file_utils import read_yaml


class Simulator:
    RESPOND_DELAY: int = 30

    def __init__(
        self, config_path: str, init_preference: list[dict] | None = None
    ) -> None:
        env_cfg = read_yaml(config_path)

        self.env: Env = OmniGibsonEnv(env_cfg)
        self.env.set_view_pose((env_cfg["camera"]["pos"], env_cfg["camera"]["ori"]))

        self.is_stop: bool = False

        self.init_comm(env_cfg)
        self.init_worker(env_cfg)

        self.query_queue: Queue[dict] = Queue()
        self.action_listener: Thread = Thread(target=self.register_action)
        self.action_listener.start()

        self.human_preference: Queue[dict] = Queue()
        if init_preference is not None:
            for i in init_preference:
                self.human_preference.put(i)

    def init_comm(self, config: dict) -> None:
        self.context: zmq.Context = zmq.Context()

        self.master_sender: zmq.Socket = self.context.socket(zmq.PUB)
        self.master_sender.bind(read_yaml(config["workers"]["master"])["url"]["master"])

        self.task_sender: zmq.Socket = self.context.socket(zmq.PUSH)
        self.task_sender.connect(
            read_yaml(config["workers"]["llm_planner"])["url"]["recv"]
        )

        self.action_receiver: zmq.Socket = self.context.socket(zmq.PULL)
        self.action_receiver.bind(
            read_yaml(config["workers"]["pog_planner"])["url"]["send"]
        )

    def init_worker(self, config: dict) -> None:
        workers = [LLMPlanner.start, POGPlanner.start]
        config_paths = [
            config["workers"]["llm_planner"],
            config["workers"]["pog_planner"],
        ]
        self.processes: list[Process] = []
        for worker, config_path in zip(workers, config_paths, strict=True):
            p = Process(target=worker, args=(read_yaml(config_path),))
            p.daemon = True
            p.start()
            self.processes.append(p)

    def register_action(self) -> None:
        while not self.is_stop:
            try:
                action = self.action_receiver.recv_json()
                assert isinstance(action, dict)
                self.query_queue.put(action)
            except zmq.error.ZMQError:
                # The action_receiver should be closed by main thread
                return

    def close(self) -> None:
        self.is_stop = True
        self.master_sender.send_multipart([b"stop", b"1"])

        for p in self.processes:
            p.join()

        self.master_sender.close()
        self.task_sender.close()
        self.action_receiver.close()
        self.context.term()
        self.action_listener.join()

        self.env.close()

    def start(self) -> None:
        self.query_queue.put({"task": "start"})
        self.run()

    def run(self) -> None:
        while self.env.is_running():
            self.env.step()

            if self.env.current_time_step_index() < self.RESPOND_DELAY:
                continue

            try:  # deal with query
                query = self.query_queue.get_nowait()
                self.process_query(query)
            except Empty:
                pass
        self.close()

    def get_objects_info(self, names: list) -> list:
        res = []
        for name in names:
            pos, ori = self.env.get_position_orientation(name)
            res.append(
                {"name": name, "pos": pos, "ori": ori, "bbox": self.env.get_bbox(name)}
            )
        return res

    def get_physical_feedback(self, data: dict) -> list[dict]:
        res = []
        # Get physical feedback from group place
        if data["feedback"] != "":
            res.append({"stage": "group_place", "content": data["feedback"]})

        # Get physical feedback from place
        self.env.pause()
        self.env.play()
        collision_str = ""
        for collision_pair in self.env.get_collisions(list(data["placement"].keys())):
            collision_str += f"{collision_pair[0]} collided with {collision_pair[1]}\n"
        if collision_str != "":
            res.append({"stage": "place", "content": collision_str})
        self.env.pause()
        return res

    def get_human_preference(self) -> list[dict]:
        res = []
        # Get all preferences for the current time
        while not self.human_preference.empty():
            res.append(self.human_preference.get())
        return res

    def process_query(self, query: dict) -> None:
        if "task" not in query or not isinstance(query["task"], str):
            return
        match query["task"].lower():
            case "start":
                self.task_sender.send_json(
                    {
                        "task": "start",
                        "objects": self.get_objects_info(
                            [
                                cfg["name"]
                                for cfg in self.env.config["objects"]
                                if cfg["type"] == "DatasetObject"
                            ]
                        ),
                        "supporters": self.get_objects_info(
                            [
                                cfg["name"]
                                for cfg in self.env.config["supporters"]
                                if cfg["type"] == "DatasetObject"
                            ]
                        ),
                        "human_preference": self.get_human_preference(),
                    }
                )
            case "place":
                self.do_action(query["action"])
                self.do_place(query["placement"])

                physical_feedback = self.get_physical_feedback(query)
                preferences = self.get_human_preference()

                if (
                    not physical_feedback and not preferences
                ):  # the plan is assumed valid
                    return

                self.task_sender.send_json(
                    {
                        "task": "modify",
                        "human_preference": preferences,
                        "physical_feedback": physical_feedback,
                    }
                )
            case "preference":
                assert query["stage"] in ("group_place", "place"), (
                    "only support two stages: `group_place` and `place`"
                )
                self.human_preference.put(
                    {
                        "stage": query["stage"],
                        "content": query["content"],
                        "type": query[
                            "type"
                        ],  # 0 for direct instructions and others for adjustments
                    }
                )
            case _:
                raise NotImplementedError(
                    f"The {query['task']} is currently unsupported!"
                )

    def do_action(self, actions: dict[str, str]) -> None:
        for name, action in actions.items():
            self.env.action(name, ActionType.from_str(action))

    def do_place(self, placement: dict[str, list]) -> None:
        self.env.pause()
        for name, trans in placement.items():
            trans = np.array(trans)
            self.env.set_position_orientation(
                name,
                (
                    trans[:3, 3].tolist(),
                    R.from_matrix(trans[:3, :3]).as_quat(canonical=False).tolist(),
                ),
            )
        self.env.play()
