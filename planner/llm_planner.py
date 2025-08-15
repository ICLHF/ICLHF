import copy
import json
import logging
from typing import Callable

import networkx as nx
import zmq

from planner.base_planner import BasePlanner
from utils.file_utils import read_prompt
from utils.llm_utils import (
    call_llm,
    process_categorize,
    process_group_place,
    process_place,
)


class LLMPlanner(BasePlanner):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.model_name: str = config["model"]["name"]
        self.base_url: str = config["model"]["url"]
        self.max_trial: int = config["model"]["max_trial"]
        self.max_context: int = config["max_context"]

        self.categorize_template_prompt: dict = read_prompt(
            config["prompt_path"]["categorize"]
        )
        self.group_place_template_prompt: dict = read_prompt(
            config["prompt_path"]["group_place"]
        )
        self.place_template_prompt: dict = read_prompt(config["prompt_path"]["place"])
        self.preference_template_prompt: dict = read_prompt(
            config["prompt_path"]["preference"]
        )

        self.scene_graph: nx.DiGraph = nx.DiGraph()

        self.preference_history: list[dict] = []
        self.group_place_llm_history: list[dict] = []
        self.place_llm_history: list[dict] = []

        self.groups: list[dict] = []

        logging.info(
            "LLM Planner started at:\n"
            f"recv: {self.receiver.getsockopt(zmq.LAST_ENDPOINT)}\n"
            f"send: {self.sender.getsockopt(zmq.LAST_ENDPOINT)}"
        )

    def infer_profile(self) -> str:
        prompt = [
            {
                "role": "system",
                "content": "You are an expert AI system specializing in human preference analysis and profile reconstruction.",
            },
            {
                "role": "user",
                "content": "\n".join(i["content"] for i in self.preference_history),
            },
        ]
        profile = call_llm(self.model_name, prompt, self.base_url)
        return profile.strip()

    def infer_preference(self, user_content: str) -> str:
        # Prepare llm prompt
        preference_prompt = copy.deepcopy(self.preference_template_prompt)
        assert isinstance(preference_prompt, list)
        preference_prompt.append({"role": "user", "content": user_content})

        # Get llm guide
        guide = call_llm(self.model_name, preference_prompt, self.base_url)
        if "Human preference:" in guide:
            guide = guide[guide.find("Human preference:") + len("Human preference:") :]
        return guide.strip()

    def extract_preference(self, data: list[dict], stage: str) -> str | None:
        preference = ""
        for i in data:
            if i["stage"] != stage:
                continue
            if i["type"] == 0:  # direct instructions
                preference += i["content"]
            else:  # adjustment
                preference += self.infer_preference(i["content"])
        if preference == "":
            return None
        else:
            self.preference_history.append({"stage": stage, "content": preference})
            return preference

    def extract_physical_feedback(self, data: list[dict], stage: str) -> str | None:
        physical_feedback = ""
        for i in data:
            if i["stage"] != stage:
                continue
            physical_feedback += i["content"]
        if physical_feedback == "":
            return None
        else:
            return physical_feedback

    def get_guide(
        self, prompt: list, post_fn: Callable[[str], tuple[str, str, list[dict]]]
    ) -> tuple[str, str, list[dict]]:
        for i in range(self.max_trial):
            guide = call_llm(self.model_name, prompt, self.base_url)
            try:
                thought, action, res = post_fn(guide)
                logging.info(
                    f"LLM thought:\n{thought}\n"
                    f"LLM action:\n{action}\n"
                    f"Parsed result:\n{res}"
                )
                return thought, action, res
            except Exception as e:
                logging.warning(f"{i + 1}'s trail failed with {e}\nGuide:\n{guide}")
        return "", "", []

    def process(self, input: dict) -> dict:
        match input["task"]:
            case "start":
                self.start_task(input)
            case "modify":
                self.modify_task(input)
            case _:
                raise NotImplementedError(
                    f"Wrong task ({input['task']}) for LLMPlanner"
                )

        # Optional: introspection to extract higher-level human characteristics
        if (
            len("".join(i["content"] for i in self.preference_history))
            > self.max_context
        ):
            logging.info(f"[Introspection] Human profile: {self.infer_profile()}")
            self.preference_history = []

        # Format output
        output = json.loads(json.dumps(self.scene_graph, default=nx.node_link_data))
        return output

    def start_task(self, input: dict) -> None:
        self.scene_graph = nx.DiGraph()
        self.preference_history = []
        self.group_place_llm_history = []
        self.place_llm_history = []
        self.groups = []

        # ==============================
        # Step 1: categorize all objects
        # ==============================
        self.categorize(input["objects"])

        # ==========================================
        # Step 2: decide the placement of each group
        # ==========================================
        assert len(input["supporters"]) == 1  # only one supporter here
        self.group_place(
            input["supporters"][0],
            preference=self.extract_preference(
                input["human_preference"], "group_place"
            ),
        )

        # ==============================================
        # Step 3: decide the placement within each group
        # ==============================================
        for idx, grp in enumerate(self.groups):
            if len(grp["objects"]) < 3:
                continue
            self.place(
                idx,
                preference=self.extract_preference(input["human_preference"], "place"),
            )

    def modify_task(self, input) -> None:
        # ============================
        # Step 1: check categorization
        # ============================
        assert len(self.groups) != 0, "No previous execution!"

        # ==========================================
        # Step 2: modify the placement of each group
        # ==========================================
        # 2.1 Preprocess human preference
        preference = self.extract_preference(input["human_preference"], "group_place")

        # 2.2 Preprocess physical feedback
        physical_feedback = self.extract_physical_feedback(
            input["physical_feedback"], "group_place"
        )

        if preference or physical_feedback:
            # 2.3 Clear previous result
            for n in self.scene_graph.nodes:
                self.scene_graph.nodes[n]["position"] = None
                self.scene_graph.nodes[n]["supporter"] = None

            # 2.4 Modify group place
            assert len(input["supporters"]) == 1  # only one supporter here
            self.group_place(input["supporters"][0], preference, physical_feedback)

        # ==============================================
        # Step 3: modify the placement within each group
        # ==============================================
        # 3.1 Preprocess human preference
        preference = self.extract_preference(input["human_preference"], "place")

        # 3.2 Preprocess physical feedback
        physical_feedback = self.extract_physical_feedback(
            input["physical_feedback"], "place"
        )

        if preference or physical_feedback:
            for idx, grp in enumerate(self.groups):
                if len(grp["objects"]) < 3:
                    continue
                # 3.3 Tracking error information
                tracking_info = ""
                if physical_feedback:
                    for collision_str in physical_feedback.split("\n"):
                        collision_objects = collision_str.split(" collided with ")
                        if set(grp["objects"]).isdisjoint(collision_objects):
                            continue
                        tracking_info += (
                            f"{collision_str}.\nIt is due to the following reason(s):\n"
                        )
                        my_graph = self.scene_graph.nodes[grp["id"]]["scene_graph"]
                        assert isinstance(my_graph, nx.DiGraph)
                        for cur_obj in set(grp["objects"]).intersection(
                            collision_objects
                        ):
                            parents = list(my_graph.predecessors(cur_obj))
                            for parent in parents:
                                for out_edge in my_graph.out_edges(parent):
                                    relations = my_graph.edges[out_edge]["relations"]
                                    tracking_info += f"{out_edge[0]} is {relations['name']} {out_edge[1]}, and its position is {relations['position']}\n"
                if tracking_info == "":
                    tracking_info = None
                if not preference and not tracking_info:
                    # this group is ok
                    continue

                # 3.4 Clear previous result
                my_graph = self.scene_graph.nodes[grp["id"]]["scene_graph"]
                assert isinstance(my_graph, nx.DiGraph)
                my_graph.clear_edges()
                nx.set_node_attributes(my_graph, None, "action")

                # 3.5 Modify corresponding place
                self.place(idx, preference, tracking_info)

    def categorize(self, objects: list) -> None:
        # Prepare llm prompt
        categorize_prompt = copy.deepcopy(self.categorize_template_prompt)
        assert isinstance(categorize_prompt, list)
        categorize_prompt.append(
            {
                "role": "user",
                "content": f"There are some objects on the table: {', '.join(i['name'] for i in objects)}",
            }
        )

        # Get llm guide
        _, _, self.groups = self.get_guide(categorize_prompt, process_categorize)

        # Check output
        assert len(self.groups) != 0, "[Categorize] LLM failed."
        all_objects = [i["name"] for i in objects]
        parsed_objects = []
        for i in self.groups:
            parsed_objects.extend(i["objects"])

        intersect = set(all_objects) & set(parsed_objects)

        left_objects = [i for i in all_objects if i not in intersect]
        if len(left_objects) != 0:
            logging.warning(f"[Categorize] Left objects: {', '.join(left_objects)}")
        nonexistent_objects = [i for i in parsed_objects if i not in intersect]
        if len(nonexistent_objects) != 0:
            logging.warning(
                f"[Categorize] LLM generate nonexistent objects: {', '.join(nonexistent_objects)}"
            )

        # Preprocess
        if len(nonexistent_objects) != 0:
            for i in self.groups:  # just remove nonexistent objects
                i["objects"] = [j for j in i["objects"] if j not in nonexistent_objects]
        if len(left_objects) != 0:
            self.groups.append({"id": len(self.groups) + 1, "objects": left_objects})
        logging.info(f"[Categorize] After preprocessed:\n{self.groups}")

        # Build initial scene graph
        objects_info = {
            obj_info["name"]: {
                "pos": obj_info["pos"],
                "ori": obj_info["ori"],
                "bbox": obj_info["bbox"],
                "action": None,
            }
            for obj_info in objects
        }
        for grp in self.groups:
            id = grp["id"]
            name = f"G_{id}"
            my_graph = nx.DiGraph(id=id, name=name)
            my_graph.add_nodes_from(
                [(k, v) for k, v in objects_info.items() if k in grp["objects"]]
            )
            self.scene_graph.add_node(
                id, name=name, scene_graph=my_graph, position=None, supporter=None
            )

    def group_place(
        self,
        supporter: str,
        preference: str | None = None,
        physical_feedback: str | None = None,
    ) -> None:
        # Prepare llm prompt
        group_place_prompt = copy.deepcopy(self.group_place_template_prompt)
        assert isinstance(group_place_prompt, list)
        for i in self.group_place_llm_history:
            group_place_prompt.append(i)
        this_query = {
            "role": "user",
            "content": json.dumps(self.groups)
            + f"\nHuman preference: {preference}"
            + f"\nPhysical feedback: {physical_feedback}",
        }
        group_place_prompt.append(this_query)

        # Get llm guide
        thought, action, res = self.get_guide(group_place_prompt, process_group_place)

        # Check output
        assert len(res) != 0, "[Group Place] LLM failed."
        if len(set(i["id"] for i in res)) != len(self.groups):
            logging.warning(
                f"[Group Place] Parsed groups:\n{', '.join(str(i['id']) for i in res)}\ndo not cover all groups!"
            )

        # Refine scene graph
        for grp in res:
            self.scene_graph.nodes[grp["id"]]["position"] = grp["position"]
        for grp in self.groups:
            self.scene_graph.nodes[grp["id"]]["supporter"] = supporter

        # Record this query
        self.group_place_llm_history.append(this_query)
        self.group_place_llm_history.append(
            {"role": "assistant", "content": f"{thought}\n{action}"}
        )

    def place(
        self,
        group_idx: int,
        preference: str | None = None,
        physical_feedback: str | None = None,
    ) -> None:
        objects = self.groups[group_idx]["objects"]

        # Prepare llm prompt
        place_prompt = copy.deepcopy(self.place_template_prompt)
        assert isinstance(place_prompt, list)
        for i in self.place_llm_history:
            if i["group_id"] != self.groups[group_idx]["id"]:
                continue
            place_prompt.append({"role": i["role"], "content": i["content"]})
        this_query = {
            "role": "user",
            "content": f"There are some objects on the table: {', '.join(objects)}"
            + f"\nHuman preference: {preference}"
            + f"\nPhysical feedback: {physical_feedback}",
        }
        place_prompt.append(this_query)

        # Get llm guide
        thought, action, res = self.get_guide(place_prompt, process_place)

        # Check output
        assert len(res) != 0, "[Place] LLM failed."
        parsed_objects = set()
        for i in res:
            match i["type"]:
                case "edge":
                    parsed_objects.add(i["parent"])
                    parsed_objects.add(i["child"])
                case "node":
                    parsed_objects.add(i["object"])

        intersect = set(objects) & parsed_objects

        left_objects = [i for i in objects if i not in intersect]
        if len(left_objects) != 0:
            logging.info(f"[Place] Left objects: {', '.join(left_objects)}")
        nonexistent_objects = [i for i in parsed_objects if i not in intersect]
        if len(nonexistent_objects) != 0:
            logging.warning(
                f"[Place] LLM generate nonexistent objects: {', '.join(nonexistent_objects)}"
            )

        # Preprocess
        idx2remove = []
        if len(nonexistent_objects) != 0:
            for idx, i in enumerate(
                res
            ):  # remove nonexistent objects and corresponding actions
                match i["type"]:
                    case "edge":
                        if (
                            i["parent"] in nonexistent_objects
                            or i["child"] in nonexistent_objects
                        ):
                            idx2remove.append(idx)
                    case "node":
                        if i["object"] in nonexistent_objects:
                            idx2remove.append(idx)
        for i in reversed(idx2remove):
            res.pop(i)

        logging.info(f"[Place] After preprocessed:\n{res}")

        # Refine scene graph
        my_graph = self.scene_graph.nodes[self.groups[group_idx]["id"]]["scene_graph"]
        assert isinstance(my_graph, nx.DiGraph)
        for i in res:
            match i["type"]:
                case "edge":
                    my_graph.add_edge(i["parent"], i["child"], relations=i["relations"])
                case "node":
                    my_graph.nodes[i["object"]]["action"] = i["action"]

        # Record this query
        this_query["group_id"] = self.groups[group_idx]["id"]
        self.place_llm_history.append(this_query)
        self.place_llm_history.append(
            {
                "group_id": self.groups[group_idx]["id"],
                "role": "assistant",
                "content": f"{thought}\n{action}",
            }
        )
