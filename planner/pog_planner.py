import copy
import logging

import networkx as nx
import numpy as np
import trimesh
import zmq
from scipy.spatial.transform import Rotation as R

from planner.base_planner import BasePlanner
from pog.algorithm.structure import simulated_annealing_structure
from pog.graph import edge, node, shape
from pog.graph.graph import Graph
from pog.graph.shape import AffordanceType
from pog.planning.planner import AStarSearcher, PlanningOnGraphProblem, test_search


class POGPlanner(BasePlanner):
    ROOT_ID: int = 0
    EXTERNAL_ID: int = 99
    MAX_TRIES: int = 3

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        logging.info(
            "POG Planner started at:\n"
            f"recv: {self.receiver.getsockopt(zmq.LAST_ENDPOINT)}\n"
            f"send: {self.sender.getsockopt(zmq.LAST_ENDPOINT)}"
        )

        self.scene_graph: nx.DiGraph = nx.DiGraph()

    @classmethod
    def init_scene(cls, g: nx.DiGraph, bounds: tuple):
        node_dict = {}
        edge_dict = {}

        ground = node.Node(
            id=cls.ROOT_ID, shape=shape.Box(size=[bounds[0], bounds[1], 0.01])
        )
        node_dict[cls.ROOT_ID] = ground

        for name in g.nodes:
            id = int(name[name.rfind("_") + 1 :])
            trans = np.eye(4)
            trans[:3, :3] = R.from_quat(g.nodes[name]["ori"]).as_matrix()
            trans[:3, 3] = g.nodes[name]["pos"]
            s = shape.Box(size=g.nodes[name]["bbox"], transform=trans)
            node_dict[id] = node.Node(id=id, shape=s)
            edge_dict[id] = edge.Edge(parent=cls.ROOT_ID, child=id)
            edge_dict[id].add_relation(
                ground.affordance["box_aff_pz"],
                node_dict[id].affordance["box_aff_nz"],
                dof_type="x-y",
                pose=[
                    g.nodes[name]["pos"][0],
                    g.nodes[name]["pos"][1],
                    R.from_quat(g.nodes[name]["ori"]).as_euler("xyz")[2],
                ],
            )
        return node_dict, edge_dict, cls.ROOT_ID

    @classmethod
    def target_scene(cls, g: nx.DiGraph, bounds: tuple):
        node_dict = {}
        edge_dict = {}

        ground = node.Node(
            id=cls.ROOT_ID, shape=shape.Box(size=[bounds[0], bounds[1], 0.01])
        )
        node_dict[cls.ROOT_ID] = ground

        for name in g.nodes:
            id = int(name[name.rfind("_") + 1 :])
            s = shape.Box(size=g.nodes[name]["bbox"])
            node_dict[id] = node.Node(id=id, shape=s)
            edge_dict[id] = edge.Edge(parent=cls.ROOT_ID, child=id)
            edge_dict[id].add_relation(
                ground.affordance["box_aff_pz"],
                node_dict[id].affordance["box_aff_nz"],
                dof_type="x-y",
            )
        for e in g.edges:
            match g.edges[e]["relations"]["name"]:
                case "near":
                    s_id = int(e[0][e[0].rfind("_") + 1 :])
                    match g.edges[e]["relations"]["position"]:
                        case "left":
                            edge_dict[s_id].relations[AffordanceType.Support]["pose"][
                                1
                            ] += 0.1
                        case "right":
                            edge_dict[s_id].relations[AffordanceType.Support]["pose"][
                                1
                            ] -= 0.1
                        case "front":
                            edge_dict[s_id].relations[AffordanceType.Support]["pose"][
                                0
                            ] += 0.1
                        case "back":
                            edge_dict[s_id].relations[AffordanceType.Support]["pose"][
                                0
                            ] -= 0.1
                case "on" | "in":
                    s_id = int(e[0][e[0].rfind("_") + 1 :])
                    t_id = int(e[1][e[1].rfind("_") + 1 :])
                    edge_dict.pop(s_id)
                    edge_dict[s_id] = edge.Edge(parent=t_id, child=s_id)
                    pose = [0.0, 0.0, 0.0]
                    match g.edges[e]["relations"]["position"]:
                        case "left":
                            pose[1] += 0.1
                        case "right":
                            pose[1] -= 0.1
                        case "front":
                            pose[0] += 0.1
                        case "back":
                            pose[0] -= 0.1
                    edge_dict[s_id].add_relation(
                        node_dict[t_id].affordance["box_aff_pz"],
                        node_dict[s_id].affordance["box_aff_nz"],
                        dof_type="x-y",
                        pose=pose,
                    )
                case _:
                    raise NotImplementedError(
                        f"Not implement {g.edges[e]['relations']['name']}"
                    )

        return node_dict, edge_dict, cls.ROOT_ID

    @classmethod
    def optimize(
        cls, g: nx.DiGraph, bounds: tuple = (10, 10), need_plan: bool = False
    ) -> tuple[Graph, bool, float]:
        g_start = Graph("start", fn=cls.init_scene, g=g, bounds=bounds)
        g_end = Graph("end", fn=cls.target_scene, g=g, bounds=bounds)

        count = 0
        g_opt_goal = None
        cnt_sat = False
        best_eval_total = np.inf

        while not cnt_sat and count < cls.MAX_TRIES:
            count += 1
            g_goal, cnt_sat, eval_total = simulated_annealing_structure(g_end)
            if eval_total < best_eval_total:
                best_eval_total = eval_total
                g_opt_goal = g_goal
        assert g_opt_goal is not None

        if cnt_sat:
            logging.info(f"Optimize {g.name}: OK\nTotal cost: {best_eval_total}")
        else:
            logging.warning(f"Optimize {g.name}: Failed\nTotal cost: {best_eval_total}")

        if not need_plan:
            return g_opt_goal, cnt_sat, best_eval_total

        try:
            path = test_search(
                AStarSearcher,
                problem=PlanningOnGraphProblem(g_start, g_opt_goal, parking_place=0),
            )
            logging.info(f"Found path for {g.name}:\n{path}")
        except Exception as e:
            logging.warning(f"Could not find path for {g.name}:\n{e}")

        return g_opt_goal, cnt_sat, best_eval_total

    @classmethod
    def extract_action(cls, g: nx.DiGraph) -> dict[str, str]:
        res = {}
        for n in g.nodes:
            if g.nodes[n]["action"] is None:
                continue
            res[n] = g.nodes[n]["action"]
        return res

    def process(self, input: dict) -> dict:
        self.scene_graph: nx.DiGraph = nx.node_link_graph(input)
        group_graph: nx.DiGraph = nx.DiGraph()
        all_graphs: list[trimesh.Scene] = []
        supporter: dict = {}
        node_actions: dict[str, str] = {}
        pog_info: dict = {}

        for id in self.scene_graph.nodes:
            my_graph: nx.DiGraph = nx.node_link_graph(
                self.scene_graph.nodes[id]["scene_graph"]
            )
            node_actions.update(self.extract_action(my_graph))
            supporter = self.scene_graph.nodes[id]["supporter"]

            # Optimize single group
            g_opt_goal, cnt_sat, cost = self.optimize(
                my_graph, (supporter["bbox"][0], supporter["bbox"][1])
            )
            pog_info[id] = {"cnt_sat": cnt_sat, "cost": cost}

            # Compute bbox for this group
            g_opt_goal.create_scene()
            g_scene = g_opt_goal.scene.copy()
            g_scene.delete_geometry("geometry_0")  # delete ground
            obb = g_scene.bounding_box_oriented

            obb_box = trimesh.creation.box(
                extents=obb.primitive.extents, transform=obb.primitive.transform
            )
            g_scene.add_geometry(
                obb_box, node_name=f"group_bbox_{id}", transform=np.eye(4)
            )

            # Build group graph
            position = self.scene_graph.nodes[id]["position"]
            pos = [0.0, 0.0, 0.0]
            ori = [0, 0, 0, 1]
            if position is None or position == "center":
                pos = [0.0, 0.0, 0.0]
            else:
                if "north" in position:
                    pos[0] += 0.1
                if "south" in position:
                    pos[0] -= 0.1
                if "west" in position:
                    pos[1] += 0.1
                if "east" in position:
                    pos[1] -= 0.1

            # Compute rotation closest to abs(obb.primitive.transform)
            U, _, Vt = np.linalg.svd(np.abs(obb.primitive.transform[:3, :3]))
            extra_rot = U @ Vt
            if np.linalg.det(extra_rot) < 0:
                extra_rot = U @ np.diag([1, 1, -1]) @ Vt

            group_graph.add_node(
                self.scene_graph.nodes[id]["name"],
                pos=pos,
                ori=ori,
                bbox=(extra_rot @ obb.primitive.extents).tolist(),
            )

            # Transform g_scene to group graph
            init_trans = np.eye(4)
            init_trans[:3, 3] = pos

            delta = extra_rot @ obb.primitive.transform[:3, :3].T
            angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1.0, 1.0))
            if np.isclose(angle, np.pi, 1e-3):
                init_trans[:3, :3] = (
                    R.from_quat(ori).as_matrix()
                    @ extra_rot
                    @ R.from_euler("y", 180, degrees=True).as_matrix()
                )
            else:
                init_trans[:3, :3] = R.from_quat(ori).as_matrix() @ extra_rot

            init_trans @= np.linalg.inv(obb.primitive.transform)
            init_trans[2, 3] = (
                0  # ignore this because it is the z-axis adjustment of group bbox
            )

            g_scene.apply_transform(init_trans)
            all_graphs.append(g_scene)

        # Optimize group graph
        assert (
            len(
                set(
                    self.scene_graph.nodes[i]["supporter"]["name"]
                    for i in self.scene_graph.nodes
                )
            )
            == 1
        ), "only one supporter here"
        group_graph_init = Graph(
            "group_graph_init",
            fn=self.init_scene,
            g=group_graph,
            bounds=(
                supporter["bbox"][0],
                supporter["bbox"][1],
            ),
        )
        init_global_trans = copy.deepcopy(group_graph_init.global_transform)

        count = 0
        group_graph_opt = None
        cnt_sat = False
        best_eval_total = np.inf

        while not cnt_sat and count < self.MAX_TRIES:
            count += 1
            group_graph_goal, cnt_sat, eval_total = simulated_annealing_structure(
                group_graph_init
            )
            if eval_total < best_eval_total:
                best_eval_total = eval_total
                group_graph_opt = group_graph_goal
        assert group_graph_opt is not None

        if cnt_sat:
            logging.info(f"Optimize group_graph: OK\nTotal cost: {best_eval_total}")
        else:
            logging.warning(
                f"Optimize group_graph: Failed\nTotal cost: {best_eval_total}"
            )

        # Check group place
        feedback = ""
        half_extents = np.array(supporter["bbox"][:2]) / 2
        for k, v in group_graph_opt.global_transform.items():
            if k == self.ROOT_ID:
                continue
            if np.all(np.abs(v[:2, 3]) < half_extents):
                continue
            feedback += f"The center of group {k} extends beyond the boundaries of the supporting object.\n"

        # Transform all objects according to the group graph
        result = {}
        supporter_trans = np.eye(4)
        supporter_trans[:3, 3] = supporter["pos"]
        supporter_trans[2, 3] += supporter["bbox"][2] / 2  # top surface
        for k, v in group_graph_opt.global_transform.items():
            if k == self.ROOT_ID:
                continue
            all_graphs[k - 1].apply_transform(
                supporter_trans @ v @ np.linalg.inv(init_global_trans[k])
            )
            for obj_id in all_graphs[k - 1].graph.nodes_geometry:
                if not isinstance(obj_id, int):
                    continue
                result[obj_id] = all_graphs[k - 1].graph.get(obj_id)[0]

        obj_names = [
            j
            for i in self.scene_graph.nodes
            for j in nx.node_link_graph(
                self.scene_graph.nodes[i]["scene_graph"]
            ).nodes()
        ]
        assert len(result) == len(obj_names), "Miss some objects!"

        id_mapping = {int(name[name.rfind("_") + 1 :]): name for name in obj_names}
        output = {
            "task": "place",
            "placement": {id_mapping[k]: v.tolist() for k, v in result.items()},
            "action": node_actions,
            "feedback": feedback,
            "info": pog_info,
        }
        return output
