import logging
import math

import numpy as np

import pog.sdf as sdf
from pog.algorithm.params import (
    EPS,
    MARGIN_COLLISION,
    SAFE_MARGIN_COLLISION,
    SINGLE_STABILITY_MARGIN,
    WEIGHT_COLLISION,
    WEIGHT_COMPACT,
    WEIGHT_ORTHOGONALITY,
    WEIGHT_SINGLE_STABILITY,
    WEIGHT_TOTAL_STABILITY,
)
from pog.graph import shape
from pog.graph.graph import Graph
from pog.graph.shape import AffordanceType

__all__ = [
    "objective",
    "objective_collision",
    "pose2arr",
    "arr2pose",
    "gen_bound",
    "objective_total_stability",
    "objective_single_stability",
    "constraint_collision",
    "constraint_single_stability",
    "objective_parent_child_stability",
    "checkConstraints",
]

DOF_BOUND = {
    "fixed": 0,
    "x-y": 3,
    "x-y-2dof": 2,
}


def checkConstraints(arr, object_pose_dict, sg: Graph):
    """check constraint satisfaction of scene graph

    Args:
        arr (list): pose array
        object_pose_dict (dict): pose dict
        sg (Graph): scene graph

    Returns:
        bool: True if all constraints are satisfied
    """
    return constraint_single_stability(
        arr, object_pose_dict.keys(), sg
    ) and constraint_collision(arr, object_pose_dict, sg)


def objective(arr, object_pose_dict, sg):
    """compute cost of scene graph

    Args:
        arr (list): pose array
        object_pose_dict (dict): pose dict
        sg (Graph): scene graph

    Returns:
        float: total cost
    """
    cost_1, step_direction_1 = objective_parent_child_stability(
        arr, object_pose_dict, sg
    )
    cost_2, step_direction_2 = objective_single_stability(arr, object_pose_dict, sg)
    cost_3, step_direction_3 = objective_collision(arr, object_pose_dict, sg)
    cost_4, step_direction_4 = objective_compact(arr, object_pose_dict, sg)
    cost_5, step_direction_5 = objective_orthogonality(arr, object_pose_dict, sg)

    total_weight = (
        WEIGHT_TOTAL_STABILITY
        + WEIGHT_SINGLE_STABILITY
        + WEIGHT_COLLISION
        + WEIGHT_COMPACT
        + WEIGHT_ORTHOGONALITY,
    )
    return cost_1 + cost_2 + cost_3 + cost_4 + cost_5, (
        step_direction_1 * WEIGHT_TOTAL_STABILITY
        + step_direction_2 * WEIGHT_SINGLE_STABILITY
        + step_direction_3 * WEIGHT_COLLISION
        + step_direction_4 * WEIGHT_COMPACT
        + step_direction_5 * WEIGHT_ORTHOGONALITY
    ) / total_weight


def objective_total_stability(arr, object_pose_dict, sg: Graph):
    """compute total stability of scene graph

    Args:
        arr (list): pose array
        object_pose_dict (dict): pose dict
        sg (Graph): scene graph

    Returns:
        float: total stability cost
    """
    pose = sg.getPose()
    pose = arr2pose(arr, object_pose_dict, pose)
    temp_cost_ = 0
    temp_cost = 0
    total_mass = 0
    for key, _ in pose.items():
        total_mass += sg.edge_dict[key].relations[shape.AffordanceType.Support]["mass"]
        temp_cost_ += sg.edge_dict[key].relations[shape.AffordanceType.Support][
            "mass"
        ] * np.array(sg.edge_dict[key].relations[shape.AffordanceType.Support]["com"])
        temp_cost += sg.edge_dict[key].relations[shape.AffordanceType.Support][
            "mass"
        ] * np.array(
            np.linalg.norm(
                sg.edge_dict[key].relations[shape.AffordanceType.Support]["com"]
            )
        )

    return WEIGHT_TOTAL_STABILITY * (
        temp_cost / total_mass + np.linalg.norm(temp_cost_) / total_mass
    ), np.zeros(len(arr))


def objective_parent_child_stability(arr, object_pose_dict, sg: Graph):
    """compute parent to child stability of scene graph

    Args:
        arr (list): pose array
        object_pose_dict (dict): pose dict
        sg (Graph): scene graph

    Returns:
        float: parent to child stability cost
    """
    first_node_parent = None
    for node in object_pose_dict.keys():
        if first_node_parent is None:
            first_node_parent = sg.edge_dict[node].parent_id
        else:
            assert (
                first_node_parent == sg.edge_dict[node].parent_id
            )  # Only optimize nodes with same parent.
    assert first_node_parent is not None

    try:
        temp_edge_relation = sg.edge_dict[first_node_parent].relations[
            shape.AffordanceType.Support
        ]
    except KeyError:
        return objective_total_stability(arr, object_pose_dict, sg)

    step_direction = np.zeros(len(arr))

    shape1 = temp_edge_relation["parent"].attributes["shape"]
    shape2 = None
    if temp_edge_relation["dof"] == "x-y":
        shape2 = (
            temp_edge_relation["child"]
            .attributes["shape"]
            .rotate(temp_edge_relation["pose"][2])
            .translate(
                np.array([temp_edge_relation["pose"][0], temp_edge_relation["pose"][1]])
            )
        )
    elif temp_edge_relation["dof"] == "x-y-2dof":
        shape2 = (
            temp_edge_relation["child"]
            .attributes["shape"]
            .translate(
                np.array([temp_edge_relation["pose"][0], temp_edge_relation["pose"][1]])
            )
        )
    elif temp_edge_relation["dof"] == "fixed":
        return 0, step_direction

    assert shape2 is not None
    intersection = sdf.d2.intersection(shape1, shape2)

    if (
        intersection(
            np.array(
                np.array([temp_edge_relation["pose"][0], temp_edge_relation["pose"][1]])
            )
        )
        / min(
            min(temp_edge_relation["child"].attributes["bb"]),
            min(temp_edge_relation["parent"].attributes["bb"]),
        )
        * 2
        + 1
        < 1e-3
    ):
        return objective_total_stability(arr, object_pose_dict, sg)

    temp_com = [
        np.dot(temp_edge_relation["parent"].get_axes("x"), temp_edge_relation["com"]),
        np.dot(temp_edge_relation["parent"].get_axes("y"), temp_edge_relation["com"]),
    ]

    cost_ = intersection(np.array(temp_com[0:2]))

    for obj in object_pose_dict.keys():
        step_direction[object_pose_dict[obj][0] : object_pose_dict[obj][0] + 2] = (
            -np.array(temp_com[0:2]) / max(EPS, float(np.linalg.norm(temp_com[0:2])))
        )

    cost_ = (
        cost_
        / min(
            min(temp_edge_relation["child"].attributes["bb"]),
            min(temp_edge_relation["parent"].attributes["bb"]),
        )
        * 2
        * WEIGHT_SINGLE_STABILITY
    )
    return cost_[0][0], step_direction


def objective_single_stability(arr, object_pose_dict, sg: Graph):
    """compute single object stability of scene graph

    Args:
        arr (list): pose array
        object_pose_dict (dict): pose dict
        sg (Graph): scene graph

    Returns:
        float: single object stability cost
    """
    pose = sg.getPose(edge_id=object_pose_dict.keys())
    step_direction = np.zeros(len(arr))

    cost = 0
    for key, value in pose.items():
        shape1 = value["parent"].attributes["shape"]
        shape2 = None
        if value["dof"] == "x-y":
            shape2 = (
                value["child"]
                .attributes["shape"]
                .rotate(value["pose"][2])
                .translate(np.array([value["pose"][0], value["pose"][1]]))
            )
        elif value["dof"] == "x-y-2dof":
            shape2 = (
                value["child"]
                .attributes["shape"]
                .translate(np.array([value["pose"][0], value["pose"][1]]))
            )

        assert shape2 is not None
        intersection = sdf.d2.intersection(shape1, shape2)
        temp_com = [
            np.dot(value["parent"].get_axes("x"), value["com"]),
            np.dot(value["parent"].get_axes("y"), value["com"]),
        ]
        cost_ = intersection(np.array(temp_com[0:2]))

        step_direction[object_pose_dict[key][0] : object_pose_dict[key][0] + 2] = (
            int(cost_ > SINGLE_STABILITY_MARGIN) * -np.array(temp_com[0:2])
        ) / max(EPS, float(np.linalg.norm(temp_com[0:2])))

        cost_ = (
            cost_
            / min(
                min(value["child"].attributes["bb"]),
                min(value["parent"].attributes["bb"]),
            )
            * 2
            * WEIGHT_SINGLE_STABILITY
        )
        cost += cost_

    assert isinstance(cost, np.ndarray)
    return cost[0][0], step_direction


def constraint_single_stability(arr, object_id, sg: Graph):
    """check single stability violation of scene graph

    Args:
        arr (list): pose array
        object_pose_dict (dict): pose dict
        sg (Graph): scene graph

    Returns:
        satisfy (bool): True if no violation is found
    """
    pose = sg.getPose(edge_id=list(object_id))
    satisfy = True
    # print(pose)
    for key, value in pose.items():
        shape1 = value["parent"].attributes["shape"]
        shape2 = None
        if value["dof"] == "x-y":
            shape2 = (
                value["child"]
                .attributes["shape"]
                .rotate(value["pose"][2])
                .translate(np.array([value["pose"][0], value["pose"][1]]))
            )
        elif value["dof"] == "x-y-2dof":
            shape2 = (
                value["child"]
                .attributes["shape"]
                .translate(np.array([value["pose"][0], value["pose"][1]]))
            )
        elif value["dof"] == "fixed":
            continue

        assert shape2 is not None
        intersection = sdf.d2.intersection(shape1, shape2)

        temp_com = [
            np.dot(value["parent"].get_axes("x"), value["com"]),
            np.dot(value["parent"].get_axes("y"), value["com"]),
        ]
        cost_ = intersection(np.array(temp_com[0:2]))

        if cost_ > SINGLE_STABILITY_MARGIN:
            logging.debug("Unstable: {}; Cost: {}".format(value["child"], cost_))
            satisfy = False
    return satisfy


def constraint_collision(arr, object_pose_dict, sg: Graph):
    """check collision violation of scene graph

    Args:
        arr (list): pose array
        object_pose_dict (dict): pose dict
        sg (Graph): scene graph

    Returns:
        satisfy (bool): True if no violation is found
    """
    objects = sg.node_dict.keys()

    for o in objects:
        sg.collision_manager.set_transform(
            name=str(o), transform=sg.global_transform[o]
        )

    is_collision, names = sg.collision_manager.in_collision_internal(  # type: ignore
        return_names=True, return_data=False
    )

    if is_collision:
        for name in names:
            logging.debug(
                "Object: {} and {} are in collision.".format(name[0], name[1])
            )

    return not is_collision


def objective_compact(arr, object_pose_dict, sg: Graph):
    res = 0.0
    step_direction = np.zeros(len(arr))

    for i in sg.depth_dict.values():
        if len(i) < 2:
            continue
        # Get level's objects' pose
        level_dist = {}
        for j in i:
            level_dist[j] = np.array(
                sg.edge_dict[j].relations[AffordanceType.Support]["pose"]
            )

        # Calculate each 2-object's manhattan distance
        for m in range(len(i)):
            for n in range(m + 1, len(i)):
                res += np.abs(level_dist[i[m]][0:2] - level_dist[i[n]][0:2]).sum()

        # Loss for area occupied
        np_dis_list = np.array([i for i in level_dist.values()])
        res += (max(np_dis_list[:, 0]) - min(np_dis_list[:, 0])) * (
            max(np_dis_list[:, 1]) - min(np_dis_list[:, 1])
        )

        level_center = np.mean(np_dis_list, axis=0)[0:2]
        for k, v in object_pose_dict.items():
            if k in level_dist:
                step_direction[v[0] : v[0] + 2] = level_center - arr[v[0] : v[0] + 2]

    return res, step_direction


def objective_orthogonality(arr, object_pose_dict, sg: Graph):
    step_direction = np.zeros(len(arr))
    half_pi = np.pi / 2

    theta = []
    for e in sg.edge_dict:
        if sg.edge_dict[e].relations[AffordanceType.Support]["dof"] != "x-y":
            continue
        my_theta = sg.edge_dict[e].relations[AffordanceType.Support]["pose"][2]
        if my_theta < 0:
            theta.append(
                my_theta % -half_pi  # box's axial symmetry
            )
        else:
            theta.append(my_theta % half_pi)
    cost = np.var(theta)

    theta_mean = np.mean(theta)
    for pose_dict in object_pose_dict.values():
        if pose_dict[1] - pose_dict[0] != DOF_BOUND["x-y"]:
            continue
        step_direction[pose_dict[1] - 1] = theta_mean - arr[pose_dict[1] - 1]

    return cost, step_direction


def objective_collision(arr, object_pose_dict, sg: Graph):
    """compute collision cost of scene graph

    Args:
        arr (list): pose array
        object_pose_dict (dict): pose dict
        sg (Graph): scene graph

    Returns:
        (float): Collision cost
    """
    objects = list(object_pose_dict.keys())
    step_direction = np.zeros(len(arr))

    for o in objects:
        sg.collision_manager.set_transform(
            name=str(o), transform=sg.global_transform[o]
        )

    _, contacts = sg.collision_manager.in_collision_internal(  # type: ignore
        return_names=False, return_data=True
    )

    collision_cost = 0

    max_penetration_dict = {key: {} for key in list(object_pose_dict.keys())}

    for contact_data in contacts:
        collision_pair = list(map(int, contact_data.names))
        if (
            collision_pair[0] in max_penetration_dict.keys()
            and contact_data.depth > MARGIN_COLLISION
            and (
                collision_pair[1] not in max_penetration_dict[collision_pair[0]].keys()
                or max_penetration_dict[collision_pair[0]][collision_pair[1]]
                < contact_data.depth
            )
        ):
            max_penetration_dict[collision_pair[0]][collision_pair[1]] = (
                contact_data.depth
            )

        if (
            collision_pair[1] in max_penetration_dict.keys()
            and contact_data.depth > MARGIN_COLLISION
            and (
                collision_pair[0] not in max_penetration_dict[collision_pair[1]].keys()
                or max_penetration_dict[collision_pair[1]][collision_pair[0]]
                < contact_data.depth
            )
        ):
            max_penetration_dict[collision_pair[1]][collision_pair[0]] = (
                contact_data.depth
            )

    for obj1, pair_dict in max_penetration_dict.items():
        total_dist = sum(pair_dict.values())
        for obj2, dist in pair_dict.items():
            if obj1 in object_pose_dict.keys() and obj2 in object_pose_dict.keys():
                direction21 = np.array(
                    arr[object_pose_dict[obj1][0] : object_pose_dict[obj1][0] + 2]
                ) - np.array(
                    arr[object_pose_dict[obj2][0] : object_pose_dict[obj2][0] + 2]
                )
                step_direction[
                    object_pose_dict[obj1][0] : object_pose_dict[obj1][0] + 2
                ] += (
                    dist
                    / total_dist
                    * direction21
                    / max(EPS, float(np.linalg.norm(direction21)))
                )
            collision_cost += dist / SAFE_MARGIN_COLLISION + 1

    return collision_cost, step_direction


def pose2arr(pose, object_pose_dict, arr=[]):
    for key, value in object_pose_dict.items():
        arr[value[0] : value[1]] = pose[key]["pose"]
    return arr


def arr2pose(arr, object_pose_dict, pose: dict):
    for key, value in object_pose_dict.items():
        pose[key]["pose"] = arr[value[0] : value[1]]
    return pose


def gen_bound(pose):
    object_pose_dict = {}
    bounds = np.empty((0, 2))
    idx = 0
    for key, value in pose.items():
        object_pose_dict[key] = (idx, idx + DOF_BOUND[value["dof"]])
        idx += DOF_BOUND[value["dof"]]
        if value["dof"] == "x-y":
            bounds = np.append(
                bounds,
                [
                    [
                        -value["parent"].attributes["bb"][0],
                        value["parent"].attributes["bb"][0],
                    ],
                    [
                        -value["parent"].attributes["bb"][1],
                        value["parent"].attributes["bb"][1],
                    ],
                    [-math.pi, math.pi],
                ],
                axis=0,
            )
        elif value["dof"] == "x-y-2dof":
            bounds = np.append(
                bounds,
                [
                    [
                        -value["parent"].attributes["bb"][0],
                        value["parent"].attributes["bb"][0],
                    ],
                    [
                        -value["parent"].attributes["bb"][1],
                        value["parent"].attributes["bb"][1],
                    ],
                ],
                axis=0,
            )

    return object_pose_dict, bounds
