import logging

import numpy as np

from pog.graph.params import TF_DIFF_THRESH
from pog.graph.shape import Affordance, AffordanceType
from pog.graph.utils import affTFxy, affTFxy2dof, p2cTF

DOF_TPYE = {
    "fixed": affTFxy,
    "x-y": affTFxy,
    "x-y-2dof": affTFxy2dof,
}


class Edge:
    relations: dict

    def __init__(self, parent: int, child: int) -> None:
        """Edge class for scene graph

        Args:
            parent (int): parent node id
            child (int): child node id
        """
        self.parent_id = parent
        self.child_id = child
        self.parent_to_child_tf = None
        self.relations = {}

    def __repr__(self) -> str:
        return str(self.parent_id) + " --> " + str(self.child_id)

    def __eq__(self, other) -> bool:
        affordance_equal = True
        for key, value in self.relations.items():
            affordance_equal = (
                affordance_equal
                and (value["parent"] == other.relations[key]["parent"])
                and (value["child"] == other.relations[key]["child"])
            )

        tf_equal = (
            np.max(np.abs(self.parent_to_child_tf - other.parent_to_child_tf))
            < TF_DIFF_THRESH
        )

        return affordance_equal and tf_equal

    def add_relation(
        self,
        parent_aff: Affordance,
        child_aff: Affordance,
        dof_type="x-y",
        pose=[0, 0, 0],
    ):
        """Add a relation between parent and child affordance

        Args:
            parent_aff (Affordance): parent affordance from parent node
            child_aff (Affordance): child affordance from child node
            dof_type (str, optional): DoF type. Defaults to 'x-y'.
            pose (list, optional): pose vector. Defaults to [0,0,0].
        """
        assert parent_aff.node_id == self.parent_id
        assert child_aff.node_id == self.child_id
        self.containment = (
            parent_aff.attributes["containment"]
            if "containment" in parent_aff.attributes.keys()
            else False
        )
        relation = {}
        relation["parent"] = parent_aff
        relation["child"] = child_aff
        if parent_aff.affordance_type == AffordanceType.Support:
            self.set_pose(
                parent_afftf=parent_aff.transform,
                child_afftf=child_aff.transform,
                pose=pose,
                dof=dof_type,
            )
            relation["pose"] = pose
            relation["dof"] = dof_type
            relation["mass"] = 0
            assert self.parent_to_child_tf is not None
            relation["com"] = np.dot(
                self.parent_to_child_tf, np.array(((0), (0), (0), (1)))
            )[0:3]
        else:
            logging.error(
                "Unrecognized affordance type: {}".format(
                    parent_aff.affordance_type.name
                )
            )

        self.relations[parent_aff.affordance_type] = relation

    def set_pose(self, **kwargs):
        """compute parent to child transform for current pose"""
        self.parent_to_child_tf = p2cTF(
            kwargs["parent_afftf"],
            kwargs["child_afftf"],
            DOF_TPYE[kwargs["dof"]](kwargs["pose"]),
        )
