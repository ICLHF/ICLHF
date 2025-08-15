# searchProblem.py - representations of search problems
# AIFCA Python3 code Version 0.9.1 Documentation at http://aipython.org

# Artificial Intelligence: Foundations of Computational Agents
# http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2020.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from abc import ABC, abstractmethod


class Arc(object):
    """An arc has a from_node and a to_node node and a (non-negative) cost"""

    def __init__(self, from_node, to_node, cost=1, action=None):
        assert cost >= 0, (
            "Cost cannot be negative for"
            + str(from_node)
            + "->"
            + str(to_node)
            + ", cost: "
            + str(cost)
        )
        self.from_node = from_node
        self.to_node = to_node
        self.action = action
        self.cost = cost

    def __repr__(self):
        """string representation of an arc"""
        if self.action:
            return (
                str(self.from_node)
                + " --"
                + str(self.action)
                + "--> "
                + str(self.to_node)
            )
        else:
            return str(self.from_node) + " --> " + str(self.to_node)


class Path(object):
    """A path is either a node or a path followed by an arc"""

    def __init__(self, initial, arc=None):
        """initial is either a node (in which case arc is None) or
        a path (in which case arc is an object of type Arc)"""
        self.initial = initial
        self.arc = arc
        if arc is None:
            self.cost = 0
        else:
            self.cost = initial.cost + arc.cost

    def end(self):
        """returns the node at the end of the path"""
        if self.arc is None:
            return self.initial
        else:
            return self.arc.to_node

    def nodes(self):
        """enumerates the nodes for the path.
        This starts at the end and enumerates nodes in the path backwards."""
        current = self
        while current.arc is not None:
            yield current.arc.to_node
            current = current.initial
        yield current.initial

    def initial_nodes(self):
        """enumerates the nodes for the path before the end node.
        This starts at the end and enumerates nodes in the path backwards."""
        if self.arc is not None:
            for nd in self.initial.nodes():
                yield nd  # could be "yield from"

    def __repr__(self):
        """returns a string representation of a path"""
        if self.arc is None:
            return str(self.initial)
        elif self.arc.action:
            return (
                str(self.initial)
                + "\n   --"
                + str(self.arc.action)
                + "--> "
                + str(self.arc.to_node)
            )
        else:
            return str(self.initial) + " --> " + str(self.arc.to_node)


class SearchProblem(ABC):
    """A search problem consists of:
    * a start node
    * a neighbors function that gives the neighbors of a node
    * a specification of a goal
    * a (optional) heuristic function.
    The methods must be overridden to define a search problem."""

    @abstractmethod
    def start_node(self):
        """returns start node"""
        raise NotImplementedError("start_node")  # abstract method

    @abstractmethod
    def is_goal(self, node):
        """is True if node is a goal"""
        raise NotImplementedError("is_goal")  # abstract method

    @abstractmethod
    def neighbors(self, node):
        """returns a list of the arcs for the neighbors of node"""
        raise NotImplementedError("neighbors")  # abstract method

    def heuristic(self, node) -> int:
        """Gives the heuristic value of node n.
        Returns 0 if not overridden."""
        return 0
