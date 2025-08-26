import re

import requests


def call_llm(
    model_name: str,
    api_key: str,
    messages: list[dict],
    base_url: str,
) -> str:
    response = requests.post(
        base_url, json={"model": model_name, "api_key": api_key, "messages": messages}
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed: {response.text}")


def process_categorize(guide: str) -> tuple[str, str, list[dict]]:
    thought, action = guide.strip().split("Action:")
    action = "Action:" + action

    res = []
    for operation in action.strip().split("\n"):
        if "group(" not in operation:
            continue
        group_id = int(operation[operation.find("(") + 1 : operation.find(",")])
        obj_list = re.findall(
            r"\w+_\d+", operation[operation.find("[") + 1 : operation.find("]")]
        )
        res.append({"id": group_id, "objects": obj_list})
    return thought, action, res


def process_group_place(guide: str) -> tuple[str, str, list[dict]]:
    thought, action = guide.strip().split("Action:")
    action = "Action:" + action

    res = []
    for operation in action.strip().split("\n"):
        if "put_on(" not in operation:
            # put on the same position is equivalent to `put_near`
            continue
        group_id = int(operation[operation.find("(") + 1 : operation.find(",")])
        position = operation[operation.find("'") + 1 : operation.rfind("'")]
        res.append({"id": group_id, "position": position})
    return thought, action, res


def process_place(guide: str) -> tuple[str, str, list[dict]]:
    thought, action = guide.strip().split("Action:")
    action = "Action:" + action

    FOUR_POSITION = set(["left", "right", "front", "back"])
    res = []
    for operation in action.strip().split("\n"):
        obj_list = re.findall(
            r"\w+_\d+", operation[operation.find("(") + 1 : operation.find(")")]
        )

        position = operation[operation.find("'") + 1 : operation.rfind("'")]
        if position not in FOUR_POSITION:
            position = None

        # Alter the relationship between objects
        if "put_on(" in operation and len(obj_list) > 1:
            res.append(
                {
                    "type": "edge",
                    "parent": obj_list[0],
                    "child": obj_list[1],
                    "relations": {"name": "on", "position": position},
                }
            )
        elif "put_near(" in operation and len(obj_list) > 1:
            res.append(
                {
                    "type": "edge",
                    "parent": obj_list[0],
                    "child": obj_list[1],
                    "relations": {"name": "near", "position": position},
                }
            )
        elif "put_in(" in operation and len(obj_list) > 1:
            res.append(
                {
                    "type": "edge",
                    "parent": obj_list[0],
                    "child": obj_list[1],
                    "relations": {"name": "in", "position": position},
                }
            )
        # Change the state of a single object
        elif "close(" in operation and len(obj_list) > 0:
            res.append({"type": "node", "object": obj_list[0], "action": "close"})
        elif "open(" in operation and len(obj_list) > 0:
            res.append({"type": "node", "object": obj_list[0], "action": "open"})
        elif "slice(" in operation and len(obj_list) > 0:
            res.append({"type": "node", "object": obj_list[0], "action": "slice"})
        elif "throw(" in operation and len(obj_list) > 0:
            res.append({"type": "node", "object": obj_list[0], "action": "throw"})
        elif "clean(" in operation and len(obj_list) > 0:
            res.append({"type": "node", "object": obj_list[0], "action": "clean"})
        elif "fold(" in operation and len(obj_list) > 0:
            res.append({"type": "node", "object": obj_list[0], "action": "fold"})
        elif "unfold(" in operation and len(obj_list) > 0:
            res.append({"type": "node", "object": obj_list[0], "action": "unfold"})
        elif "heat(" in operation and len(obj_list) > 0:
            res.append({"type": "node", "object": obj_list[0], "action": "heat"})
        elif "freeze(" in operation and len(obj_list) > 0:
            res.append({"type": "node", "object": obj_list[0], "action": "freeze"})

    return thought, action, res
