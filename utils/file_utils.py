import json
from pathlib import Path

import tomli
import yaml


def proj_dir() -> Path:
    return Path(__file__).parent.parent


def read_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    return res


def read_prompt(prompt_path: str) -> dict:
    res = read_json(prompt_path)
    for p in res:
        p["content"] = "\n".join(p["content"])
    return res


def read_toml(toml_path: str) -> dict:
    with open(toml_path, "rb") as f:
        res = tomli.load(f)
    return res


def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        res = yaml.safe_load(f)
    assert isinstance(res, dict)
    return res


def read_txt(txt_path: str) -> list:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]
