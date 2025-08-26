import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy
import torch as th
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.utils import sim_utils
from omnigibson.utils.ui_utils import KeyboardEventHandler
from scipy.spatial.transform import Rotation as R

from environment.base_env import ActionType, Env

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True


class OmniGibsonEnv(Env):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._env: og.Environment = og.Environment(config)

        KeyboardEventHandler.initialize()
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.V,  # type: ignore
            callback_fn=self.voxelize,
        )

    def step(self) -> None:
        og.sim.step()  # type: ignore

    def pause(self) -> None:
        og.sim.pause()  # type: ignore

    def play(self) -> None:
        og.sim.play()  # type: ignore

    def current_time_step_index(self) -> int:
        return og.sim.current_time_step_index  # type: ignore

    def is_running(self) -> bool:
        return og.sim.app.is_running()  # type: ignore

    def close(self) -> None:
        self._env.close()
        og.sim.close()  # type: ignore

    def _get_object(self, name: str) -> DatasetObject | PrimitiveObject:
        return self._env.scene.object_registry("name", name)

    def get_position(self, name: str) -> list[float]:
        return self._get_object(name).get_position().detach().cpu().tolist()

    def set_position(self, name: str, pos: list[float]) -> None:
        self._get_object(name).set_position(th.tensor(pos, dtype=th.float32))

    def get_orientation(self, name: str) -> list[float]:
        return self._get_object(name).get_orientation().detach().cpu().tolist()

    def set_orientation(self, name: str, ori: list[float]) -> None:
        self._get_object(name).set_orientation(th.tensor(ori, dtype=th.float32))

    def set_view_pose(self, pose: tuple[list[float], list[float]]) -> None:
        og.sim.viewer_camera.set_position_orientation(  # type: ignore
            position=th.tensor(pose[0], dtype=th.float32),
            orientation=th.tensor(pose[1], dtype=th.float32),
        )

    def get_bbox(self, name: str) -> list[float]:
        obj = self._get_object(name)
        assert isinstance(obj, DatasetObject)
        return obj.native_bbox.detach().cpu().tolist()

    def get_collisions(self, names: list[str]) -> list[tuple[str, str]]:
        res = []
        for collision in sim_utils.get_collisions([self._get_object(i) for i in names]):
            obj_name_1 = (
                lazy.omni.isaac.core.utils.prims.get_prim_at_path(collision[0])  # type: ignore
                .GetParent()
                .GetName()
            )
            obj_name_2 = (
                lazy.omni.isaac.core.utils.prims.get_prim_at_path(collision[1])  # type: ignore
                .GetParent()
                .GetName()
            )
            res.append((obj_name_1, obj_name_2))
        return res

    def action(self, name: str, action_type: ActionType) -> None:
        obj = self._get_object(name)
        match action_type:
            case ActionType.CLOSE:
                if not obj.states or object_states.Open not in obj.states:
                    return
                obj.states[object_states.Open].set_value(False)
            case ActionType.OPEN:
                if not obj.states or object_states.Open not in obj.states:
                    return
                obj.states[object_states.Open].set_value(True)
            case ActionType.SLICE:
                # Ref `Slicing Demo`
                knifes = list(
                    filter(lambda i: "knife" in i.category, self._env.scene.objects)
                )
                if len(knifes) == 0:
                    return
                knife = np.random.choice(knifes)
                pose_0 = knife.get_position_orientation()
                knife.keep_still()
                knife.set_position_orientation(
                    position=obj.get_position_orientation()[0]
                    + th.tensor([-0.15, 0.0, 0.2], dtype=th.float32),
                    orientation=R.from_euler("xyz", [-th.pi / 2, 0, 0]).as_quat(
                        canonical=False
                    ),
                )
                # Step simulation for a bit so that apple is sliced
                for _ in range(1000):
                    self.step()
                knife.set_position_orientation(
                    position=pose_0[0], orientation=pose_0[1]
                )
            case ActionType.THROW:
                if not obj.states:
                    return
                recycling_bins = list(
                    filter(
                        lambda i: "recycling_bin" in i.category,
                        self._env.scene.objects,
                    )
                )
                if len(recycling_bins) == 0:
                    return
                recycling_bin = np.random.choice(recycling_bins)
                obj.states[object_states.Inside].set_value(recycling_bin, True)
            case ActionType.CLEAN:
                # Ref `Particle Source and Sink Demo`
                sinks = list(
                    filter(lambda i: "sink" in i.category, self._env.scene.objects)
                )
                if len(sinks) == 0:
                    return
                sink = np.random.choice(sinks)
                pose_0 = obj.get_position_orientation()
                obj.set_position_orientation(
                    position=sink.get_position_orientation()[0]
                    + th.tensor([0, 0, 0.2], dtype=th.float32),
                    orientation=pose_0[1],
                )
                for _ in range(10):
                    self.step()
                sink.states[object_states.ToggledOn].set_value(True)
                for _ in range(1000):
                    self.step()
                sink.states[object_states.ToggledOn].set_value(False)
                obj.set_position_orientation(position=pose_0[0], orientation=pose_0[1])
            case ActionType.FOLD:
                if not obj.states or object_states.Folded not in obj.states:
                    return
                obj.states[object_states.Folded].set_value(True)
            case ActionType.UNFOLD:
                if not obj.states or object_states.Unfolded not in obj.states:
                    return
                obj.states[object_states.Unfolded].set_value(True)
            case ActionType.HEAT:
                if not obj.states or object_states.Heated not in obj.states:
                    return
                obj.states[object_states.Heated].set_value(True)
            case ActionType.FREEZE:
                if not obj.states or object_states.Frozen not in obj.states:
                    return
                obj.states[object_states.Frozen].set_value(True)
            case _:
                raise NotImplementedError(f"Not implement action type: {action_type}")

    def voxelize(self) -> None:
        self.pause()
        for cfg in self._env.objects_config:
            if cfg["type"] != "DatasetObject":
                continue
            bbox_name = f"bbox_{cfg['name']}"
            bbox_obj = self._get_object(bbox_name)
            if bbox_obj is None:
                bbox_obj = PrimitiveObject(
                    name=bbox_name,
                    primitive_type="Cube",
                    scale=th.tensor(self.get_bbox(cfg["name"]), dtype=th.float32),
                    rgba=(0.5, 0.5, 0.5, 0.5),
                    visual_only=True,
                )
                self._env.scene.add_object(bbox_obj)
            else:
                bbox_obj.visible = not bbox_obj.visible
            self.set_position_orientation(
                bbox_name, self.get_position_orientation(cfg["name"])
            )
        self.play()
