from __future__ import annotations

import atexit
from collections import deque
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect


def _rgb_image(observation: dict[str, Any], camera_name: str) -> np.ndarray:
    image = np.asarray(observation["observation"][camera_name]["rgb"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{camera_name} RGB must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{camera_name} RGB must be uint8, got {image.dtype}")
    return np.ascontiguousarray(image)


def encode_observation(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    head = _rgb_image(observation, "head_camera")
    left_wrist = _rgb_image(observation, "left_camera")
    right_wrist = _rgb_image(observation, "right_camera")

    endpose = observation["endpose"]
    left_pose = np.asarray(endpose["left_endpose"], dtype=np.float32).reshape(-1)
    right_pose = np.asarray(endpose["right_endpose"], dtype=np.float32).reshape(-1)
    left_gripper = np.asarray(endpose["left_gripper"], dtype=np.float32).reshape(-1)
    right_gripper = np.asarray(endpose["right_gripper"], dtype=np.float32).reshape(-1)
    if left_pose.shape != (7,) or right_pose.shape != (7,):
        raise ValueError(
            f"RoboTwin end poses must both be [xyz+quat_wxyz], got {left_pose.shape} and {right_pose.shape}"
        )
    if left_gripper.shape != (1,) or right_gripper.shape != (1,):
        raise ValueError("RoboTwin grippers must both be scalar values")

    proprio = np.concatenate(
        [left_pose, left_gripper, right_pose, right_gripper],
    ).astype(np.float32)
    if proprio.shape != (16,) or not np.isfinite(proprio).all():
        raise ValueError("RoboTwin proprio must be a finite 16D Absolute-EEF vector")
    return head, left_wrist, right_wrist, proprio


class UnifiedVLAClient:
    def __init__(self, host: str, port: int, task_name: str, timeout: float) -> None:
        task_name = " ".join(task_name.replace("_", " ").split())
        if not task_name:
            raise ValueError("task_name must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.task_name = task_name
        self.timeout = timeout
        self.actions: deque[np.ndarray] = deque()
        self.current_proprio: np.ndarray | None = None
        self.websocket = connect(
            f"ws://{host}:{port}",
            open_timeout=timeout,
            close_timeout=5,
            ping_interval=None,
            compression=None,
            max_size=None,
            proxy=None,
        )
        atexit.register(self.close)
        print(f"Connected to UnifiedVLA server at ws://{host}:{port}")

    def close(self) -> None:
        if self.websocket is not None:
            self.websocket.close()
            self.websocket = None

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.websocket is None:
            raise RuntimeError("UnifiedVLA server connection is closed")
        try:
            self.websocket.send(
                msgpack.packb(request, default=msgpack_numpy.encode, use_bin_type=True)
            )
            response = self.websocket.recv(timeout=self.timeout)
        except ConnectionClosed as exc:
            raise RuntimeError("UnifiedVLA server connection closed unexpectedly") from exc

        if isinstance(response, str):
            raise RuntimeError(f"UnifiedVLA server error:\n{response}")
        result = msgpack.unpackb(
            response,
            object_hook=msgpack_numpy.decode,
            raw=False,
        )
        if not isinstance(result, dict):
            raise TypeError(f"Expected a response dictionary, got {type(result).__name__}")
        return result

    @staticmethod
    def _check_ok(response: dict[str, Any], method: str) -> None:
        if response.get("ok") is not True:
            raise RuntimeError(f"UnifiedVLA server rejected {method}: {response}")

    def reset(self) -> None:
        self._check_ok(self._request({"method": "reset"}), "reset")
        self.actions.clear()
        self.current_proprio = None

    def observe(self, observation: dict[str, Any]) -> None:
        head, left_wrist, right_wrist, proprio = encode_observation(observation)
        response = self._request(
            {
                "method": "observe",
                "current_cam_high": head,
                "current_cam_wrist": left_wrist,
                "current_cam_right_wrist": right_wrist,
            }
        )
        self._check_ok(response, "observe")
        self.current_proprio = proprio

    def request_action_chunk(self) -> None:
        if self.current_proprio is None:
            raise RuntimeError("An observation is required before action inference")
        response = self._request(
            {
                "method": "infer",
                "current_proprio": self.current_proprio,
                "task_description": self.task_name,
            }
        )
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (4, 16):
            raise ValueError(f"Expected UnifiedVLA actions [4,16], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("UnifiedVLA actions contain non-finite values")
        self.actions.extend(action.copy() for action in actions)

    def pop_action(self) -> np.ndarray:
        if not self.actions:
            self.request_action_chunk()
        return self.actions.popleft()


def get_model(usr_args: dict[str, Any]) -> UnifiedVLAClient:
    task_name = usr_args.get("task_name")
    if not isinstance(task_name, str):
        raise ValueError("task_name must be provided by the RoboTwin evaluator")
    return UnifiedVLAClient(
        host=str(usr_args.get("host", "127.0.0.1")),
        port=int(usr_args.get("port", 8000)),
        task_name=task_name,
        timeout=float(usr_args.get("timeout", 600)),
    )


def eval(TASK_ENV, model: UnifiedVLAClient, observation: dict[str, Any]) -> None:
    # The official evaluator calls eval once per environment step. Sending that observation
    # exactly once keeps the server history aligned with the four-action rolling state.
    model.observe(observation)
    TASK_ENV.take_action(model.pop_action(), action_type="ee")


def reset_model(model: UnifiedVLAClient) -> None:
    model.reset()
