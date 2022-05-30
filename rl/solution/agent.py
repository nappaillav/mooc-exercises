#!/usr/bin/env python3

import io
import os

import numpy as np
from PIL import Image

from aido_schemas import (
    Context,
    DB20Commands,
    DB20ObservationsWithTimestamp,
    protocol_agent_DB20_timestamps,
    EpisodeStart,
    JPGImageWithTimestamp,
    LEDSCommands,
    no_hardware_GPU_available,
    PWMCommands,
    RGB,
    logger,
    wrap_direct,
)
from gym_wrappers import DTPytorchWrapper, FakeWrap, SpeedActionWrapper

__all__ = ["PytorchRLBaseline"]


class PytorchRLBaseline:
    image_processor: DTPytorchWrapper
    action_procerssor: SpeedActionWrapper
    def __init__(self):
        logger.info("Pytorch RL template init()")

        self.image_processor = DTPytorchWrapper()
        self.action_processor = SpeedActionWrapper(FakeWrap())
        from model import DDPG

        # self.check_gpu_available(context)

        self.model = DDPG(state_dim=self.image_processor.shape, action_dim=2, max_action=1, net_type="cnn")
        self.current_image = np.zeros((640, 480, 3))
        self.model.load("model", directory="/agent/solution/models")

    def init(self, context: Context):
        context.info("init()")

        self.image_processor = DTPytorchWrapper()
        self.action_processor = SpeedActionWrapper(FakeWrap())
        from model import DDPG

        self.check_gpu_available(context)

        self.model = DDPG(state_dim=self.image_processor.shape, action_dim=2, max_action=1, net_type="cnn")
        self.current_image = np.zeros((640, 480, 3))
        self.model.load("model", directory= os.getcwd() + "/solution/models")

    def check_gpu_available(self, context: Context):
        import torch

        available = torch.cuda.is_available()
        context.info(f"torch.cuda.is_available = {available!r}")
        context.info("init()")
        if available:
            i = torch.cuda.current_device()
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(i)
            context.info(f"device {i} of {count}; name = {name!r}")
        else:
            no_hardware_GPU_available(context)

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: DB20ObservationsWithTimestamp, context: Context):
        camera: JPGImageWithTimestamp = data.camera
        profiler = context.get_profiler()
        with profiler.prof("jpg2rgb"):
            obs = jpg2rgb(camera.jpg_data)
        with profiler.prof("preprocess"):
            self.current_image = self.image_processor.preprocess(obs)

    def compute_action(self, observation, profiler):

        with profiler.prof("predict"):
            action = self.model.predict(observation)
        with profiler.prof("action-process"):
            return self.action_processor.action(action.astype(float))

    def on_received_get_commands(self, context: Context):
        pwm_left, pwm_right = self.compute_action(self.current_image, context.get_profiler())
        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))
        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write("commands", commands)

    def finish(self, context: Context):
        context.info("finish()")


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """Reads JPG bytes as RGB"""
    im = Image.open(io.BytesIO(image_data))
    im = im.convert("RGB")
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data

def main():
    node = PytorchRLBaseline()
    protocol = protocol_agent_DB20_timestamps
    wrap_direct(node=node, protocol=protocol)


if __name__ == "__main__":
    main()
