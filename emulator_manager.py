
from pyboy import PyBoy
from skimage.transform import downscale_local_mean
from pyboy.utils import WindowEvent
import numpy as np

class EmulatorManager:
    def __init__(self, config):
        # Initialize PyBoy with the configuration path and window settings
        self.pyboy = PyBoy(config["gb_path"], window="null" if config["headless"] else "SDL2")
        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)
        
        # Define valid actions that can be sent to the emulator
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START
        ]

    def run_action(self, action, act_freq, render_screen):
        # Press and hold the action
        self.pyboy.send_input(self.valid_actions[action])
        self.pyboy.tick(8, render_screen)
        
        # Release the action after press_step
        self.pyboy.send_input(self.valid_actions[action] + 8)  # Releases input
        self.pyboy.tick(act_freq - 9, render_screen)

    def read_memory(self, address):
        # Read a specific memory address
        return self.pyboy.memory[address]

    def get_screen_pixels(self, reduce_res=True):
        # Get the screen pixels from the emulator, optionally downscaling for efficiency
        screen = self.pyboy.screen.ndarray[:, :, :1]
        if reduce_res:
            screen = downscale_local_mean(screen, (2, 2, 1)).astype(np.uint8)
        return screen

