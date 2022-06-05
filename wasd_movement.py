"""
Module for allowing wasd movement. Example script:

import pyvista as pv
from pyvista import examples
import wasd_movement

dataset = examples.download_bunny_coarse()
p = pv.Plotter()
p.add_mesh(dataset, style='wireframe', color='blue', label='Input')
viewer = wasd_movement.WasdViewer(p)
viewer.show()
"""

import keyboard  # pip install keyboard
import pynput  # pip install pynput
import time
import numpy as np
from math import cos, sin, pi, exp
import pyvista


class ExitViewer(Exception):
    pass


class SpeedState:
    def __init__(self, min_val=-10, max_val=10):
        self.val = 0  # internal variable that is clamped to [self.min_val, self.max_val]
        self.min_val = min_val
        self.max_val = max_val
        #
        self.speed = 1.0  # deduced from self.val via some function
        self.update_speed()

    def update_speed(self):
        self.speed = exp(self.val * 0.25)

    def inc(self, _):
        self.val = min(self.val + 1, self.max_val)
        self.update_speed()

    def dec(self, _d):
        self.val = max(self.val - 1, self.min_val)
        self.update_speed()


class WasdViewer:
    # Mouse movement is processed by getting the current cursor position and keeping track of its movement.
    # There will be issues with not being to turn further if the cursor is at the edge of the screen
    # -> hacky solution - regularly set the mouse cursor position to some fixed place
    MOUSE_BASE_POS = np.array([256.0, 256.0])
    MOUSE_SPEED = 0.05
    TRANSLATION_BASE_SPEED = np.array([1.0])
    A2R = 0.005555555 * pi  # angle to radians constant, e.g. 90 -> 0.5pi

    def __init__(self, plotter: pyvista.Plotter):
        self.plotter = plotter
        self.camera = self.plotter.camera
        self.camera.position = (0.0, 0.0, 1.0)
        self.camera.clipping_range = (0.00001, 100000.0)

        # controls
        self.W_KEY = "w"
        self.A_KEY = "a"
        self.S_KEY = "s"
        self.D_KEY = "d"
        self.EXIT_KEY = "q"
        self.DOWN_KEY = "ctrl"
        self.UP_KEY = "space"
        self.INC_SPEED_KEY = "r"
        self.DEC_SPEED_KEY = "f"

        # initialize certain vars
        self.plotter.iren.process_events = lambda: None  # turn off default events by overriding RenderWindowInteractor.process_events
        self.plotter.ren_win.HideCursor()  # hide cursor using vtkRenderWindow method
        self.speed_state = SpeedState()
        keyboard.on_press_key(self.INC_SPEED_KEY, self.speed_state.inc)
        keyboard.on_press_key(self.DEC_SPEED_KEY, self.speed_state.dec)

        # state vars
        self.current_time = 0.0
        self.mouse = pynput.mouse.Controller()
        self.up_angle = 0.0  # pitch
        self.hor_angle = 0.0  # yaw

    def show(self):
        self.plotter.show(interactive_update=True, interactive=False)
        self.mouse.position = (self.MOUSE_BASE_POS[0], self.MOUSE_BASE_POS[1])
        self.current_time = time.perf_counter()
        self.update_loop()

    def update_loop(self):
        while True:
            try:
                self.update_inner()
            except ExitViewer:
                break
            self.plotter.update()

    def update_inner(self):
        if keyboard.is_pressed(self.EXIT_KEY):
            raise ExitViewer

        # update delta time
        new_time = time.perf_counter()
        time_delta = new_time - self.current_time
        self.current_time = new_time

        # 1. keyboard movement
        forward = self.get_current_view_dir(self.up_angle, self.hor_angle)
        up = self.get_current_up_dir(self.up_angle, self.hor_angle)
        right = self.normalize_vector(np.cross(forward, up))
        local_dir = (0.0, 0.0, 0.0)
        # region keyboard input polling
        some_input = False
        if keyboard.is_pressed(self.W_KEY):
            some_input = True
            local_dir += +forward
        if keyboard.is_pressed(self.S_KEY):
            some_input = True
            local_dir += -forward
        if keyboard.is_pressed(self.A_KEY):
            some_input = True
            local_dir += -right
        if keyboard.is_pressed(self.D_KEY):
            some_input = True
            local_dir += +right
        if keyboard.is_pressed(self.DOWN_KEY):
            some_input = True
            local_dir += -np.array([0.0, 1.0, 0.0])
        if keyboard.is_pressed(self.UP_KEY):
            local_dir += np.array([0.0, 1.0, 0.0])
        if some_input:
            local_dir = self.normalize_vector(local_dir)
        # endregion

        # 2. mouse movement
        mouse_delta = (self.mouse.position - self.MOUSE_BASE_POS) * self.MOUSE_SPEED
        self.mouse.position = self.MOUSE_BASE_POS
        # update camera angles
        self.hor_angle -= mouse_delta[0]
        self.up_angle -= mouse_delta[1]
        self.up_angle = min(max(self.up_angle, -90.0), 90.0)

        # 3. set new camera params
        new_pos = self.camera.position + local_dir * self.TRANSLATION_BASE_SPEED * self.speed_state.speed * time_delta
        new_focal_point = new_pos + self.get_current_view_dir(self.up_angle, self.hor_angle)
        new_up = self.get_current_up_dir(self.up_angle, self.hor_angle)
        self.plotter.camera_position = (tuple(new_pos), tuple(new_focal_point), tuple(new_up))
        return True

    @classmethod
    def get_current_view_dir(cls, i, j):
        """Maps i,j angles to camera forward dir"""
        alpha = i * cls.A2R
        beta = (j + 180) * cls.A2R
        return np.array([cos(alpha) * sin(beta), sin(alpha), cos(alpha) * cos(beta)])

    @classmethod
    def get_current_up_dir(cls, i, j):
        """Maps i,j angles to camera up dir"""
        alpha = (i + 90) * cls.A2R
        beta = (j + 180) * cls.A2R
        return np.array([cos(alpha) * sin(beta), sin(alpha), cos(alpha) * cos(beta)])

    @staticmethod
    def normalize_vector(v):
        d = np.dot(v, v)
        if d == 0.0:
            return np.array([1.0, 0.0, 0.0])
        else:
            return v / np.sqrt(d)
