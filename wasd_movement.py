"""
Module for allowing wasd movement. Example script:

import pyvista as pv
from pyvista import examples
import wasd_movement

dataset = examples.download_bunny_coarse()
p = pv.Plotter()
p.add_mesh(dataset, style='wireframe', color='blue', label='Input')
wasd_movement.open_with_wasd(p)
"""

import keyboard  # pip install keyboard
import pynput  # pip install pynput
import time
import numpy as np
from math import cos, sin, pi, exp
import pyvista


W_KEY = "w"
A_KEY = "a"
S_KEY = "s"
D_KEY = "d"
EXIT_KEY = "q"
DOWN_KEY = "ctrl"
UP_KEY = "space"
INC_SPEED_KEY = "r"
DEC_SPEED_KEY = "f"

# Mouse movement is processed by getting the current cursor position and keeping track of its movement.
# There will be issues with not being to turn further if the cursor is at the edge of the screen
# -> hacky solution - regularly set the mouse cursor position to some fixed place
MOUSE_BASE_POS = np.array([256.0, 256.0])
MOUSE_SPEED = 0.05
TRANSLATION_BASE_SPEED = np.array([1.0])
A2R = 0.005555555 * pi  # angle to radians constant, e.g. 90 -> 0.5pi


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


def get_current_view_dir(i, j):
    """Maps i,j angles to camera forward dir"""
    alpha = i * A2R
    beta = (j + 180) * A2R
    return np.array([cos(alpha) * sin(beta), sin(alpha), cos(alpha) * cos(beta)])


def get_current_up_dir(i, j):
    """Maps i,j angles to camera up dir"""
    alpha = (i + 90) * A2R
    beta = (j + 180) * A2R
    return np.array([cos(alpha) * sin(beta), sin(alpha), cos(alpha) * cos(beta)])


def normalize_vector(v):
    d = np.dot(v, v)
    if d == 0.0:
        return np.array([1.0, 0.0, 0.0])
    else:
        return v / np.sqrt(d)


def open_with_wasd(p: pyvista.Plotter):
    """
    Updates and renders input pyvista.Plotter.
    Uses movement and rotation using WASD keys and mouse like in fps games
    """
    # initialize camera vars
    camera = p.camera
    camera.position = (0.0, 0.0, 1.0)
    camera.clipping_range = (0.00001, 100000.0)
    # get mouse controller and set up default view angles
    mouse = pynput.mouse.Controller()
    up_angle = 0.0  # pitch
    hor_angle = 0.0  # yaw
    # turn off default events by overriding RenderWindowInteractor.process_events
    p.iren.process_events = lambda: None
    # hide cursor using vtkRenderWindow method
    p.ren_win.HideCursor()
    # create speed state
    speed_state = SpeedState()
    keyboard.on_press_key(INC_SPEED_KEY, speed_state.inc)
    keyboard.on_press_key(DEC_SPEED_KEY, speed_state.dec)

    # region update loop
    p.show(interactive_update=True, interactive=False)
    mouse.position = (MOUSE_BASE_POS[0], MOUSE_BASE_POS[1])
    current_time = time.perf_counter()
    while True:
        if keyboard.is_pressed(EXIT_KEY):
            break

        # update delta time
        new_time = time.perf_counter()
        time_delta = new_time - current_time
        current_time = new_time

        # 1. keyboard movement
        # deduce local directions of camera
        forward = get_current_view_dir(up_angle, hor_angle)
        up = get_current_up_dir(up_angle, hor_angle)
        right = normalize_vector(np.cross(forward, up))
        # process keyboard input
        local_dir = (0.0, 0.0, 0.0)
        some_input = False
        if keyboard.is_pressed(W_KEY):
            some_input = True
            local_dir += +forward
        if keyboard.is_pressed(S_KEY):
            some_input = True
            local_dir += -forward
        if keyboard.is_pressed(A_KEY):
            some_input = True
            local_dir += -right
        if keyboard.is_pressed(D_KEY):
            some_input = True
            local_dir += +right
        if keyboard.is_pressed(DOWN_KEY):
            some_input = True
            local_dir += -np.array([0.0, 1.0, 0.0])
        if keyboard.is_pressed(UP_KEY):
            local_dir += np.array([0.0, 1.0, 0.0])
        if some_input:
            local_dir = normalize_vector(local_dir)

        # 2. mouse movement
        # deduce mouse offset and reset pos
        mouse_delta = (mouse.position - MOUSE_BASE_POS) * MOUSE_SPEED
        mouse.position = MOUSE_BASE_POS
        # update camera angles
        hor_angle -= mouse_delta[0]
        up_angle -= mouse_delta[1]
        up_angle = min(max(up_angle, -90.0), 90.0)

        # 3. set new camera params
        new_pos = camera.position + local_dir * TRANSLATION_BASE_SPEED * speed_state.speed * time_delta
        new_focal_point = new_pos + get_current_view_dir(up_angle, hor_angle)
        new_up = get_current_up_dir(up_angle, hor_angle)
        p.camera_position = (tuple(new_pos), tuple(new_focal_point), tuple(new_up))

        p.update()
    # endregion
