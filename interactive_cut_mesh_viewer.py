"""
Example of an extension of the WasdViewer class.

WasdViewerCutMesh takes in a mesh that is regularly clipped by a plane normal to the X axis.
Plane can be lowered/raised using the l/p buttons.
"""

import keyboard  # pip install keyboard
import numpy as np
import pyvista
from wasd_movement import WasdViewer


class CutMeshState:
    def __init__(self, mesh, plotter, increment, init_max_val):
        self.mesh = mesh
        self.plotter = plotter
        self.increment = increment
        self.max_val = init_max_val
        self.update()
        #
        self.queries = []

    def update(self):
        self.plotter.clear()
        mask = np.logical_or(self.mesh.points[:, 0] < self.max_val, self.mesh.points[:, 0] > 100)
        # cut_mesh = self.mesh.extract_points(mask, adjacent_cells=False)
        cut_mesh = self.mesh.extract_points(mask)
        self.plotter.add_mesh(cut_mesh, color='w', show_edges=True)

    def process(self):
        if len(self.queries) > 0:
            for val in self.queries:
                self.max_val += val
            self.queries.clear()
            self.update()

    def inc_query(self, _):
        self.queries.append(+self.increment)

    def dec_query(self, _):
        self.queries.append(-self.increment)


class WasdViewerCutMesh(WasdViewer):
    def __init__(self, plotter: pyvista.Plotter, mesh):
        super().__init__(plotter)
        self.cut_mesh_state = CutMeshState(mesh, plotter, increment=0.005, init_max_val=0.0)
        # lower_cut_plane_key
        keyboard.on_press_key("l", self.cut_mesh_state.dec_query)
        # raise cut plane key
        keyboard.on_press_key("p", self.cut_mesh_state.inc_query)

    def update_inner(self):
        super().update_inner()
        self.cut_mesh_state.process()
