"""
note: only required if you want to rebuild the scenarios from scratch.

this script creates the meshes for the environment.
usage: python create_obstacle_mesh.py
outputs will be written to ./object_library/meshes/
"""
import open3d as o3d
import burg_toolkit as burg
burg.visualization.configure_visualizer_mode(burg.visualization.VisualizerMode.CONFIGURABLE_VIEWER)


def create_u_shape(thickness, total_width, depth, height):
    leg1 = o3d.geometry.TriangleMesh.create_box(width=thickness, height=depth, depth=height)
    leg2 = o3d.geometry.TriangleMesh.create_box(width=thickness, height=depth, depth=height)
    leg2.translate([total_width - thickness, 0.0, 0.0])
    top = o3d.geometry.TriangleMesh.create_box(width=total_width, height=depth, depth=thickness)
    top.translate([0.0, 0.0, height])
    u_shape = burg.util.merge_o3d_triangle_meshes([leg1, leg2, top])
    return u_shape


# **** create shapes
# wall segment
wall_segment50 = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.1, depth=1.0)
wall_segment80 = o3d.geometry.TriangleMesh.create_box(width=0.8, height=0.1, depth=1.0)

# (inverted) u shapes
table = create_u_shape(thickness=0.1, total_width=2.0, depth=0.8, height=0.4)
open_shelf = create_u_shape(thickness=0.03, total_width=0.4 + 2*0.03, depth=0.5, height=0.4)
gate_opening_sizes = [0.54, 0.62, 0.70, 0.78, 0.86]
gates = []
for size in gate_opening_sizes:
    gate = create_u_shape(thickness=(2.0 - size)/2, total_width=2.0, depth=0.1, height=size)
    gates.append(gate)

# pedestal
pedestal = o3d.geometry.TriangleMesh.create_box(width=0.4, height=0.4, depth=0.5)

# **** save shapes to file
o3d.io.write_triangle_mesh('object_library/meshes/wall_segment_80.obj', wall_segment80)
o3d.io.write_triangle_mesh('object_library/meshes/table.obj', table)
o3d.io.write_triangle_mesh('object_library/meshes/open_shelf.obj', open_shelf)
for gate, size in zip(gates, gate_opening_sizes):
    o3d.io.write_triangle_mesh(f'object_library/meshes/gate_{str(size).replace(".", "_")}.obj', gate)
o3d.io.write_triangle_mesh('object_library/meshes/pedestal.obj', pedestal)

