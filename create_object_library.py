"""
note: only required if you want to rebuild the scenarios from scratch.

this script takes the meshes and creates required vhacd and urdf files, so that
they can be used within the simulation environment.
it processes both the obstacles created by create_obstacle_mesh.py and the
YCB objects, which are the target objects.

usage: python create_object_library.py
outputs will be written to ./object_library/vhacd/ and ./object_library/urdf/
the object library will be stored in ./object_library/object_library.yaml
"""
import os
import burg_toolkit as burg

# initialising object library
lib = burg.ObjectLibrary('jogramop')
lib.filename = 'object_library/object_library.yaml'

# describe object types and put into library
ycb_objects = ['004_sugar_box', '011_banana', '044_flat_screwdriver', '056_tennis_ball']
ycb_masses = [0.4, 0.2, 0.2, 0.2]

custom_objs = [
    'gate_0_54', 'gate_0_62', 'gate_0_7', 'gate_0_78', 'gate_0_86',
    'open_shelf', 'pedestal', 'table', 'wall_segment_80']
custom_masses = [0.0] * len(custom_objs)

object_ids = [*ycb_objects, *custom_objs]
masses = [*ycb_masses, *custom_masses]

mesh_fns = [os.path.join('object_library/meshes/', f'{mesh_name}.obj') for mesh_name in object_ids]
for obj_id, mesh_fn, mass in zip(object_ids, mesh_fns, masses):
    lib[obj_id] = burg.ObjectType(obj_id, mesh_fn=mesh_fn, mass=mass)

# create additional files and properties.
lib.generate_vhacd_files()
lib.generate_urdf_files()
lib.compute_stable_poses()

# print details and save to file
lib.print_details()
lib.to_yaml()
