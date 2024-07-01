"""
note: only required if you want to rebuild the scenarios from scratch.

this script assumes the object library is available and will
arrange the objects to create scenarios with different difficulty levels.

usage: python create_scenario.py
outputs will be written to ./scenarios/<scenario-id>
"""

import os
import numpy as np
import burg_toolkit as burg
from scipy.spatial.transform import Rotation as R


def create_scene_with_vertical_gap(object_library, gap_width=0.4, target_object='004_sugar_box', rotate_z=90):
    scene = burg.Scene(ground_area=(2, 2))

    # build walls - center of the gap is at x=1
    pose_wall_1 = np.eye(4)
    pose_wall_1[:3, 3] = [1 - gap_width/2 - 0.8, 0.5, 0]
    wall_1 = burg.ObjectInstance(object_library['wall_segment_80'], pose_wall_1)
    scene.bg_objects.append(wall_1)

    pose_wall_2 = np.eye(4)
    pose_wall_2[:3, 3] = [1 + gap_width/2, 0.5, 0]
    wall_2 = burg.ObjectInstance(object_library['wall_segment_80'], pose_wall_2)
    scene.bg_objects.append(wall_2)

    # put object in some stable pose, but rotate around z
    obj_type = object_library[target_object]
    angle = rotate_z * np.pi/180.
    tf_rot = np.eye(4)
    tf_rot[:3, :3] = R.from_rotvec(angle * np.array([0, 0, 1])).as_matrix()
    pose_obj = tf_rot @ obj_type.stable_poses[4][1]
    pose_obj[:2, 3] = [1, 0.75]
    obj = burg.ObjectInstance(obj_type, pose_obj)
    scene.objects.append(obj)

    return scene


def create_scene_with_obj_under_table(object_library, table_distance=0.6, target_object='056_tennis_ball',
                                      target_depth=0.2):
    scene = burg.Scene(ground_area=(2, 2))

    table_pose = np.eye(4)
    table_pose[1, 3] = table_distance
    table = burg.ObjectInstance(object_library['table'], table_pose)
    scene.bg_objects.append(table)

    target_obj_type = object_library[target_object]
    target_obj_pose = target_obj_type.stable_poses.poses[0].copy()
    target_x = 1
    target_y = table_distance + target_depth
    target_obj_pose[0, 3] += target_x
    target_obj_pose[1, 3] += target_y
    obj = burg.ObjectInstance(target_obj_type, target_obj_pose)
    scene.objects.append(obj)

    return scene


def create_scene_with_shelf_on_table(object_library, table_distance=0.6, target_object='044_flat_screwdriver',
                                     target_depth=0.15):
    scene = burg.Scene(ground_area=(2, 2))

    table_pose = np.eye(4)
    table_pose[1, 3] = table_distance
    table = burg.ObjectInstance(object_library['table'], table_pose)
    scene.bg_objects.append(table)

    shelf_pose = np.eye(4)
    shelf_pose[0, 3] = 1 - 0.23  # centered at 1, subtract half the width
    shelf_pose[1, 3] = table_distance
    shelf_pose[2, 3] = 0.5  # on top of table surface
    shelf = burg.ObjectInstance(object_library['open_shelf'], shelf_pose)
    scene.bg_objects.append(shelf)

    target_obj_type = object_library[target_object]
    target_obj_pose = target_obj_type.stable_poses.poses[0].copy()
    target_x = 1
    target_y = table_distance + target_depth
    target_obj_pose[0, 3] += target_x
    target_obj_pose[1, 3] += target_y
    target_obj_pose[2, 3] += 0.5
    obj = burg.ObjectInstance(target_obj_type, target_obj_pose)
    scene.objects.append(obj)

    return scene


def create_scene_with_gate(object_library, gate_distance=0.4, pedestal_distance_from_gate=0.6,
                           target_object='011_banana', gate_object='gate_0_6', target_depth=0.15):
    scene = burg.Scene(ground_area=(2, 2))

    gate_pose = np.eye(4)
    gate_pose[1, 3] = gate_distance  # all gates are 2.0m wide, so they should automatically be centered
    gate = burg.ObjectInstance(object_library[gate_object], gate_pose)
    scene.bg_objects.append(gate)

    pedestal_pose = np.eye(4)
    pedestal_pose[0, 3] = 1 - 0.2  # centered at 1, subtract half the width
    pedestal_pose[1, 3] = gate_distance + 0.1 + pedestal_distance_from_gate  # 0.1 is the thickness of the gate
    pedestal = burg.ObjectInstance(object_library['pedestal'], pedestal_pose)
    scene.bg_objects.append(pedestal)

    target_obj_type = object_library[target_object]
    target_obj_pose = target_obj_type.stable_poses.poses[0].copy()
    target_x = 1
    target_y = gate_distance + 0.1 + pedestal_distance_from_gate + target_depth
    target_obj_pose[0, 3] += target_x
    target_obj_pose[1, 3] += target_y
    target_obj_pose[2, 3] += 0.5
    obj = burg.ObjectInstance(target_obj_type, target_obj_pose)
    scene.objects.append(obj)

    return scene


def main():
    burg.visualization.configure_visualizer_mode(burg.visualization.VisualizerMode.CONFIGURABLE_VIEWER)
    lib = burg.ObjectLibrary.from_yaml('object_library/object_library.yaml')

    # scene type 1: screwdriver in shelf on table
    depth = [0.15, 0.25, 0.35, 0.45, 0.55]
    ids = [11, 12, 13, 14, 15]

    for scene_id, target_depth in zip(ids, depth):
        scene = create_scene_with_shelf_on_table(lib, target_depth=target_depth)
        burg.visualization.show_geometries([scene])
        save_dir = f'scenarios/{scene_id:03d}/'
        burg.io.make_sure_directory_exists(save_dir)
        scene.to_yaml(os.path.join(save_dir, 'scene.yaml'), lib)

    # scene type 2: ball under table
    depth = [0.2, 0.3, 0.4, 0.5, 0.6]
    ids = [21, 22, 23, 24, 25]

    for scene_id, target_depth in zip(ids, depth):
        scene = create_scene_with_obj_under_table(lib, target_depth=target_depth)
        burg.visualization.show_geometries([scene])
        save_dir = f'scenarios/{scene_id:03d}/'
        burg.io.make_sure_directory_exists(save_dir)
        scene.to_yaml(os.path.join(save_dir, 'scene.yaml'), lib)

    # scene type 3: grasp box through narrow gap
    gap_widths = [0.64, 0.56, 0.48, 0.4, 0.32]
    ids = [31, 32, 33, 34, 35]

    for scene_id, gap_width in zip(ids, gap_widths):
        scene = create_scene_with_vertical_gap(lib, gap_width=gap_width)
        burg.visualization.show_geometries([scene])
        save_dir = f'scenarios/{scene_id:03d}/'
        burg.io.make_sure_directory_exists(save_dir)
        scene.to_yaml(os.path.join(save_dir, 'scene.yaml'), lib)

    # scene type 4: banana on pedestal behind gate
    gate_type = ['gate_0_86', 'gate_0_78', 'gate_0_7', 'gate_0_62', 'gate_0_54']
    ids = [41, 42, 43, 44, 45]

    for scene_id, gate in zip(ids, gate_type):
        scene = create_scene_with_gate(lib, gate_object=gate)
        burg.visualization.show_geometries([scene])
        save_dir = f'scenarios/{scene_id:03d}/'
        burg.io.make_sure_directory_exists(save_dir)
        scene.to_yaml(os.path.join(save_dir, 'scene.yaml'), lib)


if __name__ == '__main__':
    main()
