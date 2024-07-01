"""
note: only required if you want to rebuild the grasps for the scenarios.

this script will, for each scenario, create grasp annotations for the target object.

usage: python create_graspset.py
outputs will be written to ./scenarios/<scenario-id>/grasps.npy
"""
import os
import numpy as np
import burg_toolkit as burg


def get_scene_files(skip_scenes=None):
    # gives paths to all scene files that can be found in the scenario directory
    scenes = os.listdir('scenarios')
    files = []
    for scene in scenes:
        if skip_scenes is not None and scene in skip_scenes:
            continue
        files.append(os.path.join('scenarios', scene, 'scene.yaml'))

    print(files)
    return files


def create_grasps(scene_file):
    """
    uses antipodal grasp sampling from BURG toolkit to create grasp annotations.
    we first sample 2000 grasps for the target object, then use a simplified gripper model
    to filter out grasps that are in collision.
    of the remaining grasp candidates, we randomly sample 200 to keep as annotations.
    """
    scene, lib, _ = burg.Scene.from_yaml(scene_file)
    target_obj = scene.objects[0]
    save_fn = os.path.join(os.path.dirname(scene_file), 'grasps.npy')
    keep = 200  # max number of grasps to keep (random selection)
    gripper_width = 0.08  # suitable for franka panda gripper

    grasp_sampler = burg.sampling.AntipodalGraspSampler(
        n_orientations=7,
        only_grasp_from_above=True,
        no_contact_below_z=None
    )
    print('sampling...')
    grasp_candidates, contacts = grasp_sampler.sample(
        target_obj,
        n=2000,
        max_gripper_width=gripper_width,  # franka panda gripper width
    )
    print(f'sampled {len(grasp_candidates)} grasps.')
    print('checking collisions...')
    gripper = burg.gripper.TwoFingerGripperVisualisation(opening_width=gripper_width)
    collisions = grasp_sampler.check_collisions(
        grasp_candidates,
        scene,
        gripper.mesh,
        with_plane=True
    )
    grasps = grasp_candidates[collisions == 0]
    print(f'{len(grasps)} collision-free grasps remaining.')

    if len(grasps) > keep:
        print(f'sampling {keep} grasps from remaining set.')
        keep_indices = np.random.choice(len(grasps), keep, replace=False)
        grasps = grasps[keep_indices]

    print(f'saving to {save_fn}')
    np.save(save_fn, grasps.poses)
    # burg.visualization.show_grasp_set([target_obj], grasps, gripper)


if __name__ == '__main__':
    skip_scene_ids = None  # if you want to skip particular scenes, e.g. ['11', '12', '13']
    scene_fns = get_scene_files(skip_scene_ids)
    for scene_fn in scene_fns:
        create_grasps(scene_fn)
