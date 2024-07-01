"""
given the created scenarios, this script will produce the files that
are used by the cpp-implementation of the planners.
they can be found at: https://github.com/Hartvi/jogramop-planners

note that there is a slight change in convention:
- cpp planners expect all obstacles (scene objects) to be merged in a single .obj file
- cpp planners expect robot to be at origin (need to transform scene objects and grasps accordingly)
- (some) cpp planners expect a list of IK solutions, which we provide as well

usage: python create_cpp_export.py
outputs will be written to ./scenarios/<scenario-id>/export/
"""
import os
import copy
import numpy as np
import open3d as o3d
import burg_toolkit as burg
from scenario import Scenario
from util import SCENARIO_IDS


def export_for_cpp(scenario):
    # get scenario path from scene_fn, add export folder
    out_dir = os.path.join(os.path.dirname(scenario.scene_fn), 'export')
    burg.io.make_sure_directory_exists(out_dir)

    # we need to get the VHACD mesh for each object instance and merge them together into one obj file
    # cpp planners expect a single obj file as collision model
    def get_vhacd_for_obj_instance(obj):
        assert obj.object_type.vhacd_fn is not None, 'object type does not have a VHACD'
        vhacd_mesh = burg.io.load_mesh(mesh_fn=obj.object_type.vhacd_fn)
        if obj.object_type.scale is not None and obj.object_type.scale != 1.0:
            raise NotImplementedError('did not implement scale feature for VHACD export')

        vhacd_mesh.transform(obj.pose)
        return vhacd_mesh

    # save all scene objects (excl plane) to one obj file
    meshes = []
    for obj in [*scenario.scene.objects, *scenario.scene.bg_objects]:
        meshes.append(get_vhacd_for_obj_instance(obj))

    # merge them
    vertices = np.empty(shape=(0, 3), dtype=np.float64)
    triangles = np.empty(shape=(0, 3), dtype=int)
    for mesh in meshes:
        v = np.asarray(mesh.vertices)  # float list (n, 3)
        t = np.asarray(mesh.triangles)  # int list (n, 3)
        t += len(vertices)  # triangles reference the vertex index
        vertices = np.concatenate([vertices, v])
        triangles = np.concatenate([triangles, t])

    # finally create the merged mesh
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles))

    # in cpp implementation, robot is at origin, so we need to transform the scene accordingly
    tf = np.linalg.inv(scenario.robot_pose)
    mesh.transform(tf)

    # and grasp as well
    grasp_poses = copy.deepcopy(scenario.gs.poses)
    export_gs = burg.GraspSet.from_poses(grasp_poses)
    export_gs.transform(tf)

    # save to files
    burg.io.save_mesh(os.path.join(out_dir, 'obstacles.obj'), mesh)
    grasps = export_gs.poses
    grasps = grasps.reshape((grasps.shape[0], -1))
    np.savetxt(os.path.join(out_dir, 'grasps.csv'), grasps, delimiter=',')

    # save robot home configuration as well
    home_conf = np.array(
        [0.0, 0.0, -0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315,
         0.029840531355804868, 1.5411935298621688, 0.7534486589746342])
    np.savetxt(os.path.join(out_dir, 'robot_start_conf.csv'), home_conf, delimiter=',')

    # finally, get IK solutions and save them as well
    ik_solutions = scenario.get_ik_solutions()
    np.savetxt(os.path.join(out_dir, 'grasp_IK_solutions.csv'), ik_solutions, delimiter=',')


if __name__ == '__main__':
    for i in SCENARIO_IDS:
        print(f'********** SCENARIO {i:03d} **********')
        s = Scenario(i)
        export_for_cpp(s)
