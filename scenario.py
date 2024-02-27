import copy
import os
import numpy as np
import open3d as o3d
import burg_toolkit as burg

import simulation
from simulation import GraspingSimulator, FrankaRobot
import util


class Scenario:
    """
    Describes a scenario, including:
    - scene with objects and obstacles
    - set of grasp candidates
    - simulator
    - robot
    """
    def __init__(self, scenario_id, robot_pose=None, with_platform=True):
        # load scene
        self.id = scenario_id
        self.scene_fn = os.path.join(util.SCENARIO_DIR, f'{scenario_id:03d}', 'scene.yaml')
        self.scene, lib, _ = burg.Scene.from_yaml(self.scene_fn)
        print(f'loaded scene with {len(self.scene.objects)} objects and {len(self.scene.bg_objects)} obstacles.')

        # fetch corresponding grasps
        grasps_fn = os.path.join(util.SCENARIO_DIR, f'{scenario_id:03d}', 'grasps.npy')
        grasps = np.load(grasps_fn, allow_pickle=True)
        # convert from BURG to franka frame
        grasps = grasps @ FrankaRobot.tf_grasp2ee
        self.gs = burg.GraspSet.from_poses(grasps)
        print(f'loaded {len(self.gs)} grasps')
        self.select_indices = None  # for grasp set sub-selection

        # robot pose
        self.robot_pose = robot_pose or self.default_robot_pose()
        self.with_platform = with_platform

    def filter_colliding_grasps(self):
        raise DeprecationWarning('this function is not used anymore since all grasps are collision-free')
        return

    @property
    def grasp_poses(self):
        if self.select_indices is None:
            return self.gs.poses
        return self.gs.poses[self.select_indices]

    def select_n_grasps(self, n=None, seed=None):
        # if n is None, will select all grasps again
        if n is None:
            print('erasing grasp selection. whole grasp set available again.')
            self.select_indices = None
            return

        if n > len(self.gs):
            raise ValueError('cannot select more grasps than available. n must be smaller than number of grasps')

        rng = np.random.default_rng(seed)
        self.select_indices = rng.choice(len(self.gs), n, replace=False)

    def get_robot_and_sim(self, with_gui=False) -> tuple[simulation.FrankaRobot, simulation.GraspingSimulator]:
        """
        Get the robot and simulator handles.
        """
        sim = GraspingSimulator(verbose=with_gui)
        sim.add_scene(self.scene)
        robot = FrankaRobot(sim, self.robot_pose, with_platform=self.with_platform)

        return robot, sim

    @staticmethod
    def default_robot_pose():
        # lift up a bit so we don't have collisions with ground (z=0.05)
        # x=1 so that it is in the middle of the scene and not on the corner
        # rotation so that it faces the scene
        pose = np.array([
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ])
        return pose

    def export_for_cpp(self):
        # get path from self.scene_fn
        out_dir = os.path.join(os.path.dirname(self.scene_fn), 'export')
        burg.io.make_sure_directory_exists(out_dir)

        def get_vhacd_for_obj_instance(obj):
            assert obj.object_type.vhacd_fn is not None, 'object type does not have a VHACD'
            vhacd_mesh = burg.io.load_mesh(mesh_fn=obj.object_type.vhacd_fn)
            if obj.object_type.scale is not None and obj.object_type.scale != 1.0:
                raise NotImplementedError('did not implement scale feature for VHACD export')

            vhacd_mesh.transform(obj.pose)
            return vhacd_mesh

        # save all scene objects (excl plane) to one obj file
        meshes = []
        for obj in [*self.scene.objects, *self.scene.bg_objects]:
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
        tf = np.linalg.inv(self.robot_pose)
        mesh.transform(tf)

        # and grasp as well
        grasp_poses = copy.deepcopy(self.gs.poses)
        export_gs = burg.GraspSet.from_poses(grasp_poses)
        export_gs.transform(tf)

        # check everything aligns by visualizing
        # burg.visualization.show_grasp_set([self.scene, mesh], export_gs, gripper=GM(0.1))

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

    def export_IK_solutions(self):
        # get path from self.scene_fn
        out_dir = os.path.join(os.path.dirname(self.scene_fn), 'export')
        burg.io.make_sure_directory_exists(out_dir)

        # get robot and sim and find IK for each grasp
        robot, sim = self.get_robot_and_sim()
        joint_limits = robot.arm_joint_limits()

        def in_joint_limits(joint_conf):
            return np.all(joint_limits[:, 0] <= joint_conf) and np.all(joint_conf <= joint_limits[:, 1])

        ik_solutions = []
        count = util.Timer()
        print('calculating IK solutions for all grasps')
        for g in range(len(self.gs)):
            count.start('calculate IK')
            target_pose = self.gs.poses[g]
            pos, orn = burg.util.position_and_quaternion_from_tf(target_pose, convention='pybullet')

            # do the actual IK
            target_conf = robot.inverse_kinematics(pos, orn, null_space_control=True)
            count.stop('calculate IK')
            count.start('perform checks')
            if target_conf is not None:
                # check for collisions
                robot.reset_arm_joints(target_conf)

                if not in_joint_limits(target_conf):
                    target_conf = None
                    count.count('IK solution not in joint limits')
                elif robot.in_self_collision():
                    target_conf = None
                    count.count('IK solution in self-collision')
                elif robot.in_collision():
                    target_conf = None
                    count.count('IK solution in collision')
                else:
                    count.count('IK solution is OK')
            else:
                count.count('IK solution too far away from goal')
            count.stop('perform checks')

            if target_conf is not None:
                ik_solutions.append(target_conf)

        count.print()
        ik_solutions = np.array(ik_solutions)
        np.savetxt(os.path.join(out_dir, 'grasp_IK_solutions.csv'), ik_solutions, delimiter=',')
        print('saved IK solutions to file')


if __name__ == '__main__':
    from visualization import show_scenario
    # burg.visualization.configure_visualizer_mode(burg.visualization.VisualizerMode.CONFIGURABLE_VIEWER)
    class GM:
        def __init__(self, size=0.01):
            self.mesh = burg.visualization.create_frame(size)

    for i in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45]:
        print(f'********** SCENARIO {i:03d} **********')
        s = Scenario(i)
        # s.select_n_grasps(10)
        # idx = s.select_indices
        # burg.visualization.show_grasp_set([s.scene], s.gs[idx], with_plane=True, gripper=GM(0.1))
        s.export_for_cpp()
        s.export_IK_solutions()
        # show_scenario(s)
