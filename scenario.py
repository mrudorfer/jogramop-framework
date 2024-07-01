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

    def get_ik_solutions(self):
        """

        """
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
        return ik_solutions


if __name__ == '__main__':
    # showing the available scenarios
    from visualization import show_scenario

    for i in util.SCENARIO_IDS:
        print(f'********** SCENARIO {i:03d} **********')
        s = Scenario(i)
        s.select_n_grasps(10)
        show_scenario(s)
